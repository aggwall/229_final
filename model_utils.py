import os
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
import yaml

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def __getattr__(self, item):
        return self.config.get(item)

def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=False),
        nn.LayerNorm(out_features),
        nn.GELU(),     
        nn.Dropout(p=p_drop),
    ) 

class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__()
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x

def lr_warmup(epoch, initial_lr=2e-6, final_lr=1e-5, warmup_epochs=100):
    '''
    Used when I'm loading in a pre-trained diffusion model for further training.
    Optional to use but has been helpful for me w.r.t. training stability.
    '''
    return (
        initial_lr + (final_lr - initial_lr) * epoch / warmup_epochs 
        if epoch < warmup_epochs 
        else final_lr
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_environment(num_threads):
    thread_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]
    for var in thread_vars:
        os.environ[var] = str(num_threads)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class CrossAttention(nn.Module):
    """
    https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/attention.py
    """
    def __init__(self, query_dim, context_dim=None, heads=16, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        with torch.autocast(enabled=False, device_type = 'cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class StableDiffEncoder(nn.Module):
    def __init__(
        self, 
        total_condition_dim: int, 
        condition_dim: int, 
        t_theta_type: str,
        num_xattn_layers: int, 
        device: str = 'cuda',
    ):
        """Helper function to encode conditions for Stable Diffusion."""
        super(StableDiffEncoder, self).__init__()
        self.t_theta_type = t_theta_type
        self.device = device

        if self.t_theta_type == 'linear':
            self.t_theta = nn.Sequential(
                full_block(total_condition_dim, condition_dim),     # Combining all conditions into one
                full_block(condition_dim, condition_dim),
                nn.Linear(condition_dim, condition_dim)
            )
        elif self.t_theta_type == 'cross_attn_plus_linear':
            self.t_theta = nn.ModuleList([
                CrossAttention(total_condition_dim, total_condition_dim, heads=16, dim_head=64, dropout=0.1)
                for _ in range(num_xattn_layers)  
            ])
            self.reduce_dim = nn.Linear(total_condition_dim, condition_dim)
        elif self.t_theta_type == 'cross_attn_plus_cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, condition_dim))
            self.total_condition_dim += condition_dim
            self.t_theta = nn.ModuleList([
                CrossAttention(self.total_condition_dim, self.total_condition_dim, heads=16, dim_head=64, dropout=0.1)
                for _ in range(num_xattn_layers)
            ])
        elif self.t_theta_type == "none":
            self.t_theta = nn.Identity()
        else:
            raise ValueError('Invalid t_theta_type: Specify an appropriate style of Stable Diffusion conditioning.')
    
    def forward(self, conds):
        theta = torch.cat(list(conds.values()), dim=-1)

        if self.t_theta_type == 'linear':
            theta = self.t_theta(theta)
        elif self.t_theta_type == 'cross_attn_plus_linear':
            for layer in self.t_theta:
                theta = layer(theta.unsqueeze(1), theta.unsqueeze(1)).squeeze(1)
            theta = self.reduce_dim(theta)
        elif self.t_theta_type == 'cross_attn_plus_cls':
            cls_tokens = self.cls_token.expand(theta.shape[0], -1, -1)
            theta = torch.cat([cls_tokens, theta.unsqueeze(1)], dim=-1)
            for layer in self.t_theta:
                theta = layer(theta, theta)
            theta = theta[:, 0, :self.condition_dim]    # only cls token
        elif self.t_theta_type == "none":
            pass

        return theta

class ClassifierMLP(nn.Module):
    def __init__(self, num_class, uce_dim=1280, time_dim=128, n_layers=6):
        super().__init__()
        self.t_embed_nn = TimeSiren(1, time_dim)

        self.layer0 = full_block(uce_dim + time_dim, uce_dim)
        self.layers = nn.ModuleList([full_block(uce_dim, uce_dim) for _ in range(n_layers)])

        self.out = nn.Linear(uce_dim, num_class, bias=True)

    def forward(self, uce, t):
        t_e = self.t_embed_nn(t.float())
        uce_w_t = torch.cat([uce, t_e], dim=-1)

        x = self.layer0(uce_w_t)
        for layer in self.layers:
            x = layer(x) + x

        x = self.out(x)
        return x
