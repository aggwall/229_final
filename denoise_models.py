import torch
import torch.nn as nn
from typing import Optional, Dict, List
from ucegen.helpers.model_utils import CrossAttention, full_block, TimeSiren, StableDiffEncoder

class DiffusionDenoiser(nn.Module):
    def __init__(
        self,
        cond_names: Optional[List[str]] = None,
        uce_dim: int = 1280,
        time_dim: int = 32,
        num_hidden_layers: int = 4,
        hidden_multiple: int = 6,
        num_xattn_layers: int = 2,
        cond_emb_type: Optional[str] = None,
        condition_combination: str = 'uncond',
        t_theta_type: str = 'linear',
        sieve: bool = True,
        cfg: bool = False,
        output_dim: Optional[int] = None,
        device: str = 'cuda',
    ):
        super(DiffusionDenoiser, self).__init__()

        self.combination = condition_combination
        self.condition_dim = uce_dim
        self.cond_names = cond_names
        self.sieve = sieve
        self.num_hidden_layers = num_hidden_layers
        self.hidden_multiple = hidden_multiple
        self.device = device
        self.cfg = cfg
        self.cond_emb_type = cond_emb_type
        
        self.time_encoder = TimeSiren(1, time_dim)
        self.n_input = uce_dim + time_dim
        self.n_hidden = self.n_input * self.hidden_multiple
        self.n_output = uce_dim if output_dim is None else output_dim    # useful to override sometimes
        
        if self.cond_names is None:
            assert condition_combination == 'uncond', "If no conditions are provided, the model must be unconditional."
        else:
            self.num_conds = len(self.cond_names)
            self.t_theta_type = t_theta_type
        
            if self.cond_emb_type == 'learned':
                for name in self.cond_names:
                    setattr(self, f'{name}_encoder', nn.Linear(self.condition_dim, self.condition_dim))
        
            self._setup_condition_combination(num_xattn_layers)
            
        # Need to do this at the end since n_input has been updated based on the combination method.
        self.layers = self._build_layers()
        
        # Evaluate at the end of init to see if the model is set up correctly
        print(f"\nDIMENSIONS: time_dim={time_dim}, n_input={self.n_input}, n_hidden={self.n_hidden}, n_output={self.n_output}")
        print(f"Num hidden layers: {self.num_hidden_layers}, hidden_multiple: {hidden_multiple}\n")
    
    def _build_layers(self):
        layers = nn.ModuleList()
        layers.append(full_block(self.n_input, self.n_hidden))
        for _ in range(1, self.num_hidden_layers + 1):
            input_dim = self.n_hidden + self.condition_dim + 1 if self.sieve else self.n_hidden
            layers.append(full_block(input_dim, self.n_hidden))
        layers.append(nn.Linear(self.n_hidden + self.condition_dim + 1, self.n_output))
        return layers
    
    def _setup_condition_combination(self, num_xattn_layers):
        if self.combination == 'concat':
            self.n_input += self.condition_dim * self.num_conds
        elif self.combination == 'cross_attn':
            for name in self.cond_names:
                setattr(self, f'{name}_attention', CrossAttention(query_dim=self.condition_dim, context_dim=self.condition_dim))
        elif 'stable_diff' in self.combination:
            if self.combination == 'stable_diff_concat':
                self.n_input += self.condition_dim
            elif self.combination == 'stable_diff_cross_attn':
                for i in range(1, self.num_hidden_layers + 1):
                    setattr(self, f'cross_attn_layer{i}', CrossAttention(query_dim=self.n_hidden, context_dim=self.condition_dim))
            else:
                raise ValueError('Invalid Stable Diffusion combination method')
            
            self.sd_encoder = StableDiffEncoder(
                self.condition_dim * self.num_conds, self.condition_dim, self.t_theta_type, num_xattn_layers, self.device
            )
        
        self.n_hidden = self.n_input * self.hidden_multiple
    
    def forward(
        self, 
        uce: torch.Tensor, 
        t: torch.Tensor, 
        conditions: Optional[Dict[str, torch.Tensor]] = None, 
        context_mask: torch.Tensor = None
    ) -> torch.Tensor:
        t_e = self.time_encoder(t.float())

        if conditions is not None:
            assert isinstance(conditions, dict), "Conditions must be a dictionary of condition names and values."
            if self.cond_emb_type == 'learned':
                for name in self.cond_names:
                    conditions[name] = getattr(self, f'{name}_encoder')(conditions[name])

            if 'stable_diff' in self.combination:
                theta = self.sd_encoder(conditions)
                if self.cfg:
                    uce, t_e, theta = self._apply_cfg_mask(uce, t_e, theta, context_mask)
                l1_input = torch.cat([uce, t_e, theta], dim=-1) if self.combination == 'stable_diff_concat' else torch.cat([uce, t_e], dim=-1)
            else:
                if len(conditions) == 1 and self.cfg:  # if there's only one condition, we can use the cfg mask
                    cond_name = list(conditions.keys())[0]
                    uce, t_e, conditions[cond_name] = self._apply_cfg_mask(uce, t_e, conditions[cond_name], context_mask)
                l1_input = self._combine_conditions(uce, t_e, conditions)
        else:
            assert self.combination == 'uncond', "If no conditions are provided, the model must be unconditional."
            l1_input = torch.cat([uce, t_e], dim=-1)

        nn1 = self.layers[0](l1_input)
        for i in range(1, self.num_hidden_layers + 1):
            nn = self.layers[i]
            if self.cfg and context_mask is None:       # if we're in inference mode
                t = t.repeat(2, 1)
            nn1 = nn(torch.cat([nn1 / 1.414, uce, t], dim=-1)) + nn1 / 1.414 if self.sieve else nn(nn1) + nn1
            if self.cfg and context_mask is None:
                t = t[:t.shape[0] // 2]
            if self.combination == 'stable_diff_cross_attn':
                uce_ca_input = nn1.unsqueeze(1)
                uce_ca_output = getattr(self, f'cross_attn_layer{i}')(uce_ca_input, theta.unsqueeze(1)).squeeze(1)
                nn1 = nn1 + uce_ca_output  # residual connections for cross attn
                
        if self.cfg and context_mask is None:
            t = t.repeat(2, 1)
        return self.layers[-1](torch.cat([nn1, uce, t], dim=-1))

    def _apply_cfg_mask(self, uce, t_e, cond, context_mask):
        """Apply the Classifier-Free Guidance mask."""
        if context_mask is not None:  # happens during training
            cond *= context_mask.unsqueeze(1).to(self.device)
        else:  # happens during inference
            uce = uce.repeat(2, 1)
            t_e = t_e.repeat(2, 1)
            cond = cond.repeat(2, 1).to(self.device)
            context_mask = torch.zeros_like(cond).to(self.device)
            context_mask[:cond.shape[0] // 2] = 1.0  # makes first set of embeddings ones
            cond = cond * context_mask 
        return uce, t_e, cond

    def _combine_conditions(self, x, t_e, conds=None):
        if self.combination == 'concat':
            return_val = torch.cat([x, t_e, *list(conds.values())], dim=-1)
        elif self.combination == 'cross_attn':
            for name in self.cond_names:
                x = getattr(self, f'{name}_attention')(x.unsqueeze(1), conds[name].unsqueeze(1)).squeeze(1)
            return_val = torch.cat([x, t_e], dim=-1)
        else:
            raise ValueError('Invalid combination method for conditions.')

        if self.cfg:
            cond_name = list(conds.keys())[0]
            conds[cond_name] = conds[cond_name][:len(conds[cond_name]) // 2]
        return return_val


class DiffusionTransformer(nn.Module):
    """
    A transformer-based diffusion model that processes UCE embeddings with conditions.
    
    This model concatenates UCE embeddings, time embeddings, and optional condition embeddings
    into a sequence, processes them through a transformer, and outputs denoised UCE embeddings.
    
    Args:
        uce_dim (int): Dimension of input UCE embeddings (default: 1280)
        model_dim (int): Internal dimension for the transformer (default: 512)
        num_layers (int): Number of transformer encoder layers (default: 4)
        num_heads (int): Number of attention heads in the transformer (default: 4)
        cond_names_and_dims (Dict[str, int], optional): Dictionary mapping condition names to their dimensions
        cfg (bool): Whether to use Classifier-Free Guidance (default: False)
        learn_pos_embed (bool): Whether to use learnable position embeddings (default: False)
    """
    def __init__(
        self,
        uce_dim: int = 1280,
        model_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        cond_names_and_dims: Optional[Dict[str, int]] = None,
        cfg: bool = False,
        learn_pos_embed: bool = False,
        device: str = 'cuda',
    ):
        super(DiffusionTransformer, self).__init__()

        self.model_dim = model_dim
        self.cond_names_and_dims = cond_names_and_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg = cfg
        self.learn_pos_embed = learn_pos_embed
        self.device = device
        
        self.make_uce_embedding = nn.Sequential(
            nn.Linear(uce_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
        self.make_time_embedding = TimeSiren(1, model_dim)    # sinusoidal encoding

        # Create linear projections for each condition
        if cond_names_and_dims is not None:
            print(f"Using {len(self.cond_names_and_dims)} conditions.")
            for name, dim in self.cond_names_and_dims.items():
                setattr(self, f'make_{name}_embedding', nn.Linear(dim, model_dim))
        
        # Optional learnable token type embeddings for each position in sequence
        if self.learn_pos_embed:
            num_tokens = 2 + (len(self.cond_names_and_dims) if cond_names_and_dims else 0)  # UCE + time + conditions
            self.token_type_embeddings = nn.Parameter(torch.randn(num_tokens, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            activation='gelu', 
            batch_first=True,
            norm_first=True, 
            bias=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_layer = nn.Linear(model_dim, uce_dim)    # back to UCE dimension
    
    def forward(
        self, 
        uce: torch.Tensor, 
        t: torch.Tensor, 
        conditions: Optional[Dict[str, torch.Tensor]] = None, 
        context_mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Process UCE embeddings with time and optional conditions through the transformer.
        
        Args:
            uce: Input UCE embeddings [batch_size, uce_dim]
            t: Timesteps [batch_size, 1]
            context_mask: Mask for CFG during training [batch_size]. 
                          During inference, we create this ourselves, so it is None.
            conditions: Dictionary of condition tensors, each [batch_size, cond_dim]
            
        Returns:
            Denoised UCE embeddings [batch_size, uce_dim]
        """

        uce = self.make_uce_embedding(uce).to(self.device)
        t_e = self.make_time_embedding(t.float()).to(self.device)

        # Create sequence of embeddings
        sequence = [uce, t_e]
        if conditions is not None:
            assert isinstance(conditions, dict), "Conditions must be a dictionary of condition names and values."
            for name in conditions:
                cond_emb = getattr(self, f'make_{name}_embedding')(conditions[name])
                # Apply CFG mask only to condition embeddings during training
                if self.cfg and context_mask is not None:
                    cond_emb = cond_emb * context_mask.unsqueeze(-1)
                sequence.append(cond_emb.to(self.device))

        sequence = torch.stack(sequence, dim=1)  # [batch, seq_len, dim]
        if self.learn_pos_embed:
            sequence = sequence + self.token_type_embeddings.unsqueeze(0)

        # Handle CFG for inference
        if self.cfg and context_mask is None and conditions is not None:
            uncond_sequence = sequence.clone()
            uncond_sequence[:, 2:] = 0.  # Zero out condition embeddings
            sequence = torch.cat([sequence, uncond_sequence], dim=0)

        transformer_output = self.transformer(sequence)   # [batch_size, seq_len, model_dim]

        # Get UCE token output and project back to UCE dimension
        uce_output = transformer_output[:, 0, :]  # [batch_size, model_dim]
        return self.final_layer(uce_output)       # [batch_size, uce_dim]
