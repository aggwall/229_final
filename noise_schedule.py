import torch
import torch.nn.functional as F
import math

def linear_beta_schedule(timesteps, beta1 = 0.0001, beta2 = 0.02):
    return torch.linspace(beta1, beta2, timesteps, dtype = torch.float32)

def scaled_linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)

def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    quad_term = torch.square(torch.arange(-1, timesteps, dtype=torch.float32))
    return (beta_end - beta_start) * quad_term / torch.max(quad_term) + beta_start

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def ddpm_schedules(beta1, beta2, T, beta_schedule="linear"):
    """
    Returns pre-computed schedules for DDPM sampling, training process. 
    These consist of various constants necessary for calculations in both the forward (closed-form) and reverse (actual diffusion) processes.
    Loosely adapted from: https://github.com/microsoft/Imitating-Human-Behaviour-w-Diffusion/blob/main/code/models.py
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    if beta_schedule == "linear":
        betas = linear_beta_schedule(T)
    elif beta_schedule == "scaled_linear":
        betas = scaled_linear_beta_schedule(T)
    elif beta_schedule == "quadratic":
        betas = quadratic_beta_schedule(T)
    elif beta_schedule == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        raise ValueError(f"Unknown beta_schedule: {beta_schedule}. Must be one of 'linear', 'scaled_linear', 'quadratic', 'cosine'.")

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)     
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

    mab_over_sqrt_one_minus_alphas_cumprod_inv = (1.0 - alphas) / torch.sqrt(1.0 - alphas_cumprod)    
    
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {
        "oneover_sqrta": torch.sqrt (1.0 / alphas),
        "mab_over_sqrt_one_minus_alphas_cumprod": mab_over_sqrt_one_minus_alphas_cumprod_inv,  
        "alphas_cumprod": alphas_cumprod,  # alphabar_t
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),  
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod), 
        "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
        "sqrt_recipm1_alphas_cumprod": torch.sqrt(1. / alphas_cumprod - 1),
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": torch.log(posterior_variance.clamp(min =1e-20)),
        "posterior_mean_coef1": betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
        "posterior_mean_coef2": (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod),
    }
