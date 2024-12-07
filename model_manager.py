import os
import torch
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

from ucegen.denoise_models import DiffusionDenoiser, DiffusionTransformer
from ucegen.helpers.model_utils import count_parameters
from ucegen.vae.vae_trainer import Encoder, Decoder, VAE
from ucegen.noise_schedule import ddpm_schedules

os.environ['SAVE_DIR'] = '/dfs/user/ayushag/ucegen/figures/'

class ModelManager:
    def __init__(self, args, config, accelerator):
        self.args = args
        self.config = config
        self.accelerator = accelerator
        self.script_type = "pert" if self.args.config.find("pert") != -1 else "uce"

    def configure_model(self, dataset):
        self.ddpm_constants = {
            k: v.to(self.accelerator.device) 
            for k, v in ddpm_schedules(
                self.config.beta_1, 
                self.config.beta_2, 
                self.config.num_diffusion_timesteps,
                beta_schedule=self.config.noise_schedule
            ).items()
        }
        
        self.cond_names_and_dims = {
            cond: (self.config.model_dim if not self.config.cond_emb_dim else self.config.cond_emb_dim)
            for cond in dataset.cond_variables
        }

        init_model_name = f"{self.config.denoise_model_type}_{self.config.model_title}_" \
            f"ep={self.config.epochs}_" \
            f"bs={self.config.batch_size}_" \
            f"ndt={self.config.num_diffusion_timesteps}_" \
            f"lr={self.config.learning_rate}_" \
            f"wd={self.config.weight_decay}"        

        if self.config.denoise_model_type == "DiffusionDenoiser":
            self.model = DiffusionDenoiser(
                cond_names_and_dims=self.cond_names_and_dims,
                uce_dim=self.config.uce_dim,                    
                time_dim=self.config.time_dim, 
                num_hidden_layers=self.config.num_mlp_hidden_layers,  
                hidden_multiple=self.config.mlp_hidden_dim_multiple,  
                num_xattn_layers=self.config.num_xattn_layers,
                cond_emb_type=self.config.cond_emb_type,
                condition_combination=self.config.condition_type,
                t_theta_type=self.config.t_theta_type,
                sieve=self.config.sieve,
                cfg=self.config.cfg,
                output_dim=self.config.output_dim if self.config.output_dim is not None else self.config.uce_dim,
                device=self.accelerator.device,                  
            )
            print("Initialized DiffusionDenoiser model...")

            if self.config.full_model_name is not None:
                self.model_name = self.config.full_model_name
            else:
                self.model_name = f"{init_model_name}_" \
                    f"num_hidden_layers={self.config.num_mlp_hidden_layers}_" \
                    f"hidden_multiple={self.config.mlp_hidden_dim_multiple}_" \
                    f"num_xattn_layers={self.config.num_xattn_layers}"

        elif self.config.denoise_model_type == "DiffusionTransformer":
            self.model = DiffusionTransformer(
                uce_dim=self.config.uce_dim,
                model_dim=self.config.model_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                cond_names_and_dims=self.cond_names_and_dims,
                cfg=self.config.cfg,
                learn_pos_embed=self.config.learn_pos_embed,
                device=self.accelerator.device,
            )
            print("Initialized DiffusionTransformer model...")

            if self.config.full_model_name is not None:
                self.model_name = self.config.full_model_name
            else:
                self.model_name = f"{init_model_name}_" \
                    f"model_dim={self.config.model_dim}_" \
                    f"num_layers={self.config.num_layers}_" \
                    f"num_heads={self.config.num_heads}_" \
                    f"learn_pos_embed={self.config.learn_pos_embed}"
        
        else:
            raise ValueError(f"Invalid model type: {self.config.denoise_model_type}")
        
        return self.model.to(self.accelerator.device), self.ddpm_constants, count_parameters(self.model)
    
    def save_model(self, model, epoch, opt, ema, overwrite=True, prefix=""):
        save_directory = f"{self.args.home_dir}/saved_models/{self.script_type}_{self.args.model}/{self.config.model_title}"  
        os.makedirs(save_directory, exist_ok=True)

        if overwrite:
            save_path = os.path.join(save_directory, f"{prefix}latest_model")
        else:
            save_path = os.path.join(save_directory, f"{prefix}{self.model_name}_epoch_{epoch}")
        
        data = {
            'epoch': epoch,
            'model': self.accelerator.get_state_dict(model),
            'opt': opt.state_dict(),
            'ema': ema.state_dict(), 
        }
        
        torch.save(data, save_path)

    def load_model(self, model, ema):
        if self.config.pretrain_path is not None:
            load_path = self.config.pretrain_path
        else:
            load_path = f"{self.args.home_dir}/saved_models/{self.script_type}_{self.args.model}/{self.config.model_title}/{self.model_name}"
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint found at {load_path}")
        
        print(f"Loading model from {load_path}...")
        data = torch.load(load_path, map_location=self.accelerator.device, weights_only=True)
        model = self.accelerator.unwrap_model(model)
        
        # if data is of type dict, load the model and ema from the dict
        if type(data) == dict:
            model.load_state_dict(data['model'])
            if self.accelerator.is_main_process:
                ema.load_state_dict(data["ema"])
            return model, ema
        else:
            model.load_state_dict(data)
            return model, None
            
    def load_optimizer(self, opt):
        load_path = f"{self.args.home_dir}/saved_models/{self.script_type}_{self.args.model}/{self.config.model_title}/{self.model_name}"
        data = torch.load(load_path, map_location=self.accelerator.device)

        return opt.load_state_dict(data['opt'])
    
    def save_samples(self, final_samples, target_adata=None, suffix=""):
        adata_out = sc.AnnData(np.vstack(final_samples))
        num_samples = final_samples.shape[0]
        
        if target_adata is not None:
            obs = target_adata.obs
            
            if num_samples > len(obs): 
                raise ValueError(f"Number of samples ({num_samples}) exceeds available data in target_adata ({len(obs)}).")
            
            if 'pert_type' in self.cond_names:
                if 'pert' in obs:
                    adata_out.obs['pert'] = obs['pert'].values[:num_samples]
                else:
                    raise KeyError("'pert' column is not found in target_adata.obs.")
            
            if 'cell_type' in self.cond_names:
                if 'cell_type' in obs:
                    adata_out.obs['cell_type'] = obs['cell_type'].values[:num_samples]
                else:
                    raise KeyError("'cell_type' column is not found in target_adata.obs.")
            
            if 'tissue_type' in self.cond_names:
                if 'tissue' in obs:
                    adata_out.obs['tissue'] = obs['tissue'].values[:num_samples]
                else:
                    raise KeyError("'tissue' column is not found in target_adata.obs.")

        output_dir = os.path.join(self.args.home_dir, f"generated_cells/{self.script_type}_{self.args.model}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"gen_diffusion_{self.model_name}{suffix}.h5ad")
        adata_out.write(output_path)
        print(f"Generated samples saved to {output_path}.")
    
    def load_samples(self):
        path = f"{self.args.home_dir}/generated_cells/{self.script_type}_{self.args.model}/gen_diffusion_{self.model_name}.h5ad"
        if os.path.exists(path):
            adata = sc.read_h5ad(path)
            return adata.X
        return None

    def make_hist_features(self, joined_ad, left_limit=-0.25, right_limit=0.25, increment=0.05):
        gen_indices = joined_ad[joined_ad.obs['data_type']=='generated'].obs.index.values.astype('int')
        bins = np.arange(left_limit, right_limit, increment)

        for feature_dim in [350, 700, 1050]:    # feature dims arbitrarily picked for consistency of plots
            plt.figure()  # Create a new figure for each histogram
            plt.hist(joined_ad.X[:, feature_dim][~gen_indices], bins=bins, alpha=0.5, label='Original')
            plt.hist(joined_ad.X[:, feature_dim][gen_indices], bins=bins, alpha=0.5, label='Generated')
            plt.title(f"Original vs. Generated Distributions\n(Unnormalized) for Feature Dim {feature_dim}")
            plt.xlim(left_limit, right_limit)
            plt.legend(loc='upper right')
            
            # Save each histogram as a separate PNG file
            fig_path = os.path.join(os.environ['SAVE_DIR'], f"hists/feature_dim_{feature_dim}_{self.model_name}.pdf")
            plt.savefig(fig_path)
            plt.close()  # Close the figure after saving to free memory
    
    def make_pca_plot(self, adata, title, extra_save_text=None):
        sc.tl.pca(adata)
        save_path = f"/{self.model_name}_{extra_save_text}.pdf" if extra_save_text is not None else f"/{self.model_name}.pdf"
        sc.pl.pca(adata, color='data_type', title=title, save=save_path)   

    def make_umap_plot(self, adata, title, extra_save_text=None):
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        save_path = f"/{self.model_name}_{extra_save_text}.pdf" if extra_save_text is not None else f"/{self.model_name}.pdf"
        sc.pl.umap(adata, color='data_type', title=title, save=save_path)
    
    def initialize_vae(self):
        encoder = Encoder(self.config.uce_dim, self.config.hidden_dim, self.config.latent_dim)
        decoder = Decoder(self.config.latent_dim, self.config.hidden_dim, self.config.uce_dim)
        vae = VAE(encoder, decoder).to(self.accelerator.device).train()

        assert self.config.vae_path is not None, "VAE path must be provided for separate VAE model"
        vae.load_state_dict(torch.load(self.config.vae_path, map_location=self.accelerator.device))
    
        if self.config.vae_model == "separate":
            # Freeze the VAE model
            for param in vae.encoder.parameters():
                param.requires_grad = False
            for param in vae.decoder.parameters():
                param.requires_grad = False

        return self.accelerator.prepare(vae)
