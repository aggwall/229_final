import scanpy as sc
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
import pickle as pkl
from einops import rearrange, reduce
from ema_pytorch import EMA
import argparse
from accelerate import Accelerator
import wandb

from ucegen.helpers.data_utils import unnormalize_data, normalize_data
from ucegen.model_manager import ModelManager
from ucegen.uce_dataset import UCEDatasetManager
from ucegen.helpers.model_utils import setup_environment, lr_warmup, Config
from ucegen.metrics import get_name_to_function_mapping
from ucegen.finetune import LoRAFineTuner


class BaseDiffusionModel:
    def __init__(self, config, args, accelerator, model_manager, dataset_manager):
        self.config = config
        self.args = args
        self.accelerator = accelerator
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.device = self.accelerator.device

        self.adata = self.dataset_manager.load_uce_datasets(
            normalize=False if config.vae_model != "none" else True,
            finetune=True if config.mode == "finetune" else False
        )
        self.train_dataloader, train_dataset = self.dataset_manager.make_uce_dataloader(self.adata)
        self.train_dataloader = self.accelerator.prepare(self.train_dataloader)

        self.model, self.ddpm_constants, self.total_params = self.model_manager.configure_model(train_dataset)
        self.ema = EMA(
            self.model,
            beta = self.config.ema_beta,                    # exponential moving average factor
            update_every = self.config.ema_update_every,    # how often to actually update, to save on compute
        ).to(self.device)
        self.vae_model = self.model_manager.initialize_vae() if self.config.vae_model != "none" else None
        
        if self.config.pretrain_path is not None:
            self.model, self.ema = self.model_manager.load_model(self.model, self.ema)
            self.model = self.ema.ema_model if self.ema is not None else self.model

        self.metric_name_to_func = get_name_to_function_mapping()
        self.objective = config.pred_type
        self.immiscible = config.immiscible
    
    def _set_loss_weight(self, _ts):
        snr = self.ddpm_constants['sqrt_alphas_cumprod'][_ts] / (1 - self.ddpm_constants['sqrt_alphas_cumprod'][_ts])
        if self.objective == 'pred_noise':
            self.loss_weight = torch.ones_like(snr)
        elif self.objective == 'pred_x0':
            self.loss_weight = snr
        elif self.objective == 'pred_v':
            self.loss_weight = snr / (snr + 1)
    
    @torch.no_grad()
    def noise_assignment(self, x_start, noise):
        # Convert to FP16 (quantize) for faster computation
        x_start = x_start.to(torch.float16) 
        noise = noise.to(torch.float16) 

        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        
        ## Using a proper Hungarian algorithm is too slow, so we approximate by greedy assignment
        assign = dist.argmin(dim=-1)  
        # from scipy.optimize import linear_sum_assignment
        # _, assign = torch.from_numpy(linear_sum_assignment(dist.cpu()))

        # Convert back to FP32 after assignment
        return assign.long().to(dist.device)
    
    def predict_start_from_noise(self, x_t, _ts, noise):
        return (
            self.ddpm_constants['sqrt_recip_alphas_cumprod'][_ts] * x_t -
            self.ddpm_constants['sqrt_recipm1_alphas_cumprod'][_ts] * noise
        )

    def predict_noise_from_start(self, x_t, _ts, x0):
        return (
            (self.ddpm_constants['sqrt_recip_alphas_cumprod'][_ts] * x_t - x0) / \
            self.ddpm_constants['sqrt_recipm1_alphas_cumprod'][_ts]
        )

    def predict_v(self, x_start, _ts, noise):
        return (
            self.ddpm_constants['sqrt_alphas_cumprod'][_ts] * noise -
            self.ddpm_constants['sqrt_one_minus_alphas_cumprod'][_ts] * x_start
        )

    def predict_start_from_v(self, x_t, _ts, v):
        return (
            self.ddpm_constants['sqrt_alphas_cumprod'][_ts] * x_t -
            self.ddpm_constants['sqrt_one_minus_alphas_cumprod'][_ts] * v
        )
        
    def q_posterior(self, x_start, x_t, _ts):
        posterior_mean = (
            self.ddpm_constants['posterior_mean_coef1'][_ts] * x_start +
            self.ddpm_constants['posterior_mean_coef2'][_ts] * x_t
        )
        posterior_log_variance_clipped = self.ddpm_constants['posterior_log_variance_clipped'][_ts]
        return posterior_mean, posterior_log_variance_clipped
    
    def _train_init(self):
        if self.args.wandb_track and self.accelerator.is_main_process:
            if not wandb.run:
                module_name = "UCE Uncond Diffusion Model" if self.args.model == "uncond" else "Conditional UCE Diffusion Model"
                wandb.init(project=module_name, config=self.args, name=self.model_manager.model_name)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        if self.config.pretrain_path is not None:
            self.optimizer = self.model_manager.load_optimizer(self.optimizer)
            
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

        print(f"\n*****TRAINING DIFFUSION MODEL: {self.total_params} PARAMS*****")
        self.train_pbar = tqdm(range(self.config.epochs), desc="Training Progress", disable=not self.accelerator.is_local_main_process)

    def _adjust_learning_rate(self, epoch):
        new_lr = lr_warmup(epoch, initial_lr=3e-6, final_lr=self.config.learning_rate, warmup_epochs=15)
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr

    def _process_batch(self, batch):
        x_start = batch[0].to(self.device)
        _ts = torch.randint(1, self.config.num_diffusion_timesteps, (x_start.shape[0], 1), device=self.device)

        if self.config.vae_model == "separate":
            x_start = self.vae_model.encode(x_start)[-1]
        elif self.config.vae_model == "joint":
            x_start = self.vae_model.module.encode(x_start)[-1]

        noise = torch.randn_like(x_start).to(self.device)
        return x_start, noise, _ts, None, None
    
    def model_predictions(self, x, _ts, conds=None, context_mask=None):
        model_pred = self.model(x, _ts / self.config.num_diffusion_timesteps, conditions=conds, context_mask=context_mask)
        
        if self.objective == 'pred_noise':
            pred_noise = model_pred
            pred_x_start = self.predict_start_from_noise(x, _ts, pred_noise)
        elif self.objective == 'pred_x0':
            pred_x_start = model_pred
            pred_noise = self.predict_noise_from_start(x, _ts, pred_x_start)
        elif self.objective == 'pred_v':
            v = model_pred
            pred_x_start = self.predict_start_from_v(x, _ts, v)
            pred_noise = self.predict_noise_from_start(x, _ts, pred_x_start)
        
        return pred_x_start, pred_noise

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, _ts, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        
        if self.immiscible:
            """
            assigning noise to each x_start based on the nearest noise.
            adapted from https://arxiv.org/abs/2406.12303. 
            UPDATE: doesn't really work well, even using a very time-consuming algorithm.
            """
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]
        
        return (
            self.ddpm_constants['sqrt_alphas_cumprod'][_ts] * x_start +
            self.ddpm_constants['sqrt_one_minus_alphas_cumprod'][_ts] * noise
        )

    def train_step(self, x_start, noise, _ts, conds, context_mask):
        x = self.q_sample(x_start, _ts, noise=noise)
        model_pred = self.model(x, _ts / self.config.num_diffusion_timesteps, conditions=conds, context_mask=context_mask)
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, _ts, noise)
            target = v
        
        loss = self.criterion(model_pred, target)
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = (loss * self.loss_weight).mean()
        
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def train(self):
        self._train_init()

        for epoch in self.train_pbar:
            if self.config.pretrain_path is not None:
                self._adjust_learning_rate(epoch)
                
            train_losses = []
            self.model.train()
            
            for batch in self.train_dataloader:
                with self.accelerator.autocast():
                    x_start, noise, _ts, conds, context_mask = self._process_batch(batch)
                    self._set_loss_weight(_ts)
                    loss = self.train_step(x_start, noise, _ts, conds, context_mask)
                    train_losses.append(self.accelerator.gather(loss).mean().item())

            avg_loss = np.mean(train_losses)
            self.train_pbar.set_description(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

            if self.accelerator.is_main_process:
                self.ema.update()
                if self.args.wandb_track:
                    wandb.log({"train_loss": avg_loss, "epoch": epoch, "param_count": self.total_params})
                self._save_model(epoch)

        self.accelerator.end_training()
        if self.args.wandb_track:
            wandb.finish()

    def _save_model(self, epoch, prefix=""):
        # Save the model every epoch (overwrite the same file)
        if self.config.save_model_bool:
            self.model_manager.save_model(self.model, epoch, self.optimizer, self.ema, overwrite=True, prefix=prefix)
            
            # Save separately every {total_epochs // 4} epochs or at the last epoch
            if (epoch % (self.config.epochs // 4) == 0 and epoch != 0) or epoch == self.config.epochs - 1:
                self.model_manager.save_model(self.model, epoch, self.optimizer, self.ema, overwrite=False, prefix=prefix)
    
    def _get_min_max(self):
        self.min_val, self.max_val = self.dataset_manager.get_min_max_values()
    
    def _sample_init(self, num_samples):
        y_i = torch.randn(num_samples, self.config.uce_dim).to(self.model.device)
        self.sample_pbar = tqdm(range(self.config.num_diffusion_timesteps - 1, -1, -1), desc="Sampling Progress")
        return y_i

    @torch.no_grad()
    def p_sample(self, num_samples=5000):
        y_i = self._sample_init(num_samples)
        pred_x_start = None
        y_i_store = {}
        
        for diff_step_i in self.sample_pbar:
            t_is = torch.tensor([diff_step_i]).repeat(num_samples, 1).to(self.model.device)
            z = torch.randn(num_samples, self.config.uce_dim).to(self.model.device) if diff_step_i > 1 else 0

            pred_x_start, _ = self.model_predictions(y_i, t_is)
            pred_x_start = pred_x_start.clamp(-1.0, 1.0) if self.config.clamp else pred_x_start
            model_mean, model_log_variance = self.q_posterior(pred_x_start, y_i, t_is)
            y_i = model_mean + torch.exp(0.5 * model_log_variance) * z

            if diff_step_i % 10 == 0 or diff_step_i < 8:
                y_i_store[diff_step_i] = y_i.cpu().numpy()

        y_i = self.vae_model.decode(y_i) if self.vae_model else y_i
        y_i = unnormalize_data(y_i.cpu().numpy(), self.min_val, self.max_val)
        self.model_manager.save_samples(y_i)
        return y_i, y_i_store
    
    @torch.no_grad()
    def ddim_sample(self, sampling_timesteps=50, eta=0., num_samples=5000):
        times = torch.linspace(0, self.config.num_diffusion_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        y_i = self._sample_init(num_samples)
        pred_x_start = None
        y_i_store = {}

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            t_is = torch.tensor([time]).repeat(num_samples, 1).to(self.model.device)
            z = torch.randn(num_samples, self.config.uce_dim).to(self.model.device) if time > 1 else 0

            pred_x_start, pred_noise = self.model_predictions(y_i, t_is)
            pred_x_start = pred_x_start.clamp(-1.0, 1.0) if self.config.clamp else pred_x_start
            
            if time_next == -1:
                y_i = pred_x_start
                continue
            
            alpha = self.ddpm_constants['alphas_cumprod'][time]
            alpha_next = self.ddpm_constants['alphas_cumprod'][time_next]
            
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            y_i = pred_x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * z
            
            if time % 10 == 0 or time < 8:
                y_i_store[time] = y_i.cpu().numpy()
        
        y_i = self.vae_model.decode(y_i) if self.vae_model else y_i
        y_i = unnormalize_data(y_i.cpu().numpy(), self.min_val, self.max_val)
        self.model_manager.save_samples(y_i)
        return y_i, y_i_store

    def _evaluate_timesteps(self, y_i_store, adata_real_samples, adata_target_samples=None):
        """Evaluate metrics and create plots for multiple timesteps.
        
        Args:
            y_i_store: Dictionary mapping timesteps to intermediate samples
            adata_real_samples: AnnData object containing real samples
            adata_target_samples: Optional AnnData object containing target samples (for conditional model)
        """
        keys_to_plot = [3, 5, 10, 20, 50, 100, 150, 200, 240][::-1]
        max_step = max(keys_to_plot)

        # Initialize dictionaries to accumulate metrics
        metrics_to_log = {metric: [] for metric in self.config.metrics_list.split(",")}
        timestep_log = []

        for step in keys_to_plot:
            intermediate_samples = y_i_store[step]
            adata_intermediate_samples = sc.AnnData(intermediate_samples)
            adata_intermediate_samples.obs['data_type'] = f'intermediate_{step}'
            # Build concatenated AnnData for plotting
            concat_dict = {
                "original": adata_real_samples,
                f"intermediate_{step}": adata_intermediate_samples
            }
            if adata_target_samples is not None:
                concat_dict[f"target={self.config.target_type}"] = adata_target_samples
                
            joined_intermediate_ad = sc.concat(concat_dict, label="data_type")

            print(f"\nGenerating PCA plot for step {step}...")
            self.model_manager.make_pca_plot(
                joined_intermediate_ad, 
                title=f"PCA at Diffusion Step {step}", 
                extra_save_text=f"{step}"
            )

            # Calculate metrics
            step_metrics = {}
            comparison_data = adata_target_samples.X if adata_target_samples is not None else adata_real_samples.X
            for metric in self.config.metrics_list.split(","):
                step_metrics[metric] = self.metric_name_to_func[metric](
                    adata_intermediate_samples.X, 
                    comparison_data
                ).round(6)
            print(f"Metrics at step {step}: {step_metrics}")

            # Accumulate metrics for logging
            timestep_log.append(max_step - step)
            for metric, value in step_metrics.items():
                metrics_to_log[metric].append(value)

        # Log all metrics at once to WandB
        if self.args.wandb_track:
            for metric, values in metrics_to_log.items():
                for i, value in enumerate(values):
                    wandb.log({f"eval/{metric}": value, "timestep": timestep_log[i]})

        return metrics_to_log

    @torch.no_grad()
    def evaluate(self):
        self.model, self.ema = self.model_manager.load_model(self.model, self.ema)
        self.model = self.ema.ema_model if self.ema is not None else self.model
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        
        self._get_min_max()
        self.metrics_dict = {}
        self.eval_adata = self.adata[np.random.choice(len(self.adata), self.config.num_samples, replace=False)]

        if self.model_manager.load_samples() is not None and not self.config.generate_new:
            print("Loading generated samples from file...")
            generated_samples = self.model_manager.load_samples()
            with open(f"{self.model_manager.args.home_dir}/uce_y_i/{self.model_manager.model_name}_y_i_store.pkl", "rb") as f:
                y_i_store = pkl.load(f)
        else:
            if self.config.use_ddim_sample:
                assert self.config.num_ddim_timesteps <= self.config.num_diffusion_timesteps
                generated_samples, y_i_store = self.ddim_sample(
                    sampling_timesteps=self.config.num_ddim_timesteps,
                    eta=self.config.ddim_eta,
                    num_samples=self.config.num_samples
                )
            else:
                generated_samples, y_i_store = self.p_sample(num_samples=self.config.num_samples)
                
            for step in y_i_store:
                y_i_store[step] = unnormalize_data(y_i_store[step], self.min_val, self.max_val)
            with open(f"{self.model_manager.args.home_dir}/uce_y_i/{self.model_manager.model_name}_y_i_store.pkl", "wb") as f:
                pkl.dump(y_i_store, f)

        adata_gen_samples = sc.AnnData(generated_samples)
        
        real_samples = unnormalize_data(self.eval_adata.X, self.min_val, self.max_val)
        adata_real_samples = sc.AnnData(real_samples)

        adata_gen_samples.obs['data_type'] = 'generated'
        adata_real_samples.obs['data_type'] = 'original'

        joined_ad = sc.concat({"original": adata_real_samples, "generated": adata_gen_samples}, label="data_type")

        self.model_manager.make_hist_features(joined_ad, left_limit=-0.3, right_limit=0.3, increment=0.05)
        self.model_manager.make_pca_plot(joined_ad, title="PCA of Original and Generated Data")
        self.model_manager.make_umap_plot(joined_ad, title="UMAP of Original and Generated Data")
        
        for metric in self.config.metrics_list.split(","):
            self.metrics_dict[metric] = self.metric_name_to_func[metric](adata_gen_samples.X, adata_real_samples.X).round(6)
        print(f"Overall Metrics (i.e. at timestep 0): {self.metrics_dict}")
        
        if self.args.wandb_track:
            wandb.log({"eval/overall": self.metrics_dict})
        
        self._evaluate_timesteps(y_i_store, adata_real_samples)
        

class UCECondDiffusionModel(BaseDiffusionModel):
    def __init__(self, config, args, accelerator, model_manager, dataset_manager):
        super().__init__(config, args, accelerator, model_manager, dataset_manager)
        self.cfg = config.cfg
        self.drop_prob = config.drop_prob if config.cfg else 0.0
        self.guide_w = config.guide_w if config.cfg else 0.0
        self.conditions = config.conditions.split(",") if config.conditions else []
    
    def _make_condition_dict(self, batch):
        conds = {}
        for condition_name in self.conditions:
            if condition_name in batch[1].keys():
                conds[condition_name] = batch[1][condition_name].to(self.device)
            else:
                raise ValueError("Invalid condition name.")
        return conds

    def _process_batch(self, batch):
        x_start, noise, _ts, _, _ = super()._process_batch(batch)
        
        train_conds = self._make_condition_dict(batch)
        context_mask = torch.bernoulli(torch.full((x_start.size(0),), 1 - self.drop_prob)).to(self.model.device) if self.cfg else None

        return x_start, noise, _ts, train_conds, context_mask

    def _sample_init(self, num_samples=5000, start_emb=None):
        y_i = super()._sample_init(num_samples)
        if self.config.sdedit_mode:
            assert start_emb is not None and self.config.sdedit_timestep is not None
            _ts = torch.tensor([int(self.config.sdedit_timestep * self.config.num_diffusion_timesteps)] * num_samples, device=self.device).unsqueeze(1)
            y_i = start_emb * self.ddpm_constants['sqrt_alphas_cumprod'][_ts] + torch.randn_like(start_emb) * self.ddpm_constants['sqrt_one_minus_alphas_cumprod'][_ts]
        return y_i

    @torch.no_grad()
    def sample(self, conditions_dict, num_samples=5000, start_emb=None, save_samples=False, calculate_likelihood=False):
        y_i = self._sample_init(num_samples=num_samples, start_emb=start_emb)
        pred_x_start = None
        y_i_store = {}
        mse_accumulator = 0 if calculate_likelihood else None
        
        for diff_step_i in self.sample_pbar:
            t_is = torch.tensor([diff_step_i]).repeat(num_samples, 1).to(self.model.device)
            z = torch.randn(num_samples, self.config.uce_dim).to(self.model.device) if diff_step_i > 1 else 0

            pred_x_start, pred_noise = self.model_predictions(y_i, t_is, conds=conditions_dict)
            pred_x_start = pred_x_start.clamp(-1.0, 1.0) if self.config.clamp else pred_x_start
            self.accelerator.wait_for_everyone()

            if self.cfg:
                cond_part_pred = pred_x_start[:num_samples]   # first half is conditional
                uncond_part_pred = pred_x_start[num_samples:]
                pred_x_start = (1 + self.guide_w) * cond_part_pred - self.guide_w * uncond_part_pred
                y_i = y_i[:num_samples]

            model_mean, model_log_variance = self.q_posterior(pred_x_start, y_i, t_is)
            y_i = model_mean + torch.exp(0.5 * model_log_variance) * z

            # Calculate MSE loss for all samples across all timesteps
            # taking mean across all num_samples, bc should be ~random/equal
            if calculate_likelihood:
                assert self.config.pred_type == "pred_noise", "only pred_noise is supported for likelihood calculation, for direct proxy purposes"
                mse = F.mse_loss(y_i, pred_noise)
                mse_accumulator += mse.item()

            if diff_step_i % 10 == 0 or diff_step_i < 8:
                y_i_store[diff_step_i] = unnormalize_data(y_i.cpu().numpy(), self.min_val, self.max_val)

        y_i = self.vae_model.decode(y_i) if self.config.vae_model != "none" else y_i
        y_i = unnormalize_data(y_i.cpu().numpy(), self.min_val, self.max_val)

        if save_samples:
            suffix = f"_{self.config.target_type}"
            if self.config.sdedit_mode:
                suffix += f"_sdedited_{self.config.initial_type}"
            self.model_manager.save_samples(y_i, target_adata=self.target_adata_slice, suffix=suffix)
        
            with open(f"{self.model_manager.args.home_dir}/uce_y_i/{self.model_manager.model_name}_y_i_store{suffix}.pkl", "wb") as f:
                pkl.dump(y_i_store, f)
                print(f"Saved y_i_store to file after generating samples. Saved at: {self.model_manager.args.home_dir}/uce_y_i/{self.model_manager.model_name}_y_i_store{suffix}.pkl")

        if calculate_likelihood:
            avg_mse = mse_accumulator / (self.config.num_diffusion_timesteps - 1)
            print(f"Average across-all-timesteps MSE: {avg_mse}")
            return y_i, y_i_store, avg_mse
        else:
            return y_i, y_i_store

    def _calculate_cos_sim_dict(self, samples, adata_slice, col):
        cos_sim_dict = {}
        for item in adata_slice.obs[col].unique():
            item_slice = adata_slice[adata_slice.obs[col] == item]
            if item_slice.X.shape[0] == 0:
                continue
            item_slice_subset = item_slice[np.random.choice(range(item_slice.X.shape[0]), size=samples.shape[0])]
            item_slice_samples = unnormalize_data(item_slice_subset.X, self.min_val, self.max_val)
            cos_sim_dict[item] = round(float(self.metric_name_to_func["cosine"](samples, item_slice_samples)), 5)
        cos_sim_dict = dict(sorted(cos_sim_dict.items(), key=lambda item: item[1], reverse=True))
        print(f"\n\ntransfer_{self.config.target_type} = {cos_sim_dict}")
        return cos_sim_dict
    
    @torch.no_grad()
    def evaluate(self):
        self.model, self.ema = self.model_manager.load_model(self.model, self.ema)
        self.model = self.ema.ema_model if self.ema is not None else self.model
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        
        self._get_min_max()
        self.metrics_dict = {}
        self.eval_adata = self.adata
            
        # each target type corresponds to the same-indexed adata col.
        # run a for loop to filter to the exact combination of target types
        self.target_types = self.config.target_type.split(",")
        self.target_adata_cols = self.config.target_adata_cols.split(",")
        self.target_adata_slice = self.eval_adata
        filtered_indices = set(range(len(self.eval_adata.obs)))  # Start with all possible indices

        # Iterate through each target type and corresponding column
        for target_type, adata_col in zip(self.target_types, self.target_adata_cols):
            current_target_indices = {i for i, target in enumerate(self.eval_adata.obs[adata_col]) if target == target_type}
            filtered_indices &= current_target_indices

        target_indices = list(filtered_indices)

        # Handle the case where there are no matching samples
        if not target_indices:
            print(f"Warning: No samples match the target combination: {self.target_types}")
            # Use all indices if no matching samples are found
            target_indices = list(range(len(self.eval_adata.obs)))
        
        # Ensure we have enough indices to sample from
        if len(target_indices) < self.config.num_samples:
            target_indices = np.random.choice(target_indices, size=self.config.num_samples, replace=True)
        
        num_samples = self.config.num_samples
        print(f"Number of samples to generate: {num_samples}")
            
        if self.config.sdedit_mode:
            if self.config.initial_type is not None:
                self.initial_adata_slice = self.eval_adata
                self.initial_types = self.config.initial_type.split(",")
                self.initial_adata_cols = self.config.initial_adata_cols.split(",")
                for initial_type, adata_col in zip(self.initial_types, self.initial_adata_cols):
                    self.initial_adata_slice = self.initial_adata_slice[self.initial_adata_slice.obs[adata_col] == initial_type]
                if len(self.initial_adata_slice) > 0:
                    start_emb = torch.tensor(self.initial_adata_slice.X.mean(axis=0)).to(self.device).unsqueeze(0)
                else:
                    print(f"Warning: No samples match the initial combination: {self.initial_types}")
                    start_emb = None
            
            elif self.config.initial_emb_path is not None:
                with open(self.config.initial_emb_path, "rb") as f:
                    start_emb_dict = pkl.load(f)
                timestep_to_get = self.config.sdedit_timestep * self.config.num_diffusion_timesteps
                # get the timestep in start_emb_dict closest to timestep_to_get
                closest_timestep = min(start_emb_dict.keys(), key=lambda x: abs(x - timestep_to_get))
                start_emb_timestep = start_emb_dict[closest_timestep]
                print(f"Using start_emb from timestep {closest_timestep} for sdedit mode")
                start_emb, emb_min_val, emb_max_val = normalize_data(start_emb_timestep[0])
                start_emb = torch.tensor(start_emb).to(self.device).unsqueeze(0)
        else:
            start_emb = None
                
        if self.model_manager.load_samples() is not None and not self.config.generate_new:
            print("Loading generated samples from file...")
            generated_samples = self.model_manager.load_samples()
        else:
            eval_dataloader = self.dataset_manager.make_uce_dataloader(self.eval_adata)
            print("Made eval dataloader. Now entering evaluation loop...")
            for batch in eval_dataloader:
                eval_conds = self._make_condition_dict(batch)
                valid_conds = {k: v[target_indices] for k, v in eval_conds.items()}
                valid_conds_subset = {k: v[:num_samples] for k, v in valid_conds.items()}
                
                generated_samples, y_i_store = self.sample(valid_conds_subset, 
                                                num_samples=num_samples, 
                                                start_emb=start_emb,
                                                save_samples=True,
                                                calculate_likelihood=False)

        cos_all_to_every_coarse_ct = self._calculate_cos_sim_dict(generated_samples, self.eval_adata, 'coarse_cell_type')
        if self.args.wandb_track:
            wandb.log({"cos_sim/all_to_coarse": cos_all_to_every_coarse_ct})

        real_samples = unnormalize_data(self.eval_adata.X, self.min_val, self.max_val)
        target_samples = unnormalize_data(self.target_adata_slice.X, self.min_val, self.max_val)
        
        # filter to num_samples length for comparison
        real_samples = real_samples[np.random.choice(range(real_samples.shape[0]), size=generated_samples.shape[0], replace=False)]
        target_samples = target_samples[np.random.choice(range(target_samples.shape[0]), size=generated_samples.shape[0], replace=False)]
        
        adata_gen_samples = sc.AnnData(generated_samples)
        adata_real_samples = sc.AnnData(real_samples)
        adata_target_samples = sc.AnnData(target_samples)
        
        adata_gen_samples.obs['data_type'] = 'generated'
        adata_real_samples.obs['data_type'] = 'original'
        adata_target_samples.obs['data_type'] = 'target'
        
        if self.config.sdedit_mode:
            if self.config.initial_type is not None:
                initial_samples = unnormalize_data(self.initial_adata_slice.X, self.min_val, self.max_val)
            elif self.config.initial_emb_path is not None:
                initial_samples = start_emb_timestep
            
            adata_initial_samples = sc.AnnData(initial_samples)[np.random.choice(range(initial_samples.shape[0]), 
                                                                                 size=generated_samples.shape[0], 
                                                                                 replace=True)]
            adata_initial_samples.obs['data_type'] = 'initial'
            joined_ad = sc.concat({"original": adata_real_samples, 
                                   "generated": adata_gen_samples, 
                                   f"target={self.config.target_type}": adata_target_samples, 
                                   f"initial={self.config.initial_type}": adata_initial_samples}, 
                                   label="data_type")
        else:
            joined_ad = sc.concat({"original": adata_real_samples, 
                                   "generated": adata_gen_samples, 
                                   f"target={self.config.target_type}": adata_target_samples}, 
                                   label="data_type")
        

        self.model_manager.make_pca_plot(joined_ad, title="PCA of Original and Generated Data", extra_save_text=f"_sdedited_{self.config.initial_type}" if self.config.sdedit_mode else "")

        print("\nCalculating Metrics...")
        for metric in self.config.metrics_list.split(","):
            if metric == "cosine":
                self.metrics_dict[metric + '_gen'] = self.metric_name_to_func[metric](generated_samples, target_samples).round(5)
                self.metrics_dict[metric + '_real'] = self.metric_name_to_func[metric](real_samples, target_samples).round(5)
            else:
                self.metrics_dict[metric]= self.metric_name_to_func[metric](generated_samples, target_samples).round(5)
        print(f"Overall Metrics (i.e. at timestep 0): {self.metrics_dict}")
        
        if self.args.wandb_track:
            wandb.log({"eval/overall": self.metrics_dict})
        
        self._evaluate_timesteps(
            y_i_store, 
            adata_real_samples,
            adata_target_samples=adata_target_samples,
        )

def main():
    setup_environment(10)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["uncond", "cond"], help="Path to the model file")
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yaml file")
    parser.add_argument("--home_dir", type=str, default="/dfs/user/ayushag/diffusion", help="Path to the home directory")
    parser.add_argument("--wandb_track", action="store_true", help="Enable wandb tracking")
    args = parser.parse_args()

    config = Config(args.config)
    
    accelerator = Accelerator(split_batches=True, project_dir=args.home_dir)
    print(f"\nConfiguration Loaded:", config.config, "\n")

    dataset_manager = UCEDatasetManager(config)
    model_manager = ModelManager(args, config, accelerator)
    
    if args.model == "uncond":
        diffusion_model = BaseDiffusionModel(config, args, accelerator, model_manager, dataset_manager)
    elif args.model == "cond":
        diffusion_model = UCECondDiffusionModel(config, args, accelerator, model_manager, dataset_manager)
    else:
        raise ValueError("Invalid model. Please choose either 'uncond' or 'cond'.")
    
    if config.mode == "train":
        diffusion_model.train()
    elif config.mode == "finetune":
        finetuner = LoRAFineTuner(diffusion_model, config, args, accelerator, model_manager, dataset_manager)
        finetuner.finetune()
    elif config.mode == "eval":
        diffusion_model.evaluate()
    else:
        raise ValueError("Invalid mode. Please choose either 'train', 'finetune', or 'eval'.")

if __name__ == "__main__":
    main()
