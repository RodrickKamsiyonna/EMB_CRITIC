# EBM World Model (on LeWorldModel)

Action-free Energy-Based World Model built on top of `le-wm` (`stable-worldmodel` + `stable-pretraining`) with a from-scratch **ResNet-18** visual encoder, a **causal RoPE Transformer EBM**, **SIGReg** regularization, and a detached **Inverse Dynamics Model (IDM)** for open-loop and closed-loop latent planning.

## Architecture

1. **Visual Encoder (`ResNet18Encoder` in `ebm_modules.py`)**
   - From-scratch `ResNet-18` (`pretrained=False`) + MLP projector (`512 -> 2048 -> D`, `BatchNorm1d`, `GELU`).
   - Clean token embeddings are centered and divided by their own std across feature dimensions (`standardize_tokens`) before corruption.
2. **Corruption & Denoising Objective (`EBMWorldModel`)**
   - Masking ratio per sequence: $r \sim U(0, 1)$, each valid token selected independently with probability $r$ (if none selected, the first valid token is selected).
   - Noise scale per sequence (shared across selected tokens): $z \sim \mathcal{N}(-1.5, 0.8^2)$, $\gamma = \sigma(z)$ (median $\approx 0.182$), $x = \gamma e + (1 - \gamma)\epsilon$ (`noise_scale=1.0`), $\epsilon \sim \mathcal{N}(0, I)$. Unselected tokens retain clean standardized embeddings.
   - Noisy inputs enter the Transformer **without** an input RMSNorm; internal block pre-RMSNorm and final RMSNorm are retained.
   - **Causal RoPE Transformer (`EBMTransformer`):** RoPE, QK-RMSNorm, `ReLU²` MLP, causal self-attention, no vocab embedding / LM head / value-embedding table.
     - Default (`model=ebm_tiny`): 6 layers, `embed_dim=192`, 3 heads × `head_dim=64` (~15M total params, matching LeWM's tiny scale for fast single-A100 training).
     - Large preset (`model=ebm_large`): 15 layers, `embed_dim=1664`, 13 heads × `head_dim=128`, `max_seq_len=1024`.
   - **Energy Head:** Bias-free 2-layer MLP $u_t = W_2\,\text{GELU}(W_1 h_t)$ with $W_1, W_2 \in \mathbb{R}^{D \times D}$; token energy is $E_t = \sqrt{\sum_{d=1}^D u_{t,d}^2 + 10^{-12}}$, summed across all positions (clean and corrupted) and batch elements to form $E(X)$.
   - **Denoising MSE:** $\hat{e} = \operatorname{sg}(x) - \frac{1}{\alpha}\nabla_x E(x)$ ($\alpha=1$): $x$ enters $E(x)$ **non-detached** (so $x$ still receives gradients through $\nabla_x E(x)$), while the additive skip term $x$ in $\hat{e}$ is detached so the MSE pushes gradients through $\nabla_x E(x)$ only; the target clean standardized embeddings are **non-detached**, averaged only over selected tokens and embedding dimensions (no $\gamma$ weighting).
3. **Regularization & Inverse Dynamics Model**
   - **SIGReg** (from LeWM) applied to clean embeddings (`weight=0.09`).
   - **IDM (`InverseDynamicsModel`):** trained on **detached** consecutive clean standardized embeddings $(e_t, e_{t+1})$ to predict the `frameskip * action_dim` action chunk between frames.

## Planning (`planner.py`)

- **Open-loop (`config/eval/pusht_ebm_openloop.yaml`):** Fix observed frame(s) $e_{\text{obs}}$ at the start and goal embedding $e_{\text{goal}}$ at the end, initialize intermediate horizon states $z_1, \dots, z_{H-1}$ with random noise, minimize $E([e_{\text{obs}}, z_1, \dots, z_{H-1}, e_{\text{goal}}])$ updating only the noise tokens, then decode consecutive state pairs via IDM and apply all actions.
- **Closed-loop (`config/eval/pusht_ebm_closedloop.yaml`):** Same energy minimization, accept the first `receding_horizon` macro-steps, execute them in the world, append the new observed frame(s) (retaining past states up to `history_size`), and replan until the full horizon finishes.

## Commands

```bash
# 1. Install LeWM + HDF5 + PushT dependencies (Python 3.10–3.13 compatible)
pip install "stable-worldmodel[train]" h5py hdf5plugin pygame pymunk shapely "imageio[ffmpeg]"

# 2. Run unit & end-to-end training/resume tests
python tests/test_ebm_wm.py
python tests/test_train_ebm_e2e.py

# 3. Train on PushT (full-episode mode, frameskip=5, max_ep_frames=64)
python train_ebm.py data=pusht data_mode=episode

# 3b. Train with full 15L x 1664d preset
python train_ebm.py data=pusht model=ebm_large embed_dim=1664 loader.batch_size=2

# 4. Evaluate Open-Loop & Closed-Loop on PushT
python eval_ebm.py --config-name=pusht_ebm_openloop.yaml policy=ebm_wm
python eval_ebm.py --config-name=pusht_ebm_closedloop.yaml policy=ebm_wm
```

## Checkpointing & Exact Auto-Resume

- **Per-step checkpointing (latest only):** Every `logging.save_every_n_steps` (default `1000` steps), overwrites `last.ckpt` (full Lightning state: model, optimizer, scheduler, epoch, step, and dataset batch cursor) and `weights_latest.pt` (inference weights for `eval_ebm.py`). Old checkpoints are not accumulated.
- **Automatic resume (`training.auto_resume=true`):** Re-running `train_ebm.py` with the same `output_model_name` automatically detects `last.ckpt`, restores model/optimizer/scheduler state, resumes the same WandB run ID (`wandb_run_id.txt`), and uses `ResumableDistributedSampler` ([resumable_sampler.py](resumable_sampler.py)) to skip already-processed batches and continue from the **exact mid-epoch dataset position** (single-GPU and DDP).

## Research Logs (WandB + `metrics.jsonl`)

Every run logs to WandB (when `wandb.enabled=true`) and persists epoch summaries to `metrics.jsonl`:
- **Losses:** `train/mse_loss`, `val/mse_loss` (main denoising MSE), `train/idm_mse_loss`, `val/idm_mse_loss` (IDM action MSE), `sigreg_loss`, `loss`, `diag/idm_cos_sim`, `diag/avg_ep_frames`
- **Denoising quality:** `diag/baseline_identity_mse`, `diag/denoise_ratio`, gamma-stratified MSE (`low_gamma`, `mid_gamma`, `high_gamma`)
- **Energy landscape:** `diag/energy_clean_mean`, `diag/energy_noisy_mean`, `diag/energy_gap_noisy_minus_clean`, `diag/energy_grad_norm_mean`, `val_probe/energy_gap_rev_minus_real`, `val_probe/energy_gap_shuf_minus_real`
- **Collapse & geometry:** `val_probe/effective_rank`, `diag/raw_emb_batch_std`, `diag/std_emb_batch_std`, `diag/temporal_diff_norm`
- **Periodic validation inpainting probe:** `val_inpaint/latent_mse`, `val_inpaint/latent_cos`, `val_inpaint/action_mse`, `val_inpaint/energy_drop`
