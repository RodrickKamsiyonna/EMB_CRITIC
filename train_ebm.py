"""Training script for EBM World Model + ResNet-18 + IDM (based on LeWM)."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from ebm_modules import compute_effective_rank
from module import SIGReg
from planner import EBMLatentPlanner
from utils import SaveCkptCallback, get_column_normalizer, get_img_preprocessor


def ebm_forward(self, batch, stage, cfg):
    """Forward step for EBMWorldModel: denoising EBM + SIGReg + detached IDM + research logs."""
    output = self.model.compute_losses(
        batch=batch,
        sigreg=self.sigreg,
        sigreg_weight=cfg.loss.sigreg.weight,
        sigreg_target=cfg.loss.sigreg.target,
        idm_weight=cfg.loss.idm.weight,
        compute_diagnostics=cfg.logging.compute_diagnostics,
    )

    # Log scalar losses and research diagnostics
    log_payload = {}
    for k, v in output.items():
        if torch.is_tensor(v) and v.ndim == 0:
            log_payload[f"{stage}/{k}"] = v.detach()

    on_step = stage == "train"
    self.log_dict(log_payload, on_step=on_step, on_epoch=True, sync_dist=True)

    # Stash a small validation sample for epoch-end research probes (erank, shuffle gap, inpainting)
    if stage == "val" and hasattr(self, "_val_probe_cache"):
        cache = self._val_probe_cache
        if cache["n_batches"] < cfg.logging.val_probe_max_batches:
            cache["embs"].append(output["emb"].detach().cpu())
            cache["actions"].append(batch["action"].detach().cpu())
            cache["n_batches"] += 1

    return output


class EBMResearchCallback(Callback):
    """Research logging callback: JSONL persistence, effective rank, temporal shuffle gap,
    and periodic trajectory-inpainting planning probes on validation data.
    """

    def __init__(self, run_dir: Path, cfg):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.metrics_file = self.run_dir / "metrics.jsonl"

    def on_fit_start(self, trainer, pl_module):
        pl_module._val_probe_cache = {"embs": [], "actions": [], "n_batches": 0}
        if trainer.is_global_zero:
            model = pl_module.model
            counts = {
                "params/encoder": sum(p.numel() for p in model.encoder.parameters()),
                "params/ebm": sum(p.numel() for p in model.ebm.parameters()),
                "params/idm": sum(p.numel() for p in model.idm.parameters()),
                "params/total": sum(p.numel() for p in model.parameters()),
            }
            print("[EBM-WM] Parameter counts:", json.dumps(counts, indent=2))
            if trainer.logger is not None:
                trainer.logger.log_metrics(counts, step=0)
            self._append_jsonl({"event": "fit_start", "epoch": 0, **counts})

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._val_probe_cache = {"embs": [], "actions": [], "n_batches": 0}

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        cache = getattr(pl_module, "_val_probe_cache", None)
        extra_metrics = {}

        if cache and len(cache["embs"]) > 0:
            device = pl_module.device
            model = pl_module.model
            all_embs = torch.cat(cache["embs"], dim=0)  # (N, T, D)
            all_acts = torch.cat(cache["actions"], dim=0)

            # 1. Representation effective rank (collapse detector)
            extra_metrics["val_probe/effective_rank"] = float(compute_effective_rank(all_embs).item())

            # 2. Temporal dynamics energy gap: real vs reversed vs cross-batch shuffled trajectories
            probe_n = min(all_embs.size(0), int(self.cfg.logging.val_probe_samples))
            sample_embs = all_embs[:probe_n].to(device)
            sample_acts = all_acts[:probe_n].to(device)

            was_training = model.training
            model.eval()
            with torch.no_grad():
                e_real, _ = model.compute_energy(sample_embs)
                # Time-reversed sequences
                e_rev, _ = model.compute_energy(torch.flip(sample_embs, dims=[1]))
                # Cross-trajectory shuffled sequences (each timestep from a different episode)
                shuf = sample_embs.clone()
                if probe_n > 1:
                    for t_idx in range(1, shuf.size(1)):
                        perm = torch.roll(torch.arange(probe_n, device=device), shifts=t_idx)
                        shuf[:, t_idx] = sample_embs[perm, t_idx]
                e_shuf, _ = model.compute_energy(shuf)

                extra_metrics["val_probe/energy_real"] = float(e_real.mean().item())
                extra_metrics["val_probe/energy_reversed"] = float(e_rev.mean().item())
                extra_metrics["val_probe/energy_shuffled"] = float(e_shuf.mean().item())
                extra_metrics["val_probe/energy_gap_rev_minus_real"] = float((e_rev - e_real).mean().item())
                extra_metrics["val_probe/energy_gap_shuf_minus_real"] = float((e_shuf - e_real).mean().item())

            # 3. Periodic trajectory-inpainting planning probe (e_0, noise..noise, e_{T-1})
            every_n = int(self.cfg.logging.val_inpaint_every_n_epochs)
            epoch_num = trainer.current_epoch + 1
            if every_n > 0 and (epoch_num % every_n == 0 or epoch_num == trainer.max_epochs):
                inpaint_n = min(probe_n, int(self.cfg.logging.val_inpaint_samples))
                planner = EBMLatentPlanner(
                    model=model,
                    n_steps=int(self.cfg.logging.val_inpaint_steps),
                    lr=float(self.cfg.logging.val_inpaint_lr),
                    num_samples=int(self.cfg.logging.val_inpaint_restarts),
                    init_noise_std=float(self.cfg.logging.val_inpaint_noise_std),
                    seed=int(self.cfg.seed),
                )
                inpaint_metrics = planner.validate_inpainting(
                    clean_emb=sample_embs[:inpaint_n],
                    true_actions=sample_acts[:inpaint_n],
                )
                extra_metrics.update(inpaint_metrics)

            if was_training:
                model.train()

        if extra_metrics:
            pl_module.log_dict(extra_metrics, on_epoch=True, sync_dist=False)

        # Persist epoch summary to JSONL
        record = {"epoch": trainer.current_epoch + 1, "global_step": trainer.global_step}
        for k, v in trainer.callback_metrics.items():
            if torch.is_tensor(v) and v.ndim == 0:
                record[k] = float(v.detach().cpu().item())
        record.update(extra_metrics)
        self._append_jsonl(record)

    def _append_jsonl(self, record: dict):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


@hydra.main(version_base=None, config_path="./config/train", config_name="ebm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        # IDM predicts the frameskip-concatenated action block between consecutive frames
        cfg.model.idm.action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    pl_module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(ebm_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=cfg.logging.save_every_n_epochs,
    )
    research_callback = EBMResearchCallback(run_dir=run_dir, cfg=cfg)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, research_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=pl_module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
