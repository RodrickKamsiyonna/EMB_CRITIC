import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations into beliefs, unroll multi-step predictions, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    belief = output["belief"]  # (B, T, D)
    frame_emb = output["frame_emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    if self.model.belief_encoder is not None:
        # Multi-step belief rollout: from every b_t, autoregressively unroll the
        # predictor to predict all future beliefs b_{t+k} ahead of it (up to
        # cfg.rollout_steps if set, otherwise all T - 1 steps ahead).
        T = belief.size(1)
        max_rollout_steps = cfg.get("rollout_steps", None)
        num_rollout_steps = (
            T - 1 if max_rollout_steps is None else min(max_rollout_steps, T - 1)
        )
        assert num_rollout_steps >= 1, f"Need at least 2 timesteps, got T={T}"

        sq_errors = []
        pred = belief[:, :-1]  # starting beliefs b_0 .. b_{T-2}
        for k in range(1, num_rollout_steps + 1):
            state_in = pred if k == 1 else pred[:, :-1]
            act_in = act_emb[:, k - 1 : T - 1]
            pred = self.model.predict(state_in, act_in)
            tgt = belief[:, k:]
            sq_errors.append((pred - tgt).pow(2))

        output["pred_loss"] = torch.cat(sq_errors, dim=1).mean()
    else:
        ctx_belief = belief[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_belief = belief[:, n_preds:]
        pred_belief = self.model.predict(ctx_belief, ctx_act)
        output["pred_loss"] = (pred_belief - tgt_belief).pow(2).mean()

    # Per-timestep belief SIGReg (independent across batch at each temporal position)
    output["sigreg_loss"] = self.sigreg(belief.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Optional frame-level SIGReg (disabled by default)
    frame_sigreg_cfg = cfg.loss.get("frame_sigreg", None)
    if (
        frame_sigreg_cfg is not None
        and frame_sigreg_cfg.get("enabled", False)
        and frame_sigreg_cfg.get("weight", 0.0) > 0
    ):
        output["frame_sigreg_loss"] = self.sigreg(frame_emb.transpose(0, 1))
        output["loss"] = (
            output["loss"]
            + frame_sigreg_cfg.get("weight", 0.0) * output["frame_sigreg_loss"]
        )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
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
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
