"""JEPA Implementation"""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        belief_encoder=None,
        belief_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.belief_encoder = belief_encoder
        self.belief_proj = belief_proj or nn.Identity()

    def encode_frames(self, pixels):
        """Encode observation frames into single-frame embeddings z_t."""
        pixels = pixels.float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        return rearrange(emb, "(b t) d -> b t d", b=b)

    def encode_belief(self, frame_emb):
        """Encode frame embeddings z_{<=t} into contextualized beliefs b_t."""
        if self.belief_encoder is None:
            return frame_emb
        belief = self.belief_encoder(frame_emb)
        if not isinstance(self.belief_proj, nn.Identity):
            b = belief.size(0)
            belief = self.belief_proj(rearrange(belief, "b t d -> (b t) d"))
            belief = rearrange(belief, "(b t) d -> b t d", b=b)
        return belief

    def encode(self, info):
        """Encode observations into frame embeddings z_t, beliefs b_t, and actions.
        info: dict with pixels and optional action keys
        """
        frame_emb = self.encode_frames(info["pixels"])
        belief = self.encode_belief(frame_emb)
        info["frame_emb"] = frame_emb
        info["belief"] = belief
        info["emb"] = belief

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next belief / state embedding from (b_t, a_t).
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(
        self,
        info,
        action_sequence,
        history_size: Optional[int] = None,
    ):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, H, C, img_H, img_W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the total action sequence length (H - 1 past + horizon future)
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        assert T >= H >= 1, f"Expected T >= H >= 1, got T={T}, H={H}"
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0

        if self.belief_encoder is not None:
            # Encode observation history into current belief b_{H-1}
            init_pixels = info["pixels"][:, 0]
            max_ctx = getattr(self.belief_encoder, "num_frames", None)
            if history_size is not None:
                max_ctx = (
                    min(history_size, max_ctx) if max_ctx is not None else history_size
                )
            if max_ctx is not None and init_pixels.size(1) > max_ctx:
                init_pixels = init_pixels[:, -max_ctx:]

            _init = self.encode({"pixels": init_pixels})
            belief = _init["belief"].unsqueeze(1).expand(B, S, -1, -1)
            info["frame_emb"] = _init["frame_emb"].unsqueeze(1).expand(B, S, -1, -1)
            info["belief"] = info["emb"] = belief

            # Roll out autoregressively from current belief using candidate actions a_{H-1..T-1}
            belief_flat = rearrange(belief, "b s ... -> (b s) ...").clone()
            curr_belief = belief_flat[:, -1:]  # (BS, 1, D)
            plan_actions = rearrange(
                action_sequence[:, :, H - 1 :], "b s ... -> (b s) ..."
            )
            plan_act_emb = self.action_encoder(plan_actions)  # (BS, T - H + 1, A_emb)

            preds = [belief_flat]
            for t in range(plan_act_emb.size(1)):
                curr_belief = self.predict(curr_belief, plan_act_emb[:, t : t + 1])
                preds.append(curr_belief)

            pred_rollout = rearrange(
                torch.cat(preds, dim=1), "(b s) ... -> b s ...", b=B, s=S
            )
            info["predicted_belief"] = info["predicted_emb"] = pred_rollout
            return info

        # Legacy frame-level sliding-window rollout (when belief_encoder is None)
        n_steps = T - H
        HS = history_size if history_size is not None else 3

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between terminal predicted belief/embedding and goal."""
        pred_emb = info_dict["predicted_emb"]  # (B, S, T+1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, 1, 1, dim) or (B, S, T_g, dim)

        if goal_emb.ndim == 3 and pred_emb.ndim == 4:
            goal_emb = goal_emb.unsqueeze(1)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # Encode single goal frame through frame encoder and belief encoder (seq len 1)
        goal_pixels = info_dict["goal"][:, 0]
        if goal_pixels.ndim == 5:  # (B, T_goal, C, H, W)
            goal_pixels = goal_pixels[:, -1:]
        elif goal_pixels.ndim == 4:  # (B, C, H, W)
            goal_pixels = goal_pixels.unsqueeze(1)

        goal = self.encode({"pixels": goal_pixels})

        info_dict["goal_emb"] = goal["emb"].unsqueeze(1)
        info_dict["goal_belief"] = goal["belief"].unsqueeze(1)
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
