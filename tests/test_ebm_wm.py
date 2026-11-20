"""Comprehensive verification suite for the EBM World Model + ResNet-18 + IDM."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ebm_modules import (
    EBMWorldModel,
    InverseDynamicsModel,
    EBMTransformer,
    ResNet18Encoder,
    compute_effective_rank,
    standardize_tokens,
)
from module import SIGReg
from planner import EBMLatentPlanner, EBMPlanSolver, run_ebm_rollout


def build_tiny_model(embed_dim=192, action_dim=10):
    encoder = ResNet18Encoder(
        embed_dim=embed_dim,
        projector_hidden_dim=512,
        pretrained=False,
        projector_norm="batch",
    )
    ebm = EBMTransformer(
        embed_dim=embed_dim,
        n_layer=6,
        n_head=3,
        head_dim=64,
        mlp_dim=768,
        energy_mlp_dim=768,
        max_seq_len=1024,
        qk_norm=True,
        activation="relu2",
    )
    idm = InverseDynamicsModel(
        embed_dim=embed_dim,
        action_dim=action_dim,
        hidden_dim=256,
    )
    return EBMWorldModel(
        encoder=encoder,
        ebm=ebm,
        idm=idm,
        alpha=1.0,
        logit_normal_mean=-1.5,
        logit_normal_std=0.8,
        noise_scale=2.0,
        detach_target=False,
        detach_idm_input=True,
    )


def test_standardization():
    x = torch.randn(4, 5, 192) * 7.5 + 3.2
    y = standardize_tokens(x)
    assert y.mean(dim=-1).abs().max().item() < 1e-5
    assert (y.std(dim=-1, unbiased=False) - 1.0).abs().max().item() < 1e-4
    print("[PASS] test_standardization")


def test_corruption_schedule():
    torch.manual_seed(0)
    model = build_tiny_model(embed_dim=64)
    clean = standardize_tokens(torch.randn(512, 4, 64))
    out = model.sample_corruption(clean)
    selected = out["selected"]
    gamma = out["gamma"]
    x = out["x"]

    # Every sequence must have >= 1 selected token
    assert (selected.sum(dim=1) >= 1).all(), "Found sequence with 0 selected tokens!"
    # Unselected tokens must remain identical to clean embeddings
    unselected = ~selected
    assert torch.allclose(x[unselected], clean[unselected], atol=1e-6)
    # Median gamma should be near sigmoid(-1.5) ~= 0.1824
    med = gamma.median().item()
    assert 0.14 < med < 0.23, f"Unexpected median gamma: {med}"
    print(f"[PASS] test_corruption_schedule (median_gamma={med:.4f}, mean_sel={selected.float().mean():.3f})")


def test_causal_attention_and_goal_gradient():
    torch.manual_seed(1)
    ebm = EBMTransformer(embed_dim=64, n_layer=3, n_head=2, head_dim=32)
    ebm.eval()

    x = torch.randn(2, 5, 64)
    x_pert = x.clone()
    x_pert[:, 4] += 10.0  # perturb only last (goal) token

    _, tok_e1 = ebm(x)
    _, tok_e2 = ebm(x_pert)

    # Causal check: tokens 0..3 must be completely unaffected by perturbing token 4
    assert torch.allclose(tok_e1[:, :4], tok_e2[:, :4], atol=1e-5), "Causal mask violated!"
    # Token 4 energy MUST change
    assert not torch.allclose(tok_e1[:, 4], tok_e2[:, 4], atol=1e-3)

    # Goal-at-end gradient check: energy of goal token (t=4) depends on intermediate tokens 1..3
    x_req = x.clone().requires_grad_(True)
    _, tok_e = ebm(x_req)
    goal_energy = tok_e[:, 4].sum()
    grad = torch.autograd.grad(goal_energy, x_req)[0]
    assert grad[:, 1:4].norm().item() > 1e-4, "Goal energy gradient did not reach intermediate tokens!"
    print("[PASS] test_causal_attention_and_goal_gradient")


def test_gradient_isolation_and_double_backward():
    torch.manual_seed(2)
    model = build_tiny_model(embed_dim=64, action_dim=10)
    # Build tiny model with 64-d for fast CPU test
    model.encoder = ResNet18Encoder(embed_dim=64, projector_hidden_dim=128, pretrained=False)
    model.ebm = EBMTransformer(embed_dim=64, n_layer=2, n_head=2, head_dim=32)
    model.idm = InverseDynamicsModel(embed_dim=64, action_dim=10, hidden_dim=64)
    sigreg = SIGReg(knots=17, num_proj=64)

    batch = {
        "pixels": torch.randn(4, 4, 3, 64, 64),
        "action": torch.randn(4, 4, 10),
    }
    # Inject a NaN action on last step to verify boundary NaN masking
    batch["action"][0, -1] = float("nan")

    # 1. Verify x is detached before E(x) and e_hat (e_hat does NOT backprop into encoder)
    #    while denoise_loss STILL flows into BOTH encoder (via non-detached target) and ebm (via e_hat), NOT idm
    model.train()
    model.zero_grad()
    out = model.compute_losses(batch, sigreg=sigreg, sigreg_weight=0.0, idm_weight=0.0)
    enc_grads_from_ehat = torch.autograd.grad(
        out["e_hat"].sum(),
        list(model.encoder.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    assert all(g is None for g in enc_grads_from_ehat), "x was NOT detached: e_hat backpropped into encoder!"

    out["denoise_loss"].backward()
    enc_gn = sum(p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None)
    ebm_gn = sum(p.grad.norm().item() for p in model.ebm.parameters() if p.grad is not None)
    idm_gn = sum(p.grad.norm().item() for p in model.idm.parameters() if p.grad is not None)
    assert enc_gn > 0, "Denoise loss produced zero grad on encoder (target should NOT be detached)!"
    assert ebm_gn > 0, "Denoise loss produced zero grad on EBM!"
    assert idm_gn == 0, "Denoise loss leaked grad into IDM!"

    # 2. Verify idm_loss (detached embeddings) flows ONLY into IDM, NOT encoder or ebm
    model.zero_grad()
    out = model.compute_losses(batch, sigreg=sigreg, sigreg_weight=0.0, idm_weight=1.0)
    out["idm_loss"].backward()
    enc_gn = sum(p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None)
    ebm_gn = sum(p.grad.norm().item() for p in model.ebm.parameters() if p.grad is not None)
    idm_gn = sum(p.grad.norm().item() for p in model.idm.parameters() if p.grad is not None)
    assert enc_gn == 0, "IDM loss leaked grad into encoder despite detach_idm_input=True!"
    assert ebm_gn == 0, "IDM loss leaked grad into EBM!"
    assert idm_gn > 0, "IDM loss produced zero grad on IDM!"

    # 3. Verify total loss + eval mode under torch.no_grad()
    model.zero_grad()
    out = model.compute_losses(batch, sigreg=sigreg, sigreg_weight=0.09, idm_weight=1.0)
    out["loss"].backward()
    for k, v in out.items():
        if torch.is_tensor(v) and v.ndim == 0:
            assert torch.isfinite(v), f"Non-finite scalar metric {k}: {v}"

    model.eval()
    with torch.no_grad():
        val_out = model.compute_losses(batch, sigreg=sigreg, sigreg_weight=0.09, idm_weight=1.0)
        assert torch.isfinite(val_out["loss"])
        erank = compute_effective_rank(val_out["emb"])
        assert torch.isfinite(erank) and erank > 1.0

    print(f"[PASS] test_gradient_isolation_and_double_backward (val_erank={erank.item():.2f})")


def test_open_loop_and_closed_loop_planning():
    torch.manual_seed(3)
    encoder = ResNet18Encoder(embed_dim=64, projector_hidden_dim=128, pretrained=False)
    ebm = EBMTransformer(embed_dim=64, n_layer=2, n_head=2, head_dim=32)
    idm = InverseDynamicsModel(embed_dim=64, action_dim=10, hidden_dim=64)
    model = EBMWorldModel(encoder=encoder, ebm=ebm, idm=idm)
    model.eval()

    planner = EBMLatentPlanner(
        model=model,
        n_steps=15,
        lr=0.1,
        num_samples=4,
        init_noise_std=1.0,
        seed=42,
    )

    obs = torch.randn(2, 1, 3, 64, 64)
    goal = torch.randn(2, 1, 3, 64, 64)

    # 1. Open-loop planning (horizon=5 -> 4 intermediate noise tokens, 5 IDM action blocks)
    plan_out = planner.plan_from_pixels(obs, goal, horizon=5)
    assert plan_out["actions"].shape == (2, 5, 10)
    assert plan_out["best_traj"].shape == (2, 6, 64)
    assert plan_out["metrics"]["plan/energy_drop"] > 0.0, "Energy did not decrease during planning!"

    # Verify observed prefix and goal suffix remained untouched
    with torch.no_grad():
        _, obs_e = model.encode_pixels(obs)
        _, goal_e = model.encode_pixels(goal)
    assert torch.allclose(plan_out["best_traj"][:, :1], obs_e, atol=1e-6)
    assert torch.allclose(plan_out["best_traj"][:, -1:], goal_e, atol=1e-6)

    # 2. Closed-loop receding-horizon rollout (total_horizon=5, receding_horizon=1, history_size=3)
    step_counter = {"n": 0}

    def mock_step(action_block):
        assert action_block.shape == (2, 10)
        step_counter["n"] += 1
        next_obs = torch.randn(2, 3, 64, 64)
        return next_obs, False, {"step": step_counter["n"]}

    rollout = run_ebm_rollout(
        step_fn=mock_step,
        init_obs_pixels=obs[:, 0],
        goal_pixels=goal[:, 0],
        planner=planner,
        total_horizon=5,
        receding_horizon=1,
        history_size=3,
        shrink_horizon=True,
        keep_first_state=True,
    )
    assert rollout["steps_executed"] == 5
    assert rollout["actions"].shape == (2, 5, 10)
    assert len(rollout["plan_logs"]) == 5
    # Verify first state was preserved in context when history grew beyond 3
    assert rollout["plan_logs"][-1]["ctx_len"] == 3

    # 3. EBMPlanSolver adapter check
    solver = EBMPlanSolver(model=model, n_steps=5, num_samples=2, device="cpu", seed=0)
    class DummyCfg:
        horizon = 5
    solver.configure(n_envs=2, config=DummyCfg())
    res = solver.solve({"pixels": obs, "goal": goal})
    assert res["actions"].shape == (2, 5, 10)

    # 4. Validation inpainting probe
    with torch.no_grad():
        _, seq_emb = model.encode_pixels(torch.randn(3, 4, 3, 64, 64))
    inpaint = planner.validate_inpainting(seq_emb, true_actions=torch.randn(3, 4, 10))
    assert "val_inpaint/latent_mse" in inpaint and "val_inpaint/action_mse" in inpaint

    print(
        f"[PASS] test_open_loop_and_closed_loop_planning "
        f"(energy_drop={plan_out['metrics']['plan/energy_drop']:.3f})"
    )


def test_tiny_and_large_ebm_shapes():
    # Default LeWM-width tiny model (6L, d=192, 3x64)
    tiny = build_tiny_model(embed_dim=192, action_dim=10)
    n_tiny = sum(p.numel() for p in tiny.parameters())
    out = tiny.compute_losses(
        {"pixels": torch.randn(2, 4, 3, 224, 224), "action": torch.randn(2, 4, 10)},
        sigreg=SIGReg(knots=17, num_proj=64),
    )
    assert out["emb"].shape == (2, 4, 192)
    assert out["pred_actions"].shape == (2, 3, 10)

    # Verify full 15L x 1664d x 13 heads x 128 head_dim architecture matches spec
    full_ebm = EBMTransformer(
        embed_dim=1664,
        n_layer=15,
        n_head=13,
        head_dim=128,
        mlp_dim=6656,
        energy_mlp_dim=6656,
        max_seq_len=1024,
    )
    assert len(full_ebm.blocks) == 15
    assert full_ebm.n_head * full_ebm.head_dim == 1664
    n_full = sum(p.numel() for p in full_ebm.parameters())
    with torch.no_grad():
        se, te = full_ebm(torch.randn(1, 4, 1664))
    assert se.shape == (1,) and te.shape == (1, 4) and (te > 0).all()

    print(
        f"[PASS] test_tiny_and_large_ebm_shapes "
        f"(tiny_total_params={n_tiny/1e6:.2f}M, large_ebm_params={n_full/1e6:.1f}M)"
    )


def test_mini_training_step_decreases_loss():
    torch.manual_seed(7)
    encoder = ResNet18Encoder(embed_dim=64, projector_hidden_dim=128, pretrained=False)
    ebm = EBMTransformer(embed_dim=64, n_layer=2, n_head=2, head_dim=32)
    idm = InverseDynamicsModel(embed_dim=64, action_dim=10, hidden_dim=64)
    model = EBMWorldModel(encoder=encoder, ebm=ebm, idm=idm)
    sigreg = SIGReg(knots=17, num_proj=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = {
        "pixels": torch.randn(4, 4, 3, 64, 64),
        "action": torch.randn(4, 4, 10),
    }

    losses = []
    for _ in range(6):
        opt.zero_grad()
        out = model.compute_losses(batch, sigreg=sigreg, sigreg_weight=0.09, idm_weight=1.0)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(out["loss"].item())

    assert all(math.isfinite(v) for v in losses), f"Non-finite training losses: {losses}"
    print(f"[PASS] test_mini_training_step_decreases_loss (start={losses[0]:.3f} -> end={losses[-1]:.3f})")


if __name__ == "__main__":
    test_standardization()
    test_corruption_schedule()
    test_causal_attention_and_goal_gradient()
    test_gradient_isolation_and_double_backward()
    test_open_loop_and_closed_loop_planning()
    test_tiny_and_large_ebm_shapes()
    test_mini_training_step_decreases_loss()
    print("\nALL EBM-WM TESTS PASSED!")
