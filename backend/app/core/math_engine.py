"""
math_engine.py — Project Aether Core Mathematical Engine (v8.0 - Pure Continuous-Time Focus)
===========================================================================
Implements the Liquid Neural CDE (Controlled Differential Equation) model.

CRITICAL FIXES APPLIED (v8.0):
  1. Removed LayerNorm from h_final. It was mathematically erasing the magnitude 
     of the shock before it reached the decoder.
  2. Fixed Hermite coefficient mutation. We now multiply ALL 4 polynomial blocks 
     (total_coeffs) instead of just the first block (:channels). This correctly 
     scales the entire continuous path X(t) rather than distorting the polynomial.
  3. Retained softsign activations to prevent tanh saturation while preserving 
     relative magnitude differences.
  4. REMOVED STRAWMAN BASELINE: Dropped the untrained NaiveRNNBaseline. The demo 
     now focuses purely on the mathematically honest adaptation of the Liquid CDE's 
     continuous trajectory under policy shocks.
"""

import copy
import math
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torchcde


# ---------------------------------------------------------------------------
# 1. Liquid Time-Constant (LTC) Cell
# ---------------------------------------------------------------------------

class LTCCell(nn.Module):
    """
    A single Liquid Time-Constant cell.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_xh = nn.Linear(input_dim, hidden_dim, bias=True)

        self.tau_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.amp_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for stability (GELU compatible)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_t: torch.Tensor, dx_dt: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ltc_input = torch.cat([x_t, dx_dt, t], dim=-1)
        
        # Constrain tau to [0.1, 10.0] for biological/clinical plausibility
        tau_raw = self.tau_net(ltc_input)
        tau = 0.1 + 9.9 * torch.sigmoid(tau_raw)
        
        amp = self.amp_net(ltc_input)
        
        # USE softsign instead of tanh. It bounds the output but preserves magnitude.
        gate_input = self.W_hh(h) + self.W_xh(ltc_input)
        gate = torch.nn.functional.softsign(gate_input)
        
        dh = (-h + gate * amp) / tau
        
        # Residual connection to prevent vanishing gradients
        dh = dh + 0.1 * h 
        
        return dh


# ---------------------------------------------------------------------------
# 2. Neural CDE Vector Field
# ---------------------------------------------------------------------------

class LiquidCDEFunc(nn.Module):
    """
    The vector field F(t, h) for the Controlled Differential Equation.
    """

    def __init__(self, input_channels: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        
        # DEVICE FIX: X is injected dynamically during the forward pass.
        self.X = None 

        ltc_input_dim = 2 * input_channels + 1
        self.ltc = LTCCell(input_dim=ltc_input_dim, hidden_dim=hidden_dim)

        self.output_linear = nn.Linear(hidden_dim, hidden_dim * input_channels)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.X is None:
            raise RuntimeError("LiquidCDEFunc.X must be set before integration.")
            
        x_t = self.X.evaluate(t)
        dx_dt = self.X.derivative(t)
        
        t_feat = t.expand(h.shape[0], 1)

        dh = self.ltc(x_t, dx_dt, t_feat, h)

        F = self.output_linear(dh)
        F = F.view(h.shape[0], self.hidden_dim, self.input_channels)
        
        # Use softsign instead of tanh to bound the control matrix without saturating.
        return torch.nn.functional.softsign(F)


# ---------------------------------------------------------------------------
# 3. Full Liquid CDE Model
# ---------------------------------------------------------------------------

class LiquidCDEModel(nn.Module):
    """
    End-to-end Liquid Neural CDE for healthcare claim-denial prediction.
    """

    def __init__(
        self,
        input_channels: int = 9,   
        hidden_dim: int = 64,
        output_dim: int = 1,
        trajectory_steps: int = 100,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.trajectory_steps = trajectory_steps

        self.initial_encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.encoder_residual = nn.Linear(input_channels, hidden_dim) if input_channels != hidden_dim else nn.Identity()

        self.cde_func = LiquidCDEFunc(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, coeffs: torch.Tensor, return_trajectory: bool = False) -> dict:
        X = torchcde.CubicSpline(coeffs)
        t0, tT = X.interval

        x0 = X.evaluate(t0)
        
        # 🚨 FIX: Cleanly compute h0_raw before normalizing to avoid shape reference errors.
        h0_raw = self.initial_encoder(x0) + self.encoder_residual(x0)
        h0 = torch.nn.functional.layer_norm(h0_raw, h0_raw.shape[-1:])

        self.cde_func.X = X

        eval_times = torch.linspace(
            float(t0), float(tT), self.trajectory_steps,
            device=coeffs.device
        )

        h_trajectory = torchcde.cdeint(
            X=X,
            func=self.cde_func,
            z0=h0,
            t=eval_times,
            method="rk4",
            options={"step_size": (float(tT) - float(t0)) / 200},
            adjoint=False,
        )

        h_final = h_trajectory[:, -1, :]
        
        # 🚨 CRITICAL FIX: REMOVED LayerNorm on h_final.
        # The softsign activations in the LTC cell keep h_final naturally bounded.
        # LayerNorm was mathematically erasing the magnitude of the shock.
        
        logits = self.decoder(h_final)

        result = {
            "logits": logits,
            "denial_prob": torch.sigmoid(logits),
        }
        
        if return_trajectory:
            result["trajectory"] = h_trajectory
            result["eval_times"] = eval_times

        return result

    # ------------------------------------------------------------------
    # Policy Shock Simulation (v8.0 - Pure Continuous-Time Focus)
    # ------------------------------------------------------------------

    def simulate_policy_shock(
        self,
        coeffs: torch.Tensor,
        shock_magnitude: float = 2.0,
        shock_type: Literal["step", "impulse", "ramp"] = "step",
        shock_start_ratio: float = 0.7,
    ) -> dict:
        """
        Simulates real-world TPA policy changes by mutating the INPUT DATA PATH.
        Focuses purely on the Liquid CDE's continuous adaptation.
        """
        # 1. BASELINE (Liquid CDE on Normal Data)
        with torch.no_grad():
            baseline = self.forward(coeffs, return_trajectory=True)

        # 2. CREATE THE "SHOCKED" DATA PATH
        shocked_coeffs = coeffs.clone()
        batch_size, num_intervals, total_coeffs = shocked_coeffs.shape
        
        shock_start_idx = int(num_intervals * shock_start_ratio)
        
        # 🚨 CRITICAL FIX: Multiply ALL 4*channels coefficients.
        # Multiplying only :channels only scales the 'a' (constant) term, 
        # which distorts the polynomial. Multiplying all terms scales the 
        # entire polynomial P(t) -> 3*P(t), correctly scaling the path X(t).
        
        if shock_type == "step":
            shocked_coeffs[:, shock_start_idx:, :] *= (shock_magnitude + 1.0)
            
        elif shock_type == "impulse":
            shock_end_idx = shock_start_idx + max(1, int(num_intervals * 0.1))
            shocked_coeffs[:, shock_start_idx:shock_end_idx, :] *= (shock_magnitude * 2.0 + 1.0)
            
        elif shock_type == "ramp":
            remaining_steps = num_intervals - shock_start_idx
            ramp_tensor = torch.linspace(1.0, shock_magnitude + 1.0, remaining_steps, device=coeffs.device)
            ramp_tensor = ramp_tensor.view(1, remaining_steps, 1).expand(batch_size, -1, total_coeffs)
            shocked_coeffs[:, shock_start_idx:, :] *= ramp_tensor

        # 3. LIQUID CDE INFERENCE (Shocked)
        with torch.no_grad():
            shocked = self.forward(shocked_coeffs, return_trajectory=True)

        # 4. METRICS
        delta_prob = (shocked["denial_prob"] - baseline["denial_prob"]).abs().mean().item()
        
        # Normalize trajectories to unit vectors to calculate angular shift
        baseline_traj_norm = baseline["trajectory"] / (baseline["trajectory"].norm(dim=-1, keepdim=True) + 1e-8)
        shocked_traj_norm = shocked["trajectory"] / (shocked["trajectory"].norm(dim=-1, keepdim=True) + 1e-8)
        
        traj_diff = (shocked_traj_norm - baseline_traj_norm).norm(dim=-1)
        adaptation_score = float(traj_diff[:, -1].mean()) 

        return {
            "baseline": {
                "denial_prob": baseline["denial_prob"].tolist(),
                "trajectory": baseline["trajectory"].tolist(),
                "eval_times": baseline["eval_times"].tolist(),
            },
            "shocked": {
                "denial_prob": shocked["denial_prob"].tolist(),
                "trajectory": shocked["trajectory"].tolist(),
                "eval_times": shocked["eval_times"].tolist(),
            },
            "delta_prob": delta_prob,
            "adaptation_score": adaptation_score,
            "shock_magnitude": shock_magnitude,
            "shock_type": shock_type,
            "shock_start_ratio": shock_start_ratio,
        }


# ---------------------------------------------------------------------------
# 4. Model Factory
# ---------------------------------------------------------------------------

def build_liquid_cde_model(
    input_channels: int = 9,
    hidden_dim: int = 64,
    output_dim: int = 1,
    trajectory_steps: int = 100,
    device: Optional[str] = None,
) -> LiquidCDEModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LiquidCDEModel(
        input_channels=input_channels,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        trajectory_steps=trajectory_steps,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Aether] LiquidCDEModel built — "
        f"{n_params:,} trainable parameters on {device}."
    )
    return model