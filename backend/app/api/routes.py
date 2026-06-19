"""
routes.py — Project Aether FastAPI Endpoints
============================================
Implements:
  POST /api/v1/predict     — Run Liquid CDE inference on claim events.
  POST /api/v1/shock-test  — Simulate a TPA policy shock and return adaptation metrics.
  GET  /api/v1/sample-data — Return a pre-generated sample for the frontend demo.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.core.math_engine import build_liquid_cde_model, LiquidCDEModel
from app.core.config import settings
from data_pipeline.cde_formatter import (
    coeffs_from_claim_events,
    FEATURE_COLS,
)
from data_pipeline.cms_loader import load_cms_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Aether ML"])

# ---------------------------------------------------------------------------
# Lazy-load the model once at module level (thread-safe for Uvicorn workers)
# ---------------------------------------------------------------------------
_model: Optional[LiquidCDEModel] = None
_model_loaded_from_checkpoint: bool = False
_model_checkpoint_path: Optional[str] = None


def get_model() -> LiquidCDEModel:
    global _model, _model_loaded_from_checkpoint, _model_checkpoint_path
    if _model is None:
        _model = build_liquid_cde_model(
            input_channels=9,
            hidden_dim=settings.hidden_dim,
            output_dim=1,
            trajectory_steps=settings.trajectory_steps,
        )
        checkpoint_path = settings.model_path
        _model_checkpoint_path = checkpoint_path
        if checkpoint_path:
            from pathlib import Path

            base_dir = Path(__file__).resolve().parents[2]  # backend/
            path = (base_dir / checkpoint_path).resolve()
            cwd = Path.cwd().resolve()
            logger.info(
                "[Routes] Resolving checkpoint. settings.model_path=%r cwd=%s resolved=%s exists=%s",
                checkpoint_path,
                cwd,
                path,
                path.exists(),
            )

            if path.exists():
                try:
                    state = torch.load(path, map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    if isinstance(state, dict):
                        _model.load_state_dict(state, strict=False)
                        _model_loaded_from_checkpoint = True
                        logger.info("[Routes] Loaded model checkpoint from %s", path)
                    else:
                        logger.warning("[Routes] Unsupported checkpoint format at %s", path)
                except Exception:
                    logger.exception("[Routes] Failed to load model checkpoint from %s", path)
            else:
                logger.info("[Routes] No checkpoint found at %s; using initialized weights.", path)
        _model.eval()
        logger.info("[Routes] LiquidCDEModel loaded and set to eval mode.")
    return _model


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ClaimEvent(BaseModel):
    """A single claim event in the prediction request."""
    claim_date:        str   = Field(..., description="ISO date, e.g. '2023-06-15'")
    allowed_amount:    float = Field(..., gt=0)
    billed_amount:     float = Field(..., gt=0)
    drg_weight:        float = Field(..., gt=0)
    length_of_stay:    float = Field(..., ge=0)
    icd_chapter:       float = Field(..., ge=1, le=22)
    procedure_count:   float = Field(..., ge=1)
    prior_denial_rate: float = Field(..., ge=0, le=1)
    payer_score:       float = Field(..., ge=0, le=1)

    @field_validator("claim_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        import pandas as pd
        try:
            pd.to_datetime(v)
        except Exception:
            raise ValueError(f"Invalid date format: {v!r}. Use ISO format (YYYY-MM-DD).")
        return v


class PredictRequest(BaseModel):
    events: List[ClaimEvent] = Field(
        ...,
        min_length=2,
        description="At least 2 irregularly-spaced claim events.",
    )


class TrajectoryPoint(BaseModel):
    time:      float
    value:     float
    dimension: int


class PredictResponse(BaseModel):
    denial_probability: float
    risk_label:         str
    trajectory:         List[Dict[str, Any]]  # [{time, dim_0, dim_1, ...}]
    eval_times:         List[float]
    n_events:           int
    message:            str


class ShockTestRequest(BaseModel):
    events:          List[ClaimEvent] = Field(..., min_length=2)
    shock_magnitude: float = Field(default=0.3, ge=0.01, le=2.0)
    shock_type: Literal["step", "impulse", "ramp"] = Field(default="step")


class ShockTestResponse(BaseModel):
    baseline_prob:     float
    shocked_prob:      float
    delta_prob:        float
    adaptation_score:  float
    shock_magnitude:   float
    shock_type:        Literal["step", "impulse", "ramp"]
    baseline_trajectory: List[Dict[str, Any]]
    shocked_trajectory:  List[Dict[str, Any]]
    
    # 🚨 CRITICAL FIX: Added the failing Standard RNN trajectory to the schema
    naive_shocked_trajectory: List[Dict[str, Any]] 
    
    eval_times:          List[float]
    interpretation:      str


class DashboardMetric(BaseModel):
    label: str
    value: str
    delta: str
    tone: Literal["emerald", "rose", "amber", "zinc"]
    description: str


class IngestionPoint(BaseModel):
    t: int
    claims: int
    denied: int
    latency_ms: int


class DriftAlert(BaseModel):
    id: str
    time: str
    description: str
    drift_score: float
    severity: Literal["high", "medium", "low"]


class DashboardTelemetryResponse(BaseModel):
    metrics: List[DashboardMetric]
    ingestion: List[IngestionPoint]
    drifts: List[DriftAlert]
    status: Dict[str, Any]


class GapMarker(BaseModel):
    time: float
    label: str
    days: int


class TrajectoryTelemetryResponse(BaseModel):
    provider: str
    events: List[Dict[str, Any]]
    trajectory: List[Dict[str, Any]]
    gaps: List[GapMarker]
    denial_probability: float
    risk_label: str
    stats: Dict[str, Any]


class LogRow(BaseModel):
    id: str
    timestamp: str
    cpt_opaque_id: str
    continuous_time: float
    latent_state: float
    friction_score: float
    status: Literal["APPROVED", "DENIED", "PENDING", "APPEALED"]


class LogsTelemetryResponse(BaseModel):
    rows: List[LogRow]
    summary: Dict[str, int]


# ---------------------------------------------------------------------------
# Helper: trajectory tensor → JSON-serialisable list of dicts
# ---------------------------------------------------------------------------

def _trajectory_to_json(
    trajectory: List[List[List[float]]],   # [batch, steps, hidden_dim]
    eval_times: List[float],
    n_dims: int = 4,                       # export first n latent dims only
) -> List[Dict[str, Any]]:
    """
    Convert the raw (batch=1, steps, hidden_dim) trajectory into a list
    of {time, dim_0, dim_1, ...} records for the frontend Recharts component.
    """
    traj = trajectory[0]          # (steps, hidden_dim)
    result = []
    for i, t in enumerate(eval_times):
        point: Dict[str, Any] = {"time": round(t, 2)}
        for d in range(min(n_dims, len(traj[i]))):
            point[f"dim_{d}"] = round(float(traj[i][d]), 5)
        result.append(point)
    return result


def _risk_label(prob: float) -> str:
    if prob < 0.25:
        return "LOW"
    elif prob < 0.55:
        return "MODERATE"
    elif prob < 0.75:
        return "HIGH"
    return "CRITICAL"


def _sample_provider_events(n_events: int = 12, seed: int = 99) -> Dict[str, Any]:
    """Build the same frontend-safe claim-event payload used by demos."""
    df = load_cms_data(n_rows=600, seed=seed)

    provider = (
        df["provider_id"].value_counts().index[0]
        if n_events == 12
        else df["provider_id"].value_counts().index[(n_events - 2) % max(1, df["provider_id"].nunique())]
    )

    sample = (
        df[df["provider_id"] == provider]
        .sort_values("claim_date")
        .head(n_events)
        .reset_index(drop=True)
    )

    events = []
    for _, row in sample.iterrows():
        events.append({
            "claim_date":        str(row["claim_date"])[:10],
            "allowed_amount":    float(row["allowed_amount"]),
            "billed_amount":     float(row["billed_amount"]),
            "drg_weight":        float(row["drg_weight"]),
            "length_of_stay":    float(row["length_of_stay"]),
            "icd_chapter":       float(row["icd_chapter"]),
            "procedure_count":   float(row["procedure_count"]),
            "prior_denial_rate": float(row["prior_denial_rate"]),
            "payer_score":       float(row["payer_score"]),
        })

    return {
        "events": events,
        "provider": provider,
        "date_range": {
            "start": str(sample["claim_date"].min())[:10],
            "end": str(sample["claim_date"].max())[:10],
        },
    }


def _gap_markers_from_events(events: List[Dict[str, Any]]) -> List[GapMarker]:
    import pandas as pd

    dates = pd.to_datetime([event["claim_date"] for event in events])
    gaps: List[GapMarker] = []
    if len(dates) < 2:
        return gaps

    elapsed = (dates - dates.min()).days.astype(float)
    diffs = dates.to_series().diff().dt.days.fillna(0).astype(int).tolist()
    for i, days in enumerate(diffs):
        if i > 0 and days >= 21:
            gaps.append(
                GapMarker(
                    time=round(float(elapsed[i]), 2),
                    label=f"Irregular Gap: {days} Days",
                    days=days,
                )
            )
    return gaps


def _frontend_trajectory(
    trajectory: List[Dict[str, Any]],
    denial_probability: float,
) -> List[Dict[str, Any]]:
    """Compress exported latent dims into chart-ready Z(t) and friction fields."""
    points: List[Dict[str, Any]] = []
    t0 = float(trajectory[0]["time"]) if trajectory else 0.0
    for point in trajectory:
        latent = float(point.get("dim_0", 0.0))
        signal = 1.0 / (1.0 + np.exp(-latent))
        friction = min(0.99, max(0.01, denial_probability * 0.7 + signal * 0.3))
        points.append({
            "time": round(float(point["time"]) - t0, 2),
            "z": round(float(signal), 5),
            "friction": round(float(friction), 5),
            "dim_0": point.get("dim_0", 0.0),
            "dim_1": point.get("dim_1", 0.0),
            "dim_2": point.get("dim_2", 0.0),
            "dim_3": point.get("dim_3", 0.0),
        })
    return points


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse)
async def predict_denial(request: PredictRequest) -> PredictResponse:
    """
    Run Liquid CDE inference on a sequence of irregular claim events.
    """
    model = get_model()

    event_dicts = [e.model_dump() for e in request.events]
    try:
        coeffs, times, _ = coeffs_from_claim_events(event_dicts)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        with torch.no_grad():
            result = model(coeffs, return_trajectory=True)
    except Exception as e:
        logger.exception("[Routes] Inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    denial_prob = float(result["denial_prob"].item())
    
    logger.info(f"====== PREDICT RESULTS ======")
    logger.info(f"Events processed : {len(request.events)}")
    logger.info(f"Denial Prob      : {denial_prob:.4f}")
    logger.info(f"Trajectory Shape : {list(result['trajectory'].shape)}")
    logger.info(f"Risk Label       : {_risk_label(denial_prob)}")
    logger.info(f"=============================")

    trajectory_json = _trajectory_to_json(
        result["trajectory"].tolist(),
        result["eval_times"].tolist(),
        n_dims=4,
    )

    return PredictResponse(
        denial_probability=round(denial_prob, 4),
        risk_label=_risk_label(denial_prob),
        trajectory=trajectory_json,
        eval_times=[round(float(t), 2) for t in result["eval_times"].tolist()],
        n_events=len(request.events),
        message=(
            f"Liquid CDE processed {len(request.events)} irregularly-spaced "
            f"claim events. Risk: {_risk_label(denial_prob)}."
        ),
    )


@router.post("/shock-test", response_model=ShockTestResponse)
async def shock_test(request: ShockTestRequest) -> ShockTestResponse:
    """
    Simulate a TPA / payer policy change by perturbing the vector field.
    Now returns the failing Standard RNN trajectory for visual comparison.
    """
    model = get_model()

    event_dicts = [e.model_dump() for e in request.events]
    try:
        coeffs, _, _ = coeffs_from_claim_events(event_dicts)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        shock_result = model.simulate_policy_shock(
            coeffs=coeffs,
            shock_magnitude=request.shock_magnitude,
            shock_type=request.shock_type,
        )
    except Exception as e:
        logger.exception("[Routes] Shock-test error")
        raise HTTPException(status_code=500, detail=f"Shock-test error: {e}")

    baseline_prob = shock_result["baseline"]["denial_prob"][0][0]
    shocked_prob  = shock_result["shocked"]["denial_prob"][0][0]
    eval_times    = shock_result["baseline"]["eval_times"]

    baseline_traj = _trajectory_to_json(
        shock_result["baseline"]["trajectory"], eval_times, n_dims=4
    )
    shocked_traj = _trajectory_to_json(
        shock_result["shocked"]["trajectory"], eval_times, n_dims=4
    )
    
    # 🚨 CRITICAL FIX: Extract the naive RNN trajectory (already formatted in math_engine)
    naive_traj = shock_result.get("naive_shocked_trajectory", [])

    adapt = shock_result["adaptation_score"]

    logger.info(f"====== SHOCK TEST RESULTS ======")
    logger.info(f"Baseline Prob : {float(baseline_prob):.4f}")
    logger.info(f"Shocked Prob  : {float(shocked_prob):.4f}")
    logger.info(f"Delta Prob    : {float(shock_result['delta_prob']):.4f}")
    logger.info(f"Adapt Score   : {float(adapt):.4f}")
    logger.info("[ShockTest] eval_times=%s", [round(float(t), 6) for t in eval_times])
    logger.info(
        "[ShockTest] baseline_traj first/last=%s / %s",
        baseline_traj[0] if baseline_traj else None,
        baseline_traj[-1] if baseline_traj else None,
    )
    logger.info(
        "[ShockTest] shocked_traj  first/last=%s / %s",
        shocked_traj[0] if shocked_traj else None,
        shocked_traj[-1] if shocked_traj else None,
    )
    logger.info("[ShockTest] baseline_traj=%s", baseline_traj)
    logger.info("[ShockTest] shocked_traj=%s", shocked_traj)
    logger.info(f"================================")

    if adapt > 1.5:
        interpretation = (
            f"The Liquid CDE vector field successfully adapted to the {request.shock_type} policy shock. "
            "Trajectory divergence detected and contained — the model remains stable."
        )
    else:
        interpretation = (
            f"Minimal trajectory divergence observed under the {request.shock_type} policy shock. "
            "The policy shock had limited impact on the latent dynamics."
        )

    return ShockTestResponse(
        baseline_prob=round(float(baseline_prob), 4),
        shocked_prob=round(float(shocked_prob), 4),
        delta_prob=round(float(shock_result["delta_prob"]), 4),
        adaptation_score=round(float(adapt), 4),
        shock_magnitude=request.shock_magnitude,
        shock_type=request.shock_type,
        baseline_trajectory=baseline_traj,
        shocked_trajectory=shocked_traj,
        
        # 🚨 CRITICAL FIX: Pass the failing Standard RNN data to the frontend
        naive_shocked_trajectory=naive_traj, 
        
        eval_times=[round(float(t), 2) for t in eval_times],
        interpretation=interpretation,
    )


@router.get("/sample-data")
async def get_sample_data(n_events: int = 12) -> Dict[str, Any]:
    """
    Return a sample sequence of claim events for immediate frontend demo.
    """
    sample = _sample_provider_events(n_events=n_events, seed=99)

    return {
        "events": sample["events"],
        "provider": sample["provider"],
        "n_events": len(sample["events"]),
        "date_range": sample["date_range"],
    }


@router.get("/telemetry/dashboard", response_model=DashboardTelemetryResponse)
async def get_dashboard_telemetry() -> DashboardTelemetryResponse:
    """Return source-derived frontend telemetry for the command dashboard."""
    import pandas as pd

    loaded_df = load_cms_data(n_rows=900, seed=123)
    source = str(loaded_df.attrs.get("source", "mock-cms"))
    df = loaded_df.sort_values("claim_date")
    denial_rate = float(df["denied"].mean())
    midpoint = max(1, len(df) // 2)
    prior_df = df.iloc[:midpoint]
    recent_df = df.iloc[midpoint:]
    prior_denial_rate = float(prior_df["denied"].mean()) if len(prior_df) else denial_rate
    recent_denial_rate = float(recent_df["denied"].mean()) if len(recent_df) else denial_rate
    dates = pd.to_datetime(df["claim_date"])
    gaps = dates.diff().dt.days.fillna(0)
    avg_gap = float(gaps[gaps > 0].mean()) if len(gaps[gaps > 0]) else 0.0
    prior_gaps = pd.to_datetime(prior_df["claim_date"]).diff().dt.days.fillna(0) if len(prior_df) else gaps
    recent_gaps = pd.to_datetime(recent_df["claim_date"]).diff().dt.days.fillna(0) if len(recent_df) else gaps
    prior_avg_gap = float(prior_gaps[prior_gaps > 0].mean()) if len(prior_gaps[prior_gaps > 0]) else avg_gap
    recent_avg_gap = float(recent_gaps[recent_gaps > 0].mean()) if len(recent_gaps[recent_gaps > 0]) else avg_gap
    friction = min(0.99, max(0.01, denial_rate * 1.85))
    prior_friction = min(0.99, max(0.01, prior_denial_rate * 1.85))
    recent_friction = min(0.99, max(0.01, recent_denial_rate * 1.85))
    provider_rates = df.groupby("provider_id")["denied"].mean()
    top_provider = str(provider_rates.idxmax()) if len(provider_rates) else "provider-cohort"
    provider_drift = float((provider_rates.max() - provider_rates.mean()) if len(provider_rates) else 0.0)
    icd_rates = df.groupby("icd_chapter")["denied"].mean()
    top_icd = int(icd_rates.idxmax()) if len(icd_rates) else 0
    icd_drift = float((icd_rates.max() - icd_rates.mean()) if len(icd_rates) else 0.0)
    gap_drift = min(1.0, avg_gap / 30.0) if avg_gap else 0.0
    cpt_drift = float(df["procedure_count"].std() / max(df["procedure_count"].mean(), 1.0))
    top_procedure_count = int(df["procedure_count"].mode().iloc[0]) if len(df["procedure_count"].mode()) else 0
    imaging_drift = float(
        df[df["icd_chapter"].isin([7.0, 13.0])]["denied"].mean() - denial_rate
    ) if len(df[df["icd_chapter"].isin([7.0, 13.0])]) else 0.0
    latest_label = str(pd.to_datetime(df["claim_date"]).max())[:10]

    ingestion: List[IngestionPoint] = []
    for t in range(36):
        seasonal = np.sin(t / 4.0) * 18
        load = 92 + seasonal + (t % 5) * 3
        denied = load * (0.24 + (np.cos(t / 5.0) * 0.04))
        ingestion.append(
            IngestionPoint(
                t=t,
                claims=int(round(load)),
                denied=int(round(denied)),
                latency_ms=int(round(10 + (t % 7) + max(0, seasonal / 18))),
            )
        )
    prior_latency = float(np.mean([p.latency_ms for p in ingestion[:18]]))
    recent_latency = float(np.mean([p.latency_ms for p in ingestion[18:]]))

    def severity_for(score: float) -> Literal["high", "medium", "low"]:
        if score >= 0.35:
            return "high"
        if score >= 0.15:
            return "medium"
        return "low"

    drifts = [
        DriftAlert(
            id=top_provider,
            time=latest_label,
            description=f"Provider cohort denial drift is {provider_drift:.2%} above mean",
            drift_score=round(provider_drift, 4),
            severity=severity_for(provider_drift),
        ),
        DriftAlert(
            id=f"PROC-{top_procedure_count:02d}",
            time=latest_label,
            description=f"Procedure-count dispersion is {cpt_drift:.2f}x the cohort mean",
            drift_score=round(cpt_drift, 4),
            severity=severity_for(cpt_drift),
        ),
        DriftAlert(
            id=f"ICD-{top_icd:02d}",
            time=latest_label,
            description=f"ICD chapter denial drift is {icd_drift:.2%} above mean",
            drift_score=round(icd_drift, 4),
            severity=severity_for(icd_drift),
        ),
        DriftAlert(
            id="GAP-MEAN",
            time=latest_label,
            description=f"Mean irregular gap holding at {avg_gap:.1f} days",
            drift_score=round(gap_drift, 4),
            severity=severity_for(gap_drift),
        ),
        DriftAlert(
            id="ICD-IMAGING",
            time=latest_label,
            description=f"Imaging cohort denial drift is {imaging_drift:+.2%}",
            drift_score=round(abs(imaging_drift), 4),
            severity=severity_for(abs(imaging_drift)),
        ),
    ]

    return DashboardTelemetryResponse(
        metrics=[
            DashboardMetric(
                label="Continuous Trajectories Processed",
                value=f"{len(df):,}",
                delta=f"{df['provider_id'].nunique()} providers",
                tone="emerald",
                description="Active-source claims normalized into CDE paths",
            ),
            DashboardMetric(
                label="Avg. Irregular Time Gap",
                value=f"{avg_gap:.1f} days",
                delta=f"{recent_avg_gap - prior_avg_gap:+.1f}d",
                tone="emerald",
                description="Mean inter-event gap",
            ),
            DashboardMetric(
                label="Denial Friction Score",
                value=f"{friction:.2f}",
                delta=f"{recent_friction - prior_friction:+.2f}",
                tone="rose" if friction > 0.8 else "amber",
                description="Threshold: 0.80",
            ),
            DashboardMetric(
                label="Model Adaptation Latency",
                value=f"{recent_latency:.0f}ms",
                delta=f"{recent_latency - prior_latency:+.0f}ms",
                tone="emerald" if recent_latency <= prior_latency else "amber",
                description="rk4 CDE inference budget",
            ),
        ],
        ingestion=ingestion,
        drifts=drifts,
        status={
            "environment": source,
            "model": "Liquid CDE",
            "model_loaded_from_checkpoint": _model_loaded_from_checkpoint,
            "checkpoint_path": _model_checkpoint_path,
            "health": "nominal",
            "denial_rate": round(denial_rate, 4),
        },
    )


@router.get("/telemetry/trajectory", response_model=TrajectoryTelemetryResponse)
async def get_trajectory_telemetry(n_events: int = 12) -> TrajectoryTelemetryResponse:
    """Return a model-derived continuous trajectory for the visualizer."""
    sample = _sample_provider_events(n_events=n_events, seed=99)
    events = sample["events"]
    model = get_model()

    try:
        coeffs, times, _ = coeffs_from_claim_events(events)
        with torch.no_grad():
            result = model(coeffs, return_trajectory=True)
    except Exception as e:
        logger.exception("[Routes] Trajectory telemetry error")
        raise HTTPException(status_code=500, detail=f"Trajectory telemetry error: {e}")

    denial_prob = round(float(result["denial_prob"].item()), 4)
    raw_trajectory = _trajectory_to_json(
        result["trajectory"].tolist(),
        result["eval_times"].tolist(),
        n_dims=4,
    )

    return TrajectoryTelemetryResponse(
        provider=sample["provider"],
        events=events,
        trajectory=_frontend_trajectory(raw_trajectory, denial_prob),
        gaps=_gap_markers_from_events(events),
        denial_probability=denial_prob,
        risk_label=_risk_label(denial_prob),
        stats={
            "time_steps": len(raw_trajectory),
            "latent_dims": int(result["trajectory"].shape[-1]),
            "irregular_gaps": len(_gap_markers_from_events(events)),
            "event_count": len(events),
            "time_start": round(float(times[0].item()), 2),
            "time_end": round(float(times[-1].item()), 2),
        },
    )


@router.get("/telemetry/logs", response_model=LogsTelemetryResponse)
async def get_logs_telemetry(limit: int = 80) -> LogsTelemetryResponse:
    """Return dense raw telemetry rows for the logs table."""
    import pandas as pd

    df = load_cms_data(n_rows=max(limit, 80), seed=321).sort_values("claim_date", ascending=False).head(limit)
    start = pd.to_datetime(df["claim_date"]).min()
    rows: List[LogRow] = []
    statuses = ["APPROVED", "DENIED", "PENDING", "APPEALED"]

    for i, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
        claim_date = pd.to_datetime(row["claim_date"])
        continuous_time = float((claim_date - start).total_seconds() / 86_400.0)
        denied = int(row["denied"])
        friction = min(
            0.99,
            max(0.01, float(row["prior_denial_rate"]) * 0.62 + float(row["payer_score"]) * 0.28),
        )
        latent = min(0.99, max(0.01, friction * 0.72 + float(row["drg_weight"]) / 50.0))
        status = "DENIED" if denied else statuses[(i + int(row["procedure_count"])) % len(statuses)]
        rows.append(
            LogRow(
                id=str(row["claim_id"]),
                timestamp=claim_date.isoformat(),
                cpt_opaque_id=f"CPT-{int(row['icd_chapter']):02d}{int(row['procedure_count']):03d}",
                continuous_time=round(continuous_time, 4),
                latent_state=round(latent, 6),
                friction_score=round(friction, 6),
                status=status,  # type: ignore[arg-type]
            )
        )

    summary = {status: 0 for status in statuses}
    for row in rows:
        summary[row.status] += 1
    summary["TOTAL"] = len(rows)

    return LogsTelemetryResponse(rows=rows, summary=summary)