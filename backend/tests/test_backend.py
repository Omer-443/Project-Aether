import pytest
import pandas as pd
import numpy as np
import torch
from fastapi.testclient import TestClient
from app.main import app
from app.core.math_engine import build_liquid_cde_model
from data_pipeline.cde_formatter import prepare_cms_for_cde, coeffs_from_claim_events

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "torch" in data

def test_get_sample_data():
    response = client.get("/api/v1/sample-data?n_events=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data["events"]) == 5
    assert "provider" in data
    assert "date_range" in data

def test_predict_endpoint_valid():
    sample_res = client.get("/api/v1/sample-data?n_events=3")
    events = sample_res.json()["events"]
    
    response = client.post("/api/v1/predict", json={"events": events})
    assert response.status_code == 200
    data = response.json()
    assert "denial_probability" in data
    assert "risk_label" in data
    assert "trajectory" in data
    assert "eval_times" in data
    assert len(data["eval_times"]) > 0
    assert 0 <= data["denial_probability"] <= 1.0

def test_predict_endpoint_invalid_short_sequence():
    sample_res = client.get("/api/v1/sample-data?n_events=1")
    events = sample_res.json()["events"]
    
    response = client.post("/api/v1/predict", json={"events": events})
    assert response.status_code == 422 # FastAPI validation should fail for min_length=2

def test_shock_test_endpoint_valid():
    sample_res = client.get("/api/v1/sample-data?n_events=4")
    events = sample_res.json()["events"]
    
    response = client.post("/api/v1/shock-test", json={"events": events, "shock_magnitude": 0.5})
    assert response.status_code == 200
    data = response.json()
    assert "baseline_prob" in data
    assert "shocked_prob" in data
    assert "adaptation_score" in data
    assert "delta_prob" in data
    assert data["adaptation_score"] >= 0
    assert len(data["baseline_trajectory"]) == len(data["shocked_trajectory"])
    assert len(data["baseline_trajectory"]) == len(data["eval_times"])

def test_dashboard_telemetry_endpoint():
    response = client.get("/api/v1/telemetry/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert len(data["metrics"]) == 4
    assert len(data["ingestion"]) > 10
    assert len(data["drifts"]) > 0
    assert all(isinstance(drift["drift_score"], float) for drift in data["drifts"])
    assert data["status"]["health"] == "nominal"

def test_trajectory_telemetry_endpoint():
    response = client.get("/api/v1/telemetry/trajectory?n_events=6")
    assert response.status_code == 200
    data = response.json()
    assert len(data["events"]) == 6
    assert len(data["trajectory"]) > 0
    assert 0 <= data["denial_probability"] <= 1
    assert data["risk_label"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    first = data["trajectory"][0]
    assert {"time", "z", "friction"}.issubset(first.keys())
    assert 0 <= first["z"] <= 1
    assert 0 <= first["friction"] <= 1

def test_logs_telemetry_endpoint():
    response = client.get("/api/v1/telemetry/logs?limit=12")
    assert response.status_code == 200
    data = response.json()
    assert len(data["rows"]) == 12
    assert data["summary"]["TOTAL"] == 12
    row = data["rows"][0]
    assert {"timestamp", "cpt_opaque_id", "continuous_time", "latent_state", "friction_score", "status"}.issubset(row.keys())
    assert 0 <= row["latent_state"] <= 1
    assert 0 <= row["friction_score"] <= 1

def test_cde_formatter_edge_cases():
    # Test with identical dates to ensure _make_strictly_monotone works
    events = [
        {"claim_date": "2023-01-01", "allowed_amount": 100, "billed_amount": 200, "drg_weight": 1.0, "length_of_stay": 2, "icd_chapter": 1, "procedure_count": 1, "prior_denial_rate": 0.1, "payer_score": 0.5},
        {"claim_date": "2023-01-01", "allowed_amount": 150, "billed_amount": 250, "drg_weight": 1.2, "length_of_stay": 3, "icd_chapter": 2, "procedure_count": 2, "prior_denial_rate": 0.2, "payer_score": 0.6},
        {"claim_date": "2023-01-01", "allowed_amount": 200, "billed_amount": 300, "drg_weight": 1.5, "length_of_stay": 4, "icd_chapter": 3, "procedure_count": 3, "prior_denial_rate": 0.3, "payer_score": 0.7}
    ]
    
    coeffs, times_t, stats = coeffs_from_claim_events(events)
    # Check that times_t is strictly monotonically increasing
    times_np = times_t.cpu().numpy()
    assert np.all(np.diff(times_np) > 0)
    assert coeffs.shape[0] == 1
    assert coeffs.shape[1] == 2 # n_intervals = n - 1 = 3 - 1 = 2
    
    # Test with out-of-order dates
    events_unordered = [events[2], events[0], events[1]]
    coeffs_uo, times_t_uo, stats_uo = coeffs_from_claim_events(events_unordered)
    # The formatter should sort chronologically
    assert torch.allclose(times_t, times_t_uo)

def test_math_engine_logic():
    # Build a small dummy model to test the core logic rapidly
    model = build_liquid_cde_model(input_channels=9, hidden_dim=8, output_dim=1, trajectory_steps=10)
    model.eval()
    
    events = [
        {"claim_date": "2023-01-01", "allowed_amount": 100, "billed_amount": 200, "drg_weight": 1.0, "length_of_stay": 2, "icd_chapter": 1, "procedure_count": 1, "prior_denial_rate": 0.1, "payer_score": 0.5},
        {"claim_date": "2023-02-01", "allowed_amount": 150, "billed_amount": 250, "drg_weight": 1.2, "length_of_stay": 3, "icd_chapter": 2, "procedure_count": 2, "prior_denial_rate": 0.2, "payer_score": 0.6},
        {"claim_date": "2023-03-01", "allowed_amount": 200, "billed_amount": 300, "drg_weight": 1.5, "length_of_stay": 4, "icd_chapter": 3, "procedure_count": 3, "prior_denial_rate": 0.3, "payer_score": 0.7}
    ]
    coeffs, _, _ = coeffs_from_claim_events(events)
    
    with torch.no_grad():
        out = model(coeffs, return_trajectory=True)
    
    assert "denial_prob" in out
    assert "trajectory" in out
    assert out["trajectory"].shape == (1, 10, 8)
    
    shock_result = model.simulate_policy_shock(coeffs, shock_magnitude=0.5)
    
    # Verify the adaptation_score is stable (not inf, not nan)
    score = shock_result["adaptation_score"]
    assert not np.isnan(score)
    assert not np.isinf(score)
    assert score >= 0
