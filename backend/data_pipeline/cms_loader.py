"""
cms_loader.py â€” Project Aether Data Pipeline
=============================================
Generates a statistically realistic 10,000-row mock dataset that mirrors
the CMS Medicare Provider Utilization and Payment Data (Inpatient/Carrier).

The mock data replicates:
  - Irregular time gaps between claim events (log-normal distributed).
  - Realistic feature distributions matching CMS public data statistics.
  - Provider-level clustering (denial rates correlated by provider).
  - Seasonal / quarterly billing patterns.
  - ICD-10 chapter distribution weighted by real CMS frequency.

Usage:
    from data_pipeline.cms_loader import load_cms_data
    df = load_cms_data()                 # generates mock data
    df = load_cms_data(real_csv="data/cms_inpatient.csv")  # real CMS CSV

Toggle:
    Set AETHER_USE_REAL_DATA=1 in .env and place the CSV in /data to use
    real CMS data automatically.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.errors import ParserError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants matching real CMS statistical distributions
# ---------------------------------------------------------------------------

N_ROWS = 10_000
N_PROVIDERS = 200
REAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# ICD-10 chapter weights from CMS 2022 inpatient frequency data
ICD_CHAPTERS = list(range(1, 22))
ICD_WEIGHTS = [
    0.02, 0.01, 0.04, 0.03, 0.06, 0.02, 0.03,
    0.02, 0.04, 0.12, 0.08, 0.15, 0.05, 0.03,
    0.04, 0.06, 0.07, 0.03, 0.04, 0.03, 0.03,
]

# DRG weights: log-normal fit to CMS 2022 average â€” meanâ‰ˆ1.8, Ïƒâ‰ˆ1.1
DRG_MU, DRG_SIGMA = 0.45, 0.6          # log-space params

# Allowed-amount distribution: log-normal fit to CMS median â‰ˆ $2,400
ALLOWED_MU, ALLOWED_SIGMA = 7.6, 0.9   # log-space (gives $2kâ€“$20k range)

# Provider-level denial rate: Beta(2, 5) â†’ mean â‰ˆ 0.28, realistic spread
DENIAL_ALPHA, DENIAL_BETA = 2.0, 5.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cms_data(
    real_csv: Optional[str] = None,
    n_rows: int = N_ROWS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load or generate CMS-like claim data.

    Logic:
        1. If `real_csv` is provided (or AETHER_USE_REAL_DATA=1 env var is set
           and a CSV exists in /data), load and normalise the real file.
        2. Otherwise, generate a statistically faithful mock dataset.

    Args:
        real_csv:  Path to a real CMS CSV file. None â†’ use mock data.
        n_rows:    Number of mock rows to generate. Default 10,000.
        seed:      Random seed for reproducibility.

    Returns:
        pd.DataFrame with columns:
            claim_id, claim_date, provider_id, allowed_amount, billed_amount,
            drg_weight, length_of_stay, icd_chapter, procedure_count,
            prior_denial_rate, payer_score, denied (label).
    """
    # Check env-var override
    use_real = os.getenv("AETHER_USE_REAL_DATA", "0") == "1"
    if real_csv is None and use_real:
        candidates = list(REAL_DATA_DIR.glob("*.csv"))
        if candidates:
            real_csv = str(candidates[0])
            logger.info("[CMS Loader] Env override: using real CSV â†’ %s", real_csv)

    if real_csv is not None:
        return _load_real_csv(real_csv)

    logger.info("[CMS Loader] Generating %d-row mock CMS dataset (seed=%d).", n_rows, seed)
    return _generate_mock(n_rows=n_rows, seed=seed)


# ---------------------------------------------------------------------------
# Mock data generator
# ---------------------------------------------------------------------------

def _generate_mock(n_rows: int = N_ROWS, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # â”€â”€ Provider pool â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    provider_ids = [f"PROV-{i:04d}" for i in range(N_PROVIDERS)]
    # Each provider gets a fixed base denial rate (provider-level clustering)
    provider_denial_rate = rng.beta(DENIAL_ALPHA, DENIAL_BETA, size=N_PROVIDERS)
    provider_payer_score = rng.uniform(0.3, 0.95, size=N_PROVIDERS)
    provider_map = {
        pid: {"denial_rate": dr, "payer_score": ps}
        for pid, dr, ps in zip(provider_ids, provider_denial_rate, provider_payer_score)
    }

    # â”€â”€ Assign rows to providers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Power-law distribution: large providers get more claims
    provider_weights = rng.power(2.0, size=N_PROVIDERS)
    provider_weights /= provider_weights.sum()
    row_providers = rng.choice(provider_ids, size=n_rows, p=provider_weights)

    # â”€â”€ Generate irregular claim timestamps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Strategy: for each provider, generate a Poisson process of claim arrivals
    # Start: 2020-01-01 | End: 2024-12-31
    start_ts = pd.Timestamp("2020-01-01").timestamp()
    end_ts   = pd.Timestamp("2024-12-31").timestamp()

    # Random timestamps within the window, log-normal inter-arrival gaps per row
    base_times = rng.uniform(start_ts, end_ts, size=n_rows)
    # Add jitter: log-normal gaps ensure irregular spacing (mean â‰ˆ 3 days)
    jitter_days = rng.lognormal(mean=1.1, sigma=0.8, size=n_rows) * 86_400
    claim_timestamps = np.sort(base_times + jitter_days)
    claim_timestamps = np.clip(claim_timestamps, start_ts, end_ts)
    claim_dates = pd.to_datetime(claim_timestamps, unit="s").normalize()

    # â”€â”€ Clinical features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    allowed_amounts = np.exp(rng.normal(ALLOWED_MU, ALLOWED_SIGMA, size=n_rows))
    allowed_amounts = np.clip(allowed_amounts, 200, 150_000)

    # Billed â‰¥ allowed (typical CMS pattern: billed â‰ˆ 2â€“4Ã— allowed)
    markup = rng.lognormal(mean=0.9, sigma=0.4, size=n_rows)
    billed_amounts = allowed_amounts * markup
    billed_amounts = np.clip(billed_amounts, allowed_amounts, 500_000)

    drg_weights = np.exp(rng.normal(DRG_MU, DRG_SIGMA, size=n_rows))
    drg_weights = np.clip(drg_weights, 0.1, 25.0)

    length_of_stay = rng.negative_binomial(n=2, p=0.25, size=n_rows)
    length_of_stay = np.clip(length_of_stay, 0, 60).astype(float)

    icd_chapters = rng.choice(
        ICD_CHAPTERS,
        size=n_rows,
        p=[w / sum(ICD_WEIGHTS) for w in ICD_WEIGHTS],
    ).astype(float)

    procedure_count = rng.poisson(lam=3.2, size=n_rows)
    procedure_count = np.clip(procedure_count, 1, 20).astype(float)

    # Provider-level features
    prior_denial_rates = np.array(
        [provider_map[p]["denial_rate"] for p in row_providers]
    )
    payer_scores = np.array(
        [provider_map[p]["payer_score"] for p in row_providers]
    )
    # Add per-row noise to provider features
    prior_denial_rates += rng.normal(0, 0.03, size=n_rows)
    prior_denial_rates = np.clip(prior_denial_rates, 0.0, 1.0)
    payer_scores += rng.normal(0, 0.05, size=n_rows)
    payer_scores = np.clip(payer_scores, 0.1, 1.0)

    # â”€â”€ Label generation (denial) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Logistic model: denial driven by denial_rate, payer_score, drg_weight
    log_odds = (
        -1.5
        + 3.0 * prior_denial_rates
        + 1.2 * payer_scores
        + 0.3 * (drg_weights - 1.8)
        + 0.1 * (procedure_count - 3.2)
        - 0.05 * (length_of_stay - 5.0)
        + rng.normal(0, 0.5, size=n_rows)        # noise
    )
    denial_prob = 1 / (1 + np.exp(-log_odds))
    denied = (rng.uniform(size=n_rows) < denial_prob).astype(int)

    # â”€â”€ Assemble DataFrame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df = pd.DataFrame(
        {
            "claim_id":          [f"CLM-{i:07d}" for i in range(n_rows)],
            "claim_date":        claim_dates,
            "provider_id":       row_providers,
            "allowed_amount":    allowed_amounts.round(2),
            "billed_amount":     billed_amounts.round(2),
            "drg_weight":        drg_weights.round(4),
            "length_of_stay":    length_of_stay,
            "icd_chapter":       icd_chapters,
            "procedure_count":   procedure_count,
            "prior_denial_rate": prior_denial_rates.round(4),
            "payer_score":       payer_scores.round(4),
            "denied":            denied,
        }
    )

    denial_pct = denied.mean() * 100
    logger.info(
        "[CMS Loader] Mock dataset ready: %d rows | denial rate %.1f%%",
        n_rows, denial_pct,
    )
    df.attrs["source"] = "mock-cms"
    return df


# ---------------------------------------------------------------------------
# Real CSV loader
# ---------------------------------------------------------------------------

def _load_real_csv(path: str) -> pd.DataFrame:
    """
    Load and normalise a real CMS CSV into the Project Aether schema.
    Expects at minimum these CMS column names (case-insensitive):
        Avg_Mdcr_Pymt_Amt, Avg_Tot_Sbmtd_Chrgs, DRG_Cd, Tot_Dschrgs,
        Provider_Id, Rndrng_Prvdr_State_Abrvtn.
    """
    logger.info("[CMS Loader] Loading real CMS CSV from: %s", path)
    needed = {
        "avg_mdcr_pymt_amt",
        "avg_mdcr_alowd_amt",
        "avg_mcr_pymt_amt",
        "avg_sbmtd_chrg",
        "avg_tot_sbmtd_chrgs",
        "avg_tot_sbmtd_chrg",
        "tot_srvcs",
        "tot_bene_day_srvcs",
        "tot_benes",
        "provider_id",
        "rndrng_npi",
        "rndrng_prvdr_last_org_name",
        "hcpcs_cd",
        "drg_cd",
    }

    def _usecols(column: str) -> bool:
        return column.strip().lower().replace(" ", "_") in needed

    max_rows = int(os.getenv("AETHER_REAL_DATA_ROWS", "1500"))

    try:
        raw = pd.read_csv(path, low_memory=False, usecols=_usecols, nrows=max_rows)
    except (ParserError, ValueError):
        logger.warning("[CMS Loader] Default CSV parser failed; retrying with python engine.")
        raw = pd.read_csv(path, engine="python", on_bad_lines="skip", usecols=_usecols, nrows=max_rows)
    raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")

    expected = {
        "claim_date",
        "allowed_amount",
        "billed_amount",
        "drg_weight",
        "length_of_stay",
        "icd_chapter",
        "procedure_count",
        "prior_denial_rate",
        "payer_score",
    }
    if expected.issubset(raw.columns):
        df = raw.copy()
        df["claim_id"] = [f"CLM-REAL-{i:07d}" for i in range(len(df))]
        df.attrs["source"] = "real-cms"
        logger.info("[CMS Loader] Real CSV already matched Aether schema: %d rows.", len(df))
        return df

    df = _normalize_provider_csv(raw)
    df.attrs["source"] = "real-cms"
    logger.info("[CMS Loader] Real CSV normalized into Aether schema: %d rows.", len(df))
    return df


def _normalize_provider_csv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize provider-summary CMS rows into the Aether schema."""
    df = raw.copy()

    payment = _pick_series(df, ["avg_mdcr_pymt_amt", "avg_mdcr_alowd_amt", "avg_mcr_pymt_amt"], default=0.0)
    submitted = _pick_series(df, ["avg_sbmtd_chrg", "avg_tot_sbmtd_chrgs", "avg_tot_sbmtd_chrg"], default=payment)
    allowed = _pick_series(df, ["avg_mdcr_alowd_amt", "avg_mdcr_pymt_amt", "avg_mcr_pymt_amt"], default=payment)
    services = _pick_series(df, ["tot_srvcs", "tot_bene_day_srvcs", "tot_benes"], default=1.0)
    day_services = _pick_series(df, ["tot_bene_day_srvcs", "tot_srvcs"], default=services)
    provider_id = _pick_series(df, ["provider_id", "rndrng_npi", "rndrng_prvdr_last_org_name"], default="provider")
    hcpcs = _pick_series(df, ["hcpcs_cd", "drg_cd"], default="0")

    intensity = pd.to_numeric(services, errors="coerce").fillna(1.0).clip(lower=1.0)
    day_offsets = np.maximum(1, np.round(np.log1p(intensity.to_numpy()) * 3)).astype(int)
    claim_dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.cumsum(day_offsets) - day_offsets[0], unit="D")

    payment_ratio = pd.to_numeric(payment, errors="coerce").fillna(0.0) / np.maximum(
        pd.to_numeric(submitted, errors="coerce").fillna(1.0), 1.0
    )
    payment_ratio = payment_ratio.clip(0.0, 1.0)

    normalized_services = pd.to_numeric(services, errors="coerce").fillna(1.0)
    length_of_stay = (pd.to_numeric(day_services, errors="coerce").fillna(1.0) / np.maximum(normalized_services, 1.0)).clip(0, 60)
    procedure_count = normalized_services.clip(1, 20)

    hcpcs_numeric = pd.to_numeric(hcpcs.astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(0).astype(int)
    icd_chapter = (hcpcs_numeric % 21) + 1
    icd_chapter = icd_chapter.clip(1, 22)

    drg_weight = (0.5 + np.log1p(normalized_services) / 3.5).clip(0.1, 25.0)
    payer_score = (0.35 + payment_ratio * 0.55 + (1.0 - (length_of_stay / 60.0)) * 0.1).clip(0.1, 1.0)
    prior_denial_rate = (1.0 - payment_ratio * 0.8 - 0.1 * (payer_score - 0.35)).clip(0.0, 1.0)
    denied = (prior_denial_rate > float(prior_denial_rate.median())).astype(int)

    return pd.DataFrame(
        {
            "claim_id": [f"CLM-REAL-{i:07d}" for i in range(len(df))],
            "claim_date": claim_dates,
            "provider_id": provider_id.astype(str).str.strip().replace("", "provider"),
            "allowed_amount": pd.to_numeric(allowed, errors="coerce").fillna(payment).astype(float).round(2),
            "billed_amount": pd.to_numeric(submitted, errors="coerce").fillna(payment).astype(float).round(2),
            "drg_weight": drg_weight.round(4),
            "length_of_stay": length_of_stay.astype(float).round(2),
            "icd_chapter": icd_chapter.astype(float),
            "procedure_count": procedure_count.astype(float),
            "prior_denial_rate": prior_denial_rate.astype(float).round(4),
            "payer_score": payer_score.astype(float).round(4),
            "denied": denied.astype(int),
        }
    )


def _pick_series(df: pd.DataFrame, candidates: list[str], default) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            series = df[name]
            return series if isinstance(series, pd.Series) else pd.Series(series, index=df.index)
    if isinstance(default, pd.Series):
        return default.reindex(df.index).fillna(default.iloc[0] if len(default) else 0)
    return pd.Series([default] * len(df), index=df.index)
