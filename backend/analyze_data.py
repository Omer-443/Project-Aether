"""backend/analyze_data.py - Comprehensive Data Audit & Pipeline Analysis Script

This script acts as a diagnostic gatekeeper before model training. It evaluates 
the clinical event registry to profile feature distributions, multicollinearity, 
temporal sparsity, and sequence properties specifically for the LiquidCDEModel.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

from data_pipeline.cms_loader import load_cms_data
from data_pipeline.cde_formatter import FEATURE_COLS

# Attempt to import visualization libraries, gracefully fallback if missing
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIS = True
except ImportError:
    HAS_VIS = False


def run_comprehensive_audit(df: pd.DataFrame, output_dir: str = "backend/reports") -> None:
    """Performs deep profiling of the dataset across structural, statistical,
    temporal, and directional dimensions.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(output_dir) / "data_audit_report.md"
    
    # Ensure chronological order and type safety before math operations
    df = df.copy()
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df = df.sort_values(by=["provider_id", "claim_date"]).reset_index(drop=True)

    print("🚀 Starting comprehensive data architecture audit...")
    
    markdown_buffer = []
    markdown_buffer.append("# Project Aether: Data Architecture & Optimization Audit")
    markdown_buffer.append("---")

    # ---------------------------------------------------------
    # 1. High-Level Dataset Topology
    # ---------------------------------------------------------
    total_records = len(df)
    unique_providers = df["provider_id"].nunique()
    
    markdown_buffer.append("## 1. High-Level Dataset Topology")
    markdown_buffer.append(f"* **Total Claim Events Profiled:** {total_records:,}")
    markdown_buffer.append(f"* **Unique Provider Cohorts (Entities):** {unique_providers:,}")
    markdown_buffer.append(f"* **Average Density (Events per Provider):** {total_records / unique_providers:.2f}")
    
    # ---------------------------------------------------------
    # 2. Continuous-Time Trajectory Density Analysis
    # ---------------------------------------------------------
    markdown_buffer.append("## 2. Continuous-Time Trajectory Density Analysis")
    markdown_buffer.append("> Liquid CDE models process trajectories. Short sequences introduce boundary condition artifacts, while excessively long sequences cause gradient attenuation during ODE integration.")
    
    seq_counts = df.groupby("provider_id").size()
    markdown_buffer.append("\n### Sequence Length Distribution (Events per Provider):")
    markdown_buffer.append(f"* **Minimum Sequence Length:** {seq_counts.min()}")
    markdown_buffer.append(f"* **25th Percentile:** {seq_counts.quantile(0.25):.1f}")
    markdown_buffer.append(f"* **Median (50th Percentile):** {seq_counts.median():.1f}")
    markdown_buffer.append(f"* **75th Percentile:** {seq_counts.quantile(0.75):.1f}")
    markdown_buffer.append(f"* **Maximum Sequence Length:** {seq_counts.max()}")
    
    # Analyze irregular time steps (Time Deltas)
    df["time_delta_days"] = df.groupby("provider_id")["claim_date"].diff().dt.days
    deltas = df["time_delta_days"].dropna()
    
    markdown_buffer.append("\n### Temporal Sparsity Dynamics (Days between consecutive claims):")
    markdown_buffer.append(f"* **Mean Delta:** {deltas.mean():.2f} days")
    markdown_buffer.append(f"* **Median Delta:** {deltas.median():.1f} days")
    markdown_buffer.append(f"* **95th Percentile Gap:** {deltas.quantile(0.95):.1f} days")
    markdown_buffer.append(f"* **Zero-day Concurrent Events:** {(deltas == 0).sum()} ({((deltas == 0).sum() / len(deltas)) * 100:.2f}%)")

    # ---------------------------------------------------------
    # 3. Supervised Class Imbalance & Last-Event Target Drift
    # ---------------------------------------------------------
    markdown_buffer.append("## 3. Supervised Class Target Dynamics")
    
    global_imbalance = df["denied"].value_counts(normalize=True)
    # Target definition matches backend/train.py: y = denied of the LAST event in sequence
    last_event_targets = df.groupby("provider_id").last()["denied"]
    last_imbalance = last_event_targets.value_counts(normalize=True)
    
    markdown_buffer.append("\n| Dimension | Approved (0.0) | Denied (1.0) | Base Ratio (0:1) |")
    markdown_buffer.append("| :--- | :--- | :--- | :--- |")
    markdown_buffer.append(f"| **Global Claim Distribution** | {global_imbalance.get(0, 0)*100:.1f}% | {global_imbalance.get(1, 0)*100:.1f}% | 1:{global_imbalance.get(1,0)/global_imbalance.get(0,1):.2f} |")
    markdown_buffer.append(f"| **Sequence Terminal Targets ($y_{{last}}$)** | {last_imbalance.get(0, 0)*100:.1f}% | {last_imbalance.get(1, 0)*100:.1f}% | 1:{last_imbalance.get(1,0)/last_imbalance.get(0,1):.2f} |")

    # ---------------------------------------------------------
    # 4. Neural Input Scaling & Distribution Profiles
    # ---------------------------------------------------------
    markdown_buffer.append("## 4. Neural Input Scaling & Distribution Profiles")
    markdown_buffer.append("> Large variance or scale differences between continuous-time channels destabilize the numerical solvers (`torchdiffeq`) used inside CDEs.")
    
    markdown_buffer.append("\n| Feature Column | Mean | Std Dev | Min | Max | Missing (NaN) | Skewness | Status |")
    markdown_buffer.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for col in FEATURE_COLS:
        if col not in df.columns:
            markdown_buffer.append(f"| `{col}` | N/A | N/A | N/A | N/A | N/A | N/A | ❌ CRITICAL MISSING |")
            continue
            
        mean_v = df[col].mean()
        std_v = df[col].std()
        min_v = df[col].min()
        max_v = df[col].max()
        nan_pct = (df[col].isna().sum() / total_records) * 100
        skew_v = df[col].skew()
        
        # Determine health status flags
        status = "✅ Stable"
        if abs(skew_v) > 2.0:
            status = "⚠️ Heavy Skew (Log transform suggested)"
        if std_v > 1000 or max_v > 10000:
            status = "⚡ Unscaled Range (Scaling required)"
        if nan_pct > 0:
            status = f"🚨 Imbalanced ({nan_pct:.1f}% NaNs)"
            
        markdown_buffer.append(f"| `{col}` | {mean_v:.2f} | {std_v:.2f} | {min_v:.2f} | {max_v:.2f} | {nan_pct:.2f}% | {skew_v:.2f} | {status} |")

    # ---------------------------------------------------------
    # 5. Feature Multi-Collinearity (Vector Field Redundancy)
    # ---------------------------------------------------------
    markdown_buffer.append("## 5. Feature Multi-Collinearity Profile")
    markdown_buffer.append("> Highly collinear input variables introduce mathematical redundancies in the vector field function $f_\\theta(h)$, causing parameter inflation.")
    
    valid_cols = [c for c in FEATURE_COLS if c in df.columns]
    corr_matrix = df[valid_cols].corr()
    
    high_corr_pairs = []
    for i in range(len(valid_cols)):
        for j in range(i + 1, len(valid_cols)):
            c1, c2 = valid_cols[i], valid_cols[j]
            r_val = corr_matrix.loc[c1, c2]
            if abs(r_val) > 0.75:
                high_corr_pairs.append((c1, c2, r_val))
                
    if high_corr_pairs:
        markdown_buffer.append("\n### High Multicollinearity Risk Vector Identifications ($|r| > 0.75$):")
        for c1, c2, r in high_corr_pairs:
            markdown_buffer.append(f"* **Pair:** `{c1}` $\\leftrightarrow$ `{c2}` | Coefficient: **$r = {r:.3f}$** (Recommendation: Prune or combine)")
    else:
        markdown_buffer.append("\n* ✅ **No explicit high multicollinearity vectors detected.** Channels represent independent trajectories.")

    # ---------------------------------------------------------
    # 6. Strategic Pipeline Architecture Recommendations
    # ---------------------------------------------------------
    markdown_buffer.append("## 6. Strategic Architecture Actions for Best-Ever Architecture")
    
    # Generate recommendations based on the actual analysis results
    markdown_buffer.append("Based on the data profile, adjust your `backend/train.py` configuration to use these pipeline configurations:")
    
    # Rec 1: Sequence specification bounds
    min_spec = max(5, int(seq_spec_calc(seq_counts, 0.10)))
    max_spec = int(seq_spec_calc(seq_counts, 0.90))
    markdown_buffer.append(f"1. **Set Sequence Bounds to Dynamic Realities:** Configure your `SequenceSpec(min_events={min_spec}, max_events={max_spec})`. This eliminates extreme outlier trajectories and matches your exact clinical density profile.")
    
    # Rec 2: Handle duplicate concurrent events
    if (deltas == 0).sum() > 0:
        markdown_buffer.append("2. **De-duplicate Concurrent Integrations:** Found concurrent events ($0.0$ day deltas). Standard cubic spline interpolation will encounter undefined gradients or singular matrices if identical timestamps exist. Group same-day claims via `.groupby(['provider_id', 'claim_date']).sum()` or add a sub-second fractional jitter ($+1e-5$ seconds) to order them chronologically.")
        
    # Rec 3: Loss Function Adjustment
    pos_weight = last_imbalance.get(0, 1) / max(1e-5, last_imbalance.get(1, 1))
    if abs(pos_weight - 1.0) > 0.15:
        markdown_buffer.append(f"3. **Switch to Weighted Classification Objectives:** The target variable distribution shows class structural skew. Replace your standard `nn.BCELoss()` with `nn.BCEWithLogitsLoss(pos_weight=torch.tensor([{pos_weight:.3f}]))` to balance gradients against class dominance.")
    else:
        markdown_buffer.append("3. **Stable Losses:** Class target tracking shows normal balanced ratios. `nn.BCELoss()` remains safe.")

    # Save Markdown report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_buffer))
    print(f"📑 Detailed markdown data analysis audit saved to: {report_path}")

    # Generate Heatmap Plot if system has requirements installed
    if HAS_VIS:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
        plt.title("Aether Continuous-Time Feature Path Correlations")
        plt.tight_layout()
        plot_path = Path(output_dir) / "feature_correlation_matrix.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"📊 Graphical Feature Correlation Map saved to: {plot_path}")
    else:
        print("💡 Tip: Install `matplotlib` and `seaborn` in your `.venv` to auto-generate graphical correlation matrices next run.")


def seq_spec_calc(counts: pd.Series, quantile: float) -> float:
    """Helper to safely calculate quantiles or fallback to default thresholds."""
    if len(counts) == 0:
        return 5.0
    return counts.quantile(quantile)


if __name__ == "__main__":
    # Load dataset through matching runtime entry point pipeline
    # Simulates configuration used inside training script
    try:
        raw_df = load_cms_data(real_csv=None, n_rows=2000, seed=42)
        run_comprehensive_audit(raw_df)
        print("\n✅ Data architecture analysis complete. Review report details to refine training execution hyper-parameters.")
    except Exception as e:
        print(f"\n❌ Pipeline Diagnostic Execution Error: {str(e)}")