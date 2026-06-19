# Project Aether: Data Architecture & Optimization Audit
---
## 1. High-Level Dataset Topology
* **Total Claim Events Profiled:** 2,000
* **Unique Provider Cohorts (Entities):** 198
* **Average Density (Events per Provider):** 10.10
## 2. Continuous-Time Trajectory Density Analysis
> Liquid CDE models process trajectories. Short sequences introduce boundary condition artifacts, while excessively long sequences cause gradient attenuation during ODE integration.

### Sequence Length Distribution (Events per Provider):
* **Minimum Sequence Length:** 1
* **25th Percentile:** 7.0
* **Median (50th Percentile):** 11.0
* **75th Percentile:** 13.0
* **Maximum Sequence Length:** 23

### Temporal Sparsity Dynamics (Days between consecutive claims):
* **Mean Delta:** 148.87 days
* **Median Delta:** 103.0 days
* **95th Percentile Gap:** 445.9 days
* **Zero-day Concurrent Events:** 10 (0.55%)
## 3. Supervised Class Target Dynamics

| Dimension | Approved (0.0) | Denied (1.0) | Base Ratio (0:1) |
| :--- | :--- | :--- | :--- |
| **Global Claim Distribution** | 50.4% | 49.6% | 1:0.98 |
| **Sequence Terminal Targets ($y_{last}$)** | 53.0% | 47.0% | 1:0.89 |
## 4. Neural Input Scaling & Distribution Profiles
> Large variance or scale differences between continuous-time channels destabilize the numerical solvers (`torchdiffeq`) used inside CDEs.

| Feature Column | Mean | Std Dev | Min | Max | Missing (NaN) | Skewness | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `allowed_amount` | 2954.79 | 3529.73 | 200.00 | 74850.02 | 0.00% | 7.16 | ⚡ Unscaled Range (Scaling required) |
| `billed_amount` | 7815.74 | 9840.34 | 233.42 | 174650.87 | 0.00% | 5.27 | ⚡ Unscaled Range (Scaling required) |
| `drg_weight` | 1.83 | 1.28 | 0.25 | 18.93 | 0.00% | 3.06 | ⚠️ Heavy Skew (Log transform suggested) |
| `length_of_stay` | 6.12 | 4.85 | 0.00 | 33.00 | 0.00% | 1.27 | ✅ Stable |
| `icd_chapter` | 11.69 | 4.96 | 1.00 | 21.00 | 0.00% | -0.17 | ✅ Stable |
| `procedure_count` | 3.23 | 1.75 | 1.00 | 10.00 | 0.00% | 0.72 | ✅ Stable |
| `prior_denial_rate` | 0.29 | 0.16 | 0.00 | 0.89 | 0.00% | 0.51 | ✅ Stable |
| `payer_score` | 0.63 | 0.20 | 0.17 | 1.00 | 0.00% | 0.00 | ✅ Stable |
## 5. Feature Multi-Collinearity Profile
> Highly collinear input variables introduce mathematical redundancies in the vector field function $f_\theta(h)$, causing parameter inflation.

### High Multicollinearity Risk Vector Identifications ($|r| > 0.75$):
* **Pair:** `allowed_amount` $\leftrightarrow$ `billed_amount` | Coefficient: **$r = 0.889$** (Recommendation: Prune or combine)
## 6. Strategic Architecture Actions for Best-Ever Architecture
Based on the data profile, adjust your `backend/train.py` configuration to use these pipeline configurations:
1. **Set Sequence Bounds to Dynamic Realities:** Configure your `SequenceSpec(min_events=5, max_events=15)`. This eliminates extreme outlier trajectories and matches your exact clinical density profile.
2. **De-duplicate Concurrent Integrations:** Found concurrent events ($0.0$ day deltas). Standard cubic spline interpolation will encounter undefined gradients or singular matrices if identical timestamps exist. Group same-day claims via `.groupby(['provider_id', 'claim_date']).sum()` or add a sub-second fractional jitter ($+1e-5$ seconds) to order them chronologically.
3. **Stable Losses:** Class target tracking shows normal balanced ratios. `nn.BCELoss()` remains safe.