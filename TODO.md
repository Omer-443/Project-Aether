# Project Aether - Shock Graph Audit / Fix TODO

## Completed (audit)
- Reviewed backend shock simulation and model forward pass.
- Reviewed frontend visualizer (trajectory) and shock-test page rendering.
- Verified CDE coefficient tensor layout from `data_pipeline/cde_formatter.py`.

## Remaining (to implement fixes)
1. **Fix frontend overlap** in `frontend/app/shock-test/page.tsx`
   - Remove `shocked[index] ?? point` fallback which can force the shocked line to equal baseline.
   - Align by time/index safely (e.g., only compute `liquid` when `shocked[index]` exists).
2. **Fix frontend “same graph for all shock types” masking**
   - Plot at least `dim_0` and `dim_1` (or remove sigmoid compression / show raw values) so shock-induced latent changes are visible.
3. **Fix backend perturbation correctness** in `backend/app/core/math_engine.py`
   - Update `simulate_policy_shock()` to scale the correct Hermite coefficient blocks.
   - Add debug logging to confirm coefficient deltas differ across `step/impulse/ramp`.
4. **Add verification**
   - Ensure `baseline_trajectory` and `shocked_trajectory` differ numerically per shock type before returning.
   - If they differ in backend, ensure frontend plotting consumes the differing arrays.

## Test
- Manually run shock-test page for Step/Impulse/Ramp with same magnitude.
- Confirm: (a) baseline vs shocked diverge, (b) different shock types produce visibly different curves.

