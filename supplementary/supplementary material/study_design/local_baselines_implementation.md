# Stage 1 implementation specification

Prepared 29 September 2026 before inspecting baseline development scores. This implements [locked design v1.0](locked_v1/protocol.md); the locked files and inputs remain unchanged.

## Scope and chronology

Run Saudi prevalence only, both sexes, all eleven age groups. Fit the finite configuration grid at origins 2003–2013. Origins 2003–2008 supply the earliest inner forecasts; 2009–2013 are the historical family-selection stage. Save all candidate predictions before obtaining their verification values. Score through **2018 only**. No primary 2018-origin prediction, 2019–2023 performance score, transfer comparison, interval or future projection is produced at this stage.

For outer origin `t`, choose each family's window/penalty using the equal-age mean horizon-five absolute log error at origins 2003 through `t−5`. No outer forecast's own later observations select its setting. Fit settings commonly across ages within sex. Choose subsequent local champion families using only completed 2009–2013 development forecasts, applying the locked completion rule separately at origins 2014–2018. Their settings can be prepared using completed inner forecasts; their evaluation forecasts remain for stage 5.

## Numerical conventions fixed before results

- Use Python 3.9.23, NumPy 1.26.4, SciPy 1.13.1, pandas 2.3.1 and statsmodels 0.14.5 from `agpu`. Limit BLAS threads to one per worker and use at most four independent CPU workers. These statistical models do not require CUDA.
- Fit log-rate models and exponentiate point predictions. These are geometric-scale forecasts, not unbiased arithmetic means. Do not clip medically plausible values to improve results; reject nonfinite/unsafe exponential outputs and record a persistence fallback.
- Linear trends use unweighted least squares and a time coordinate centered at the fit origin.
- Damped ETS uses estimated initialization, additive trend, no seasonality, optimized parameters with L-BFGS-B and a brute-force initial smoothing-parameter search, and no bias removal. Reject an optimizer result explicitly marked unsuccessful.
- ARIMA uses the 16 configured `(p,d,q)` orders. Use a constant when `d=0`, linear time trend (drift after differencing) when `d=1`, stationary/invertible parameter constraints, state-space maximum likelihood and at most 200 optimizer iterations. Select the smallest finite training AICc among converged candidates; ties use parameter count then `(p,d,q)`. Record every rejected candidate, reason and warning. Starting-parameter warnings do not alone invalidate a converged fit. If no candidate is eligible, emit persistence for that cell.
- The age-smoothed model minimizes mean squared log error plus the configured sum-of-squared second-difference slope penalty, using time in decades. Smooth only the ten closed age bands. The 95+ intercept/slope are fitted independently. Solve the augmented least-squares system without rounding source outcomes.
- Family/setting score ties use fitted parameter count followed by the fixed configuration order. Persistence counts as zero fitted coefficients; local trend and age-smoothed trend each have two coefficients per age; ETS has five optimized mean/trend parameters per age; ARIMA counts its estimated coefficients including innovation variance. This convention affects exact ties only.

The [installed API documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) describes the ARIMA interface; the actual installed signatures/version are recorded locally because the online stable documentation can advance beyond the environment.

## Required checks and outputs

Before development, test known metric values, exact log-linear extrapolation, the open-ended age treatment, fitted-model invariance to changes in future rows and other countries, rejection of incomplete histories/forecast grids, and time-restricted tuning/family selection. Test numerical-failure fallback and ARIMA candidate rejection. Tests of current models are necessary because the stage-0 time-accessor test alone did not validate fitted forecasts.

The run writes candidate predictions, fitted parameters, the full ARIMA order audit, separate score tables, origin/horizon summaries, tuning decisions, local champion choices for later evaluation, timestamps, code/source/configuration hashes and a validation report. Warnings and failures remain visible; no cells are silently dropped. Results are historical development evidence, not final forecast accuracy or evidence that transfer learning works.

Commands from the repository root:

```bash
/home/saif/agpu_env/bin/python tests/test_local_baselines.py
/home/saif/agpu_env/bin/python scripts/run_local_baselines.py
```

The runner requires a passing validation report matching current tested code and refuses an existing run directory. A repeated full experiment must use a new `--output` path and retain the earlier outputs.
