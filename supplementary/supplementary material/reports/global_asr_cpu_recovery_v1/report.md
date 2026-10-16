# Separate global ASR benchmark — CPU execution recovery

This is the fixed-design CPU recovery of the original CUDA-failed execution. The original run retained 110 failed and ten successful source fits; every five-seed ensemble failed, so all 720 original TCN cells were persistence fallbacks. Those original files remain preserved and cannot establish a neural accuracy comparison.

Both donor scopes were rerun on CPU with the same countries, origin, settings and all five seeds. All 120 source fits and all 2,640 forecasts succeeded, including 720 TCN forecasts. The independent audit replayed all 216 neural/non-neural checkpoints. All forecast ledgers preceded final-period scoring. No predictive setting was changed during recovery.

## Saudi five-year endpoint

| Outcome | Sex | Family | Scope | H5 absolute log error |
|---|---|---|---|---:|
| incidence | Female | donor_boosting_adapted | global | 0.027377 |
| incidence | Female | donor_boosting_unadapted | global | 0.012992 |
| incidence | Female | donor_ridge_adapted | global | 0.018552 |
| incidence | Female | donor_ridge_unadapted | global | 0.002221 |
| incidence | Female | pooled_boosting | global | 0.013253 |
| incidence | Female | pooled_ridge | global | 0.001729 |
| incidence | Female | tcn_adapted | global | 0.025818 |
| incidence | Female | tcn_intercept | global | 0.019856 |
| incidence | Female | tcn_unadapted | global | 0.015433 |
| incidence | Female | arima | local | 0.033824 |
| incidence | Female | damped_ets | local | 0.023595 |
| incidence | Female | log_trend | local | 0.027335 |
| incidence | Female | persistence | local | 0.040063 |
| incidence | Female | donor_boosting_adapted | regional | 0.006361 |
| incidence | Female | donor_boosting_unadapted | regional | 0.007964 |
| incidence | Female | donor_ridge_adapted | regional | 0.024988 |
| incidence | Female | donor_ridge_unadapted | regional | 0.015518 |
| incidence | Female | pooled_boosting | regional | 0.018064 |
| incidence | Female | pooled_ridge | regional | 0.020763 |
| incidence | Female | tcn_adapted | regional | 0.011018 |
| incidence | Female | tcn_intercept | regional | 0.008736 |
| incidence | Female | tcn_unadapted | regional | 0.001849 |
| incidence | Male | donor_boosting_adapted | global | 0.001938 |
| incidence | Male | donor_boosting_unadapted | global | 0.006792 |
| incidence | Male | donor_ridge_adapted | global | 0.000264 |
| incidence | Male | donor_ridge_unadapted | global | 0.018858 |
| incidence | Male | pooled_boosting | global | 0.006711 |
| incidence | Male | pooled_ridge | global | 0.018170 |
| incidence | Male | tcn_adapted | global | 0.002296 |
| incidence | Male | tcn_intercept | global | 0.001516 |
| incidence | Male | tcn_unadapted | global | 0.004911 |
| incidence | Male | arima | local | 0.002809 |
| incidence | Male | damped_ets | local | 0.006432 |
| incidence | Male | log_trend | local | 0.002350 |
| incidence | Male | persistence | local | 0.048747 |
| incidence | Male | donor_boosting_adapted | regional | 0.024000 |
| incidence | Male | donor_boosting_unadapted | regional | 0.023408 |
| incidence | Male | donor_ridge_adapted | regional | 0.005991 |
| incidence | Male | donor_ridge_unadapted | regional | 0.012569 |
| incidence | Male | pooled_boosting | regional | 0.022905 |
| incidence | Male | pooled_ridge | regional | 0.006500 |
| incidence | Male | tcn_adapted | regional | 0.019884 |
| incidence | Male | tcn_intercept | regional | 0.019018 |
| incidence | Male | tcn_unadapted | regional | 0.031086 |
| prevalence | Female | donor_boosting_adapted | global | 0.024320 |
| prevalence | Female | donor_boosting_unadapted | global | 0.011726 |
| prevalence | Female | donor_ridge_adapted | global | 0.020486 |
| prevalence | Female | donor_ridge_unadapted | global | 0.003267 |
| prevalence | Female | pooled_boosting | global | 0.011894 |
| prevalence | Female | pooled_ridge | global | 0.002884 |
| prevalence | Female | tcn_adapted | global | 0.021150 |
| prevalence | Female | tcn_intercept | global | 0.015674 |
| prevalence | Female | tcn_unadapted | global | 0.012772 |
| prevalence | Female | arima | local | 0.029294 |
| prevalence | Female | damped_ets | local | 0.030008 |
| prevalence | Female | log_trend | local | 0.021938 |
| prevalence | Female | persistence | local | 0.056534 |
| prevalence | Female | donor_boosting_adapted | regional | 0.014439 |
| prevalence | Female | donor_boosting_unadapted | regional | 0.005170 |
| prevalence | Female | donor_ridge_adapted | regional | 0.018816 |
| prevalence | Female | donor_ridge_unadapted | regional | 0.006966 |
| prevalence | Female | pooled_boosting | regional | 0.017248 |
| prevalence | Female | pooled_ridge | regional | 0.002687 |
| prevalence | Female | tcn_adapted | regional | 0.012830 |
| prevalence | Female | tcn_intercept | regional | 0.001759 |
| prevalence | Female | tcn_unadapted | regional | 0.007913 |
| prevalence | Male | donor_boosting_adapted | global | 0.004283 |
| prevalence | Male | donor_boosting_unadapted | global | 0.015439 |
| prevalence | Male | donor_ridge_adapted | global | 0.003526 |
| prevalence | Male | donor_ridge_unadapted | global | 0.024510 |
| prevalence | Male | pooled_boosting | global | 0.014875 |
| prevalence | Male | pooled_ridge | global | 0.024030 |
| prevalence | Male | tcn_adapted | global | 0.004002 |
| prevalence | Male | tcn_intercept | global | 0.006067 |
| prevalence | Male | tcn_unadapted | global | 0.008263 |
| prevalence | Male | arima | local | 0.018287 |
| prevalence | Male | damped_ets | local | 0.004914 |
| prevalence | Male | log_trend | local | 0.003118 |
| prevalence | Male | persistence | local | 0.071859 |
| prevalence | Male | donor_boosting_adapted | regional | 0.010607 |
| prevalence | Male | donor_boosting_unadapted | regional | 0.024919 |
| prevalence | Male | donor_ridge_adapted | regional | 0.014406 |
| prevalence | Male | donor_ridge_unadapted | regional | 0.042647 |
| prevalence | Male | pooled_boosting | regional | 0.024811 |
| prevalence | Male | pooled_ridge | regional | 0.031181 |
| prevalence | Male | tcn_adapted | regional | 0.019451 |
| prevalence | Male | tcn_intercept | regional | 0.024842 |
| prevalence | Male | tcn_unadapted | regional | 0.040265 |

## Global versus regional adapted TCN

Negative differences favor global donors. This is a separate full-age ASR supporting comparison, with no replacement of the regional primary verdict.

| Target | Outcome | Sex | Global minus regional log error | Relative improvement |
|---|---|---|---:|---:|
| Bahrain | incidence | Female | 0.004688 | -52.1% |
| Bahrain | incidence | Male | 0.002361 | -12.6% |
| Bahrain | prevalence | Female | -0.000327 | 4.3% |
| Bahrain | prevalence | Male | 0.006760 | -30.5% |
| Kuwait | incidence | Female | 0.011541 | -335.0% |
| Kuwait | incidence | Male | 0.000679 | -4.3% |
| Kuwait | prevalence | Female | -0.007031 | 72.5% |
| Kuwait | prevalence | Male | 0.007485 | -510.1% |
| Oman | incidence | Female | -0.007041 | 69.5% |
| Oman | incidence | Male | -0.001726 | 30.6% |
| Oman | prevalence | Female | 0.002479 | -2289.0% |
| Oman | prevalence | Male | 0.002964 | -14.9% |
| Qatar | incidence | Female | 0.017752 | -983.1% |
| Qatar | incidence | Male | 0.022019 | -91.3% |
| Qatar | prevalence | Female | 0.016218 | -282.5% |
| Qatar | prevalence | Male | 0.010579 | -37.7% |
| Saudi Arabia | incidence | Female | 0.014800 | -134.3% |
| Saudi Arabia | incidence | Male | -0.017588 | 88.5% |
| Saudi Arabia | prevalence | Female | 0.008320 | -64.8% |
| Saudi Arabia | prevalence | Male | -0.015449 | 79.4% |
| United Arab Emirates | incidence | Female | -0.011613 | 53.0% |
| United Arab Emirates | incidence | Male | 0.031286 | -178.8% |
| United Arab Emirates | prevalence | Female | -0.010430 | 97.2% |
| United Arab Emirates | prevalence | Male | 0.011834 | -36.2% |

## Interpretation limits

The source contains modeled male/female prevalence and incidence ASRs for 203 supplied countries/territories. Workbook agreement and UN geographic-label matching do not independently establish native GBD provenance, exact location boundaries or standard weights. The retrospective 2018-origin endpoint was previously inspected. No interval-calibration, count-conversion, causal biological or independent-country inference follows from this point-only benchmark. The original failed execution, recovery rationale, successful CPU ledger and independent audit are all retained.
