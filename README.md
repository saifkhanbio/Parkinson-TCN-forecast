# Parkinson-TCN-forecast

Research scripts for **Sex-Specific Transfer Learning and Forecast Reliability for Parkinson’s Disease in Saudi Arabia: A GBD 2023 Study with Gulf Benchmarking**.

The study evaluates whether regional pretraining with limited Saudi adaptation improves five-year forecasts of male and female GBD-estimated Parkinson’s prevalence. Saudi Arabia is the primary target; Bahrain, Kuwait, Oman, Qatar and the United Arab Emirates undergo the same evaluation. Jordan contributes regional donor data.

**Repository contents:** Python source and this README. Datasets, study configuration files, lock manifests, validation records, fitted checkpoints, results and generated reports are excluded. Full study execution requires those additional local artifacts; cloning this repository alone does not reproduce the analyses.

## Study scope

- **Primary outcome:** age-specific prevalence at ages 45+, across eleven bands from 45–49 through 95+, reported separately by sex. The primary five-year endpoint is 2018→2023.
- **Key secondary outcome:** incidence. Supporting analyses include mortality, disability, target-history budgets and a separate full-age age-standardized-rate (ASR) benchmark.
- **Comparators:** local statistical forecasts, pooled and donor-only ridge/boosting models, and compact temporal convolutional networks (TCNs) with limited target adaptation.
- **Reliability:** chronological evaluation, donor/adaptation harm, interval coverage, width, weighted interval score, and coherent counts, age shares and sex ratios. Ages 80+ remain explicit.
- **Demography:** population-source sensitivity, native-count reconciliation and conditional projections from the 2023 disease-data cutoff.

Age-specific errors averaged across the eleven bands are distinct from full-age ASRs. Survival and individual patient progression are outside the study scope.

## Repository layout

| Directory | Contents |
|---|---|
| [`src/gbd_park/`](src/gbd_park/) | Forecasting models, adaptation, donor selection, scoring, intervals, demography and distribution pooling. |
| [`scripts/`](scripts/) | Stage runners, benchmarks, source verification, reporting and finalization scripts. |
| [`tests/`](tests/) | `unittest` checks for chronology, leakage, numerical calculations and pipeline behavior. |
| [`study_design/locked_v1/`](study_design/locked_v1/) | Design-input builder; its required JSON specification is excluded. |
| [`study_design/literature_review/`](study_design/literature_review/) | Literature retrieval scripts. Some reference machine-specific plugin paths. |
| [`supporting_data/`](supporting_data/) | Dated source acquisition, preparation and metadata scripts. |
| [`work/`](work/) | Independent audit, diagnostic and validation scripts. |

## Environment

Run commands from the repository root. The recorded primary run used Python **3.9.23**, NumPy **1.26.4**, pandas **2.3.1**, SciPy **1.13.1**, scikit-learn **1.6.1**, statsmodels **0.14.5**, joblib **1.5.1**, and PyTorch **2.8.0+cu128**. These are recorded versions, not a dependency lockfile.

```bash
git clone https://github.com/saifkhanbio/Parkinson-TCN-forecast.git
cd Parkinson-TCN-forecast
python3 -m venv .venv
source .venv/bin/activate
```

Install the listed numerical/modeling packages in the environment. Reporting and acquisition scripts additionally use `matplotlib`, `lxml`, `openpyxl`, `psutil`, `requests`, and, in some scripts, `pip._vendor.requests`. PDF extraction scripts require `pdftotext`. Select a PyTorch installation appropriate to your CPU or CUDA environment; GPU support is optional for runners that expose CPU execution. Linux/WSL matches the original workflow; some supervision scripts use `fcntl`.

Machine-specific paths, including `/home/saif/agpu_env`, must be reviewed before reuse. There is no application build or package installation step for this source tree; runners set their local import paths.

## Required local inputs

The complete research workspace supplies:

1. Native GBD age-specific exports under `More data/`, separate ASR inputs under `age_standard/`, and demographic/source downloads under the dated `supporting_data/` directories.
2. `study_design/locked_v1/design.json`, its protocol and lock manifest, plus specifications for later analyses.
3. Prepared tables under `data/processed/design_v1/`, including `regional_outcomes.csv`.
4. Stage-specific test/preflight records under `work/`, implementation documents, and the completed prior-stage outputs required by each runner.

Obtain data through the relevant providers, including [GBD Results](https://vizhub.healthdata.org/gbd-results/), [UN World Population Prospects](https://population.un.org/wpp/), GASTAT and GCC-Stat, observing their access and reuse terms. The scripts preserve release identifiers, units and source hashes. Raw downloads should remain unchanged.

The design builder **reads** its JSON specification; it does not recreate the missing configuration or the entire execution record. Passing unit tests alone does not create the preflight records required by production runners. Do not fabricate manifests or disable hash checks to bypass missing prerequisites.

## Analysis workflow

The following commands show the core stage order **after the required inputs and stage-specific validation records have been supplied**. Later commands depend on successfully completed earlier stages.

```bash
python supporting_data/2026-09-26/prepare_data.py
python study_design/locked_v1/build_design.py
python scripts/run_local_baselines.py --workers 4
python scripts/run_nonneural.py --workers 4
python scripts/run_tcn.py --device cpu --workers 4
python scripts/run_intervals.py --workers 4
python scripts/run_primary.py --workers 4
python scripts/run_secondary.py --device cpu --workers 12 --neural-workers 4
```

Use `--device cuda:0` where supported and appropriately benchmarked. Worker counts are examples, not hardware requirements. Most runners refuse to overwrite completed output directories. Where `--output` is supported, changing it may require corresponding downstream path changes.

Additional entry points include:

| Purpose | Script |
|---|---|
| Population assumptions and derived burden | [`run_population_sensitivity.py`](scripts/run_population_sensitivity.py), [`run_demography.py`](scripts/run_demography.py) |
| Native-count reconciliation | [`run_count_coherence.py`](scripts/run_count_coherence.py) |
| Donor and target-history comparisons | [`run_donor_comparisons.py`](scripts/run_donor_comparisons.py), [`run_learning_curves.py`](scripts/run_learning_curves.py) |
| Supporting mortality/disability | [`run_supporting.py`](scripts/run_supporting.py), [`run_disability_components.py`](scripts/run_disability_components.py) |
| Successful CPU global-ASR execution recovery | [`run_global_asr_cpu_recovery.py`](scripts/run_global_asr_cpu_recovery.py) |
| Exploratory distribution mixture | [`run_distribution_mixture.py`](scripts/run_distribution_mixture.py) |
| Native ASR and hierarchy verification | [`verify_native_asr_export.py`](scripts/verify_native_asr_export.py), [`verify_downloaded_gbd_hierarchy.py`](scripts/verify_downloaded_gbd_hierarchy.py) |
| Final integrated synthesis | [`report_study_synthesis_verified.py`](scripts/report_study_synthesis_verified.py) |

The mixture pools saved complete trajectories; it does not train new models. Report and recovery scripts also require their documented prior artifacts. Inspect each script’s arguments and input checks before execution; some load configuration before argument parsing.

## Validation

A source-only checkout supports Python syntax compilation:

```bash
python -m compileall -q src scripts tests study_design supporting_data work
```

With the required configuration and fixtures restored, run the relevant test suites, for example:

```bash
python -m unittest discover -s tests -p 'test_local_baselines.py' -v
python -m unittest discover -s tests -p 'test_distribution_mixture.py' -v
```

Many tests read excluded configuration or prior artifacts, so these commands are conditional on the complete workspace. Run without `-O`, which disables assertions. Retain fit failures, source/code hashes, chronological selection records and independent replay audits; syntax checks alone do not establish scientific validity.

## Interpretation

The study concerns retrospectively modeled aggregate GBD outcomes. Its joint primary superiority criterion was not met: transfer benefits vary by sex and comparator. Later calibration and mixture experiments remain exploratory and do not establish adequate interval reliability.

Source estimation bounds are distinct from forecast intervals. Marginal GBD/UN bounds do not identify joint uncertainty, ASRs must not be multiplied by population to obtain counts, and population scenarios are not probability intervals. Overlapping origins, age cells, shared-donor countries and neural seeds are not independent epidemiological samples. Sex/age patterns do not establish biological mechanisms, clinical utility or validated cross-release forecast stability.
