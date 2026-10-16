# TCN implementation specification

This numerical specification implements stage 3 of the [locked protocol](locked_v1/protocol.md) and [configuration](locked_v1/design.json). It is written before inspecting neural development errors. The locked study scope, model grid, and temporal cutoffs remain controlling.

## Architecture and information

`src/gbd_park/tcn.py` implements three causal Conv1d layers with kernel 3, dilations 1/2/4, ReLU, and dropout 0.1 after each activation. Explicit left padding preserves all eight temporal positions without looking forward. The final temporal position is concatenated with the 13 static features and passed directly to a five-output linear head. There are no residual blocks, normalization layers, country embeddings, output activations, or additional inputs.

The input contains eight origin-centered log lags followed by the last log level, male indicator, and eleven age indicators. Outputs are five future log changes. The 16-channel model has 1,782 parameters; the 32-channel model has 6,566. Both meet the strict fewer-than-10,000 requirement.

## Source preprocessing and objective

The primary source pool contains all six countries other than the target, excluding both target sexes. The caller constructs completed windows using the frozen calendar: the latest fifth-horizon label must be no later than the fit origin. All country/sex/age strata and eligible windows receive their hierarchical equal weights. Weights are rescaled to mean one, identically to the non-neural controls.

A weighted `StandardScaler` is fitted only to source training features; it is applied independently to each of the 21 columns. Labels remain unscaled log changes. Static features are scaled too, while constant columns retain the scaler's unit scale. Target rows never influence source preprocessing.

Training minimizes the mean of `sample_weight * mean_horizon_absolute_error` in each shuffled minibatch. Uniform shuffling and mean-one global weights retain the intended hierarchical objective; the final smaller batch is retained. The recorded epoch loss is the example-count-weighted mean of online minibatch losses, including training-mode dropout. It is an optimization diagnostic, not an evaluation score.

## Optimization and numerical controls

The eight settings are the Cartesian product of channels 16/32, AdamW weight decay 0.001/0.01, and 50/100 epochs, in that order. Use learning rate 0.001, batch size 64, gradient-norm clipping at 1, float32 tensors, and default PyTorch Conv1d/Linear initialization. AdamW uses betas `(0.9, 0.999)`, epsilon `1e-8`, `amsgrad=False`, and applies decay to all parameters, including biases. Disable fused/foreach optimizer and clipping implementations to fix the numerical path. There is no early stopping or scheduler.

Seed Python, NumPy, and PyTorch. Shuffle using a separately seeded CPU generator. Enable deterministic PyTorch algorithms and deterministic cuDNN, disable cuDNN benchmarking and TF32, and initialize `CUBLAS_WORKSPACE_CONFIG=:4096:8` before torch import. CPU operations use one thread per worker. Seed 11 governs inner tuning; the fixed seeds 11, 23, 37, 53, and 71 govern reported outer ensembles. A requested unavailable CUDA device raises an explicit error; it does not silently switch to CPU. Reproducibility is checked within the same device/software configuration; CPU and CUDA need not produce identical fitted weights.

## Frozen inference and adaptation

Training returns the model in CPU evaluation mode with gradients disabled. Target prediction and the existing two-coefficient adaptation operate after freezing both encoder and head. A SHA-256 fingerprint covers the architecture, every model tensor, and all learned scaler fields before and after prediction/calibration. Trusted local joblib checkpoints contain the frozen state and its fingerprint; loading verifies both. Never load checkpoints from an untrusted source.

The existing `adaptation.py` fits the identical regularized absolute-error correction used by the non-neural controls. Neural weights and scaler parameters cannot change during correction. Chronological selection, zero-adaptation/intercept-only ablations, ensemble assembly, persistence fallbacks, and forecast-before-score ledgers are responsibilities of the stage-3 runner. Completed inner blocks alone determine settings. Reported outer forecasts use all five seeds in log-prediction space; no seed is selected using forecast errors.

## Acceptance checks

Before the development run, verify causal temporal states, parameter counts, reproducible seeded fits, source-only scaling, target exclusion, future-row invariance, unchanged state during adaptation, checkpoint replay, and explicit unavailable-device behavior. Run a real-data CUDA smoke fit in `agpu` and record its time and device. Preserve the final 2019–2023 evaluation block for its later stage.

## Chronological selection and reporting

The locked protocol requires shared TCN settings selected by equal-sex/age horizon-five loss and separately fitted correction coefficients for each sex. It does not uniquely specify whether correction penalties must be shared. This implementation fixes the following operational choice before neural development errors are viewed: for each base setting, select the correction penalty separately for each sex using that sex's completed inner blocks; then select one shared base setting by the equal-sex mean of those two minimum losses. Average equally over eligible origins and ages within each sex. Exact penalty ties use the declared penalty-grid order; exact base ties use fewer network parameters, then the declared base-grid order. This choice does not authorize sex-specific neural architectures.

Fit all eight base settings independently with seed 11 at candidate origins 2003–2013, using all six non-target countries. The 50- and 100-epoch candidates are independent fits from their prescribed seed; no continuation or checkpoint substitution is used. Evaluate the four adaptation penalties for each candidate and sex. At outer origin `t`, only inner origins 2003 through `t−5`, including their fully completed fifth-horizon labels, can select a setting. Use the locked cold-start defaults when no completed inner origin exists.

For reported development origins 2009–2013, fit the selected shared base under all five locked seeds. Reuse its existing seed-11 checkpoint. Average the five seeds' predicted log changes for both historical target adaptation examples and current target inputs, then fit exactly one two-coefficient correction per sex at the selected penalty. Thus the ensemble retains two target coefficients per sex, not a separate correction for every seed. No seed is selected or removed on the basis of forecast accuracy.

The zero-adaptation and intercept-only ablations use that same selected five-seed base ensemble. For the intercept-only ablation, minimize the same mean absolute-error plus squared-coefficient objective with the selected penalty and `b1=0`; do not retune the ablation. Save individual seed outputs as well as ensemble outputs. If any required seed fails, retain all five seed identities in the audit and emit a labeled persistence fallback for the complete ensemble. Never average only the surviving seeds. Correction failures likewise emit a reason and persistence fallback for the affected arm and sex.

Commit candidate predictions before loading their verification values. Save each chronological selection with its eligible origins and last label year, and commit the resulting five-seed development predictions before scoring them. Development verification ends in 2018. Prepare choices for later origins 2014–2018 using completed inner evidence only, but do not fit those origins or score 2019–2023 during stage 3. The adapted TCN remains the prespecified primary method regardless of its development rank.

Stage 4 must additionally construct reported-method five-seed prequential forecasts at origins 2003–2008 for interval calibration, using cold-start defaults through 2007 and only completed tuning evidence thereafter. The stage-3 seed-11 candidate grid is not a substitute for these ensemble residual blocks. Regional interval banks, final evaluation, alternative donor pools, and other target countries remain later stages.
