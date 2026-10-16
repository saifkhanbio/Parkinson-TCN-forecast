# Prespecified donor comparisons: implementation

This secondary analysis implements the donor comparisons already specified in the locked protocol. The completed CPU Saudi prevalence primary experiment remains immutable. All donor arms, including a newly fitted all-six-country reference, use CUDA on the same machine and numerical environment. Differences from the CPU primary are reported as a numerical-device sensitivity; the GPU reference does not replace that primary result.

## Arms and historical reconstruction

Analyze eight arms: all six other regional countries; the other five GCC countries; three historically similar countries; and five random three-country pools under seeds 101, 211, 307, 401, and 503. Similarity follows [the frozen donor specification](donor_implementation.md). Recompute donor selection at each fit origin, using only its available history and requested target sex. Every selected source country supplies both sexes. Pair male and female forecasts from their respective source pools; a similarity arm may therefore use different male/female source models.

For every arm, fit the original eight architecture/weight-decay/epoch settings with tuning seed 11 at origins 2003–2013. Choose a shared architecture for both target sexes using completed inner origins `2003..t−5`, with a separate adaptation penalty per sex and equal-sex/equal-age horizon-five loss. Use the original cold defaults before the first completed inner block. Do not choose a random donor seed or pool using final performance.

Reported historical forecasts at origins 2003–2013 and reliability forecasts at 2014–2018 use all five model seeds 11, 23, 37, 53, and 71. Average log changes before fitting one correction per sex. Retain unadapted, intercept-only, and two-parameter arms from exactly the same source ensemble. A failed seed causes persistence fallback for its complete target-sex ensemble, without deleting the failed seed or cells. The 2018 fit appears once.

## Pool caching and execution

Fit identity includes origin, canonical donor membership, architecture settings, model seed, and device. Reuse a source model across arms or target sexes only when every identity field matches. Preserve the original configured training-row order within each pool. Use the benchmark-selected four CUDA worker processes and one CPU thread per worker; no silent CPU fallback is permitted.

Each completed job stores its frozen checkpoint, prediction payload, exact job identity, and SHA-256 completion marker. `--resume` validates code, design, source, primary reference, numerical versions, device, and completed job hashes. Uncommitted interrupted jobs may be recomputed; committed forecast tables and metadata must reproduce exactly. The primary reference and earlier studies are never modified.

## Intervals, contrasts, and reporting

Construct each arm/family's own five-seed historical residual blocks. At origins 2014–2018 use seven through eleven complete joint sex/age/horizon blocks with the original centering, empirical-quantile, and WIS procedures. Save historical forecasts before historical scoring; save all evaluation point forecasts, intervals, and joint draws before joining final verification values.

Report every arm and sex, adaptation harm against its identical unadapted ensemble, donor-pool harm against the newly fitted all-six GPU reference, and borrowing harm against the frozen CPU local and non-neural champions. Include 2018 horizon-five contrasts, all reliability origins/horizons, interval coverage/width/WIS, source failures, and source/country-selection audits. Save the GPU all-six versus CPU-primary comparison separately. Dependent age cells and random pools do not support independent-sample significance claims, and this analysis does not introduce new architecture searches or replace the prespecified primary endpoint.
