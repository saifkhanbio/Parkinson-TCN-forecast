# Source access and reuse

The study uses modelled aggregate GBD disease estimates and public demographic tabulations. It does not contain individual patient records.

- Obtain GBD 2023 exports from [GBD Results](https://vizhub.healthdata.org/gbd-results/) under IHME access and reuse conditions. Use the extraction definitions, identifiers and SHA-256 hashes in the locked specification and source citation crosswalk. The public source-verification extract retains all 4284 comparison keys and match flags, but excludes `val`, `upper`, `lower`, `workbook_value` and `native_minus_workbook`; the original file hash remains in `manifest.csv`. The acquisition and verification scripts recreate the full audit when the authorised inputs are supplied.
- Obtain United Nations World Population Prospects 2024 inputs from [UN WPP](https://population.un.org/wpp/). UN population data are © 2024 United Nations, CC BY 3.0 IGO.
- Saudi population and Ministry of Health inputs are credited to [GASTAT](https://www.stats.gov.sa/) and the [Ministry of Health](https://www.moh.gov.sa/). GCC population tabulations are credited to the GCC Statistical Centre, Sultanate of Oman, at [GCC-Stat](https://gccstat.org/). Insurance tabulations are credited to the [Council of Health Insurance](https://www.chi.gov.sa/).

The original provider bibliography, source register and file-to-citation crosswalk are in Supplementary Material S1. They distinguish modelling inputs from acquisition-only and eligibility-review resources. Original source manifests preserve hashes and provenance; a listed source is not necessarily included in this deposit.

All forecasts, demographic transformations, decompositions and graphics are the authors' analyses. The providers do not endorse the analyses or conclusions. Source: Institute for Health Metrics and Evaluation. Used with permission. All rights reserved. This prescribed acknowledgement does not document separate permission to redistribute the source datasets. Raw GBD exports are therefore not included.

Reconstruction requires the authorised upstream inputs, the preserved specifications, an appropriate numerical environment and the stage-specific execution records. Some original scripts refer to machine-specific paths. Cloning this repository alone is not represented as sufficient to repeat every historical model fit.
