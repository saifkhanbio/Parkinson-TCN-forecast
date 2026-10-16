"""Add a source-verified study synthesis without altering any scientific outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/study_synthesis_v1_2"
OLD = ROOT / "reports/study_synthesis_v1_1"
MIX = ROOT / "reports/distribution_mixture_v1"
TITLE = ("Sex-Specific Transfer Learning and Forecast Reliability for Parkinson’s "
         "Disease in Saudi Arabia: A GBD 2023 Study with Gulf Benchmarking")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_text())


def frame(path):
    return pd.read_csv(ROOT / path, float_precision="round_trip")


def check_hashes(hashes):
    for name, expected in hashes.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Hash mismatch: " + name)


def verify_sources():
    """Verify existing preservation snapshots and the reports used for synthesis."""
    hashes = {}

    def add(path, digest):
        name = str(path.relative_to(ROOT))
        if name in hashes and hashes[name] != digest:
            raise ValueError("Conflicting frozen hashes: " + name)
        hashes[name] = digest

    records = [
        ("work/distribution-mixture-validation/preflight.json", ROOT,
         ["protected_sha256", "input_sha256", "code_sha256", "specification_sha256"]),
        ("reports/study_synthesis_v1_1/validation.json", ROOT, ["source_sha256"]),
        ("reports/study_synthesis_v1_1/validation.json", OLD, ["artifact_sha256"]),
        ("reports/distribution_mixture_v1/validation.json", MIX, ["artifact_sha256"]),
        ("results/distribution_mixture_v1/run_manifest.json",
         ROOT / "results/distribution_mixture_v1", ["output_sha256"]),
        ("results/source_asr_verification_v1/validation.json", ROOT,
         ["input_sha256", "protected_sha256", "output_sha256"]),
        ("reports/source_asr_verification_v1/validation.json", ROOT, ["artifact_sha256"]),
        ("results/source_hierarchy_verification_v1/validation.json", ROOT,
         ["input_sha256", "protected_sha256", "output_sha256"]),
    ]
    for name, base, keys in records:
        add(ROOT / name, sha(ROOT / name))
        data = read_json(name)
        for key in keys:
            for relative, digest in data[key].items():
                add(base / relative, digest)
    mixture_validation = read_json("reports/distribution_mixture_v1/validation.json")
    assert mixture_validation["passed"] and mixture_validation["all_cases_committed_before_scoring"]
    assert not mixture_validation["primary_guard_pass"]
    assert mixture_validation["source_manifest_sha256"] == sha(ROOT / "results/distribution_mixture_v1/run_manifest.json")
    assert read_json("results/distribution_mixture_v1/run_manifest.json")["status"] == "complete"
    check_hashes(hashes)
    return hashes


def md_table(table, columns, digits=2):
    lines = ["| " + " | ".join(columns.values()) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in table.iterrows():
        cells = []
        for key in columns:
            value = row[key]
            cells.append(f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired(table, keys):
    metrics = ["coverage", "mean_width", "wis"]
    control = table.loc[table.family.eq("tcn_adapted__cdf"), keys + metrics]
    candidate = table.loc[table.family.eq("mixture__equal_weight"), keys + metrics]
    result = control.merge(candidate, on=keys, suffixes=("_tcn", "_mixture"), validate="one_to_one")
    result["coverage_tcn_pct"] = 100 * result.coverage_tcn
    result["coverage_mixture_pct"] = 100 * result.coverage_mixture
    result["wis_change_pct"] = 100 * (result.wis_mixture / result.wis_tcn - 1)
    result["width_change_pct"] = 100 * (result.mean_width_mixture / result.mean_width_tcn - 1)
    return result.sort_values(keys).reset_index(drop=True)


def check_links(outputs):
    count = 0
    for name, data in outputs.items():
        if not name.endswith(".md"):
            continue
        for link in re.findall(r"\]\(([^)]+)\)", data.decode()):
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            destination = ((OUT / name).parent / link.split("#", 1)[0]).resolve()
            pending = destination.parent == OUT and destination.name in outputs
            if not pending and not destination.is_file():
                raise ValueError(f"Broken link in {name}: {link}")
            count += 1
    return count


def build():
    if OUT.exists():
        raise FileExistsError(f"Refusing to replace {OUT}; use --verify-only")
    preserved = verify_sources()
    sources = dict(preserved)
    outputs = {}

    def save(name, content):
        outputs[name] = content if isinstance(content, bytes) else content.encode()

    def csv(name, table):
        save(name, table.to_csv(index=False))

    primary = frame("results/primary_v1/primary_contrasts.csv")
    assert len(primary) == 4 and primary.strictly_lower.sum() == 3
    assert set(primary.origin) == {2018} and set(primary.forecast_year) == {2023}
    np.testing.assert_allclose(primary.relative_improvement_percent,
                               100 * (1 - primary.tcn_error / primary.comparator_error))
    assert primary.loc[(primary.sex == "Female") & (primary.comparator == "nonneural_champion"),
                       "relative_improvement_percent"].item() < 0
    save("primary_contrasts.csv", (ROOT / "results/primary_v1/primary_contrasts.csv").read_bytes())

    rates = frame("reports/distribution_mixture_v1/rate_comparison.csv")
    selected = rates.loc[(rates.target == "Saudi Arabia") & (rates.horizon == 5)
                         & (rates.scale == "rate") & rates.age_scope.isin(["45+", "80+"])
                         & rates.family.isin(["tcn_adapted__cdf", "mixture__equal_weight"])]
    assert len(selected) == 16 and selected.origins.eq(5).all()
    rate_pairs = paired(selected, ["outcome", "sex", "age_scope"])
    assert len(rate_pairs) == 8
    csv("saudi_mixture_rate_comparison.csv", rate_pairs)
    csv("saudi_mixture_rate_rows.csv", selected)
    burden = frame("reports/distribution_mixture_v1/burden_comparison.csv")
    nodes = ["Both__45+", "Male__45+", "Female__45+", "Male__80+_within_45+", "Female__80+_within_45+"]
    burden_selected = burden.loc[(burden.target == "Saudi Arabia") & (burden.horizon == 5)
                                & (burden.population_method == "log_trend_last8")
                                & burden.node.isin(nodes)
                                & burden.family.isin(["tcn_adapted__cdf", "mixture__equal_weight"])]
    assert len(burden_selected) == 20 and burden_selected.origins.eq(5).all()
    burden_pairs = paired(burden_selected, ["outcome", "measure", "node", "unit"])
    csv("saudi_mixture_burden_comparison.csv", burden_pairs)
    female_share = burden_pairs.loc[(burden_pairs.outcome == "prevalence")
                                    & (burden_pairs.node == "Female__80+_within_45+")].iloc[0]
    np.testing.assert_allclose([female_share.coverage_tcn, female_share.coverage_mixture], [.6, .4])
    male_shares = burden_pairs.loc[burden_pairs.node == "Male__80+_within_45+"]
    np.testing.assert_allclose(male_shares[["coverage_tcn", "coverage_mixture"]], .2)
    gates = frame("reports/distribution_mixture_v1/decision_gates.csv")
    decisions = frame("reports/distribution_mixture_v1/decisions.csv")
    assert len(decisions) == 12 and not decisions.passed.any() and decisions.gates_total.eq(38).all()
    failed = gates.loc[(gates.target == "Saudi Arabia") & (gates.outcome == "prevalence") & ~gates.passed]
    assert set(failed.endpoint) == {"Female 45+", "Female__80+_within_45+"}
    csv("gcc_mixture_decisions.csv", decisions)
    csv("saudi_prevalence_failed_guards.csv", failed)

    asr = read_json("results/source_asr_verification_v1/summary.json")
    hierarchy = read_json("results/source_hierarchy_verification_v1/validation.json")
    assert asr["matched_workbook_values"] == 4284 and asr["native_rows"] == 4704
    assert asr["checks_failed"] == 0 and not asr["forecast_intervals_changed"]
    assert hierarchy["exact_match_world_bank_region_set_1300"]
    assert hierarchy["exact_match_world_bank_income_set_1305"]
    assert (hierarchy["supplied_locations"], hierarchy["matched_national_locations"],
            hierarchy["gbd_national_locations"]) == (203, 201, 204)
    status = ROOT / "results/source_asr_verification_v1/discrepancy_status_update.csv"
    save("source_status.csv", status.read_bytes())
    for name in ["saudi_2028_scenario_totals.csv", "saudi_global_asr_comparisons.csv",
                 "saudi_learning_curve_tcn.csv", "saudi_projection_scenarios.png",
                 "saudi_projection_scenarios.svg"]:
        save(name, (OLD / name).read_bytes())

    rate_table = md_table(rate_pairs.loc[rate_pairs.age_scope == "45+"], {
        "outcome": "Outcome", "sex": "Sex", "coverage_tcn_pct": "TCN coverage %",
        "coverage_mixture_pct": "Mixture coverage %", "wis_change_pct": "WIS change %",
        "width_change_pct": "Width change %"})
    oldest_table = md_table(rate_pairs.loc[rate_pairs.age_scope == "80+"], {
        "outcome": "Outcome", "sex": "Sex", "coverage_tcn_pct": "TCN coverage %",
        "coverage_mixture_pct": "Mixture coverage %", "wis_change_pct": "WIS change %"})
    primary_table = md_table(primary, {"sex": "Sex", "comparator_source_family": "Comparator",
        "tcn_error": "TCN ALE", "comparator_error": "Comparator ALE",
        "relative_improvement_percent": "Relative improvement %"}, 6)

    source_section = """## Source verification and remaining uncertainty

The new native GBD export supplies **4,704 full-age ASR records with lower/upper source bounds** for all six GCC countries and Jordan. All **4,284 overlapping workbook values**, including all 54 Saudi checkpoints and 612 Saudi overlapping values, agree within workbook rounding (maximum absolute difference 0.004999 per 100,000). The other 420 records extend deaths/YLL history to 1980–1989. No outcome correction or model rerun was warranted. This independent confirmation covers seven of the 203 workbook locations; the other 196 locations and the workbook’s original production history remain incompletely verified. [Native ASR verification](../source_asr_verification_v1/report.md).

The official hierarchy confirms that the 203-location roster exactly matches its World Bank region and income sets: 201 of 204 main GBD national locations plus Hong Kong and Macao. Cook Islands, Niue and Tokelau are absent. This corrects the earlier description of national coverage; it does not change donors or establish population-boundary overlap. The six GCC identifiers, Jordan and Parkinson’s cause 544 are verified. A sentence mentioning 2021 remains in the officially titled 2023 information sheet and is retained as a metadata caveat. [Hierarchy verification](../source_hierarchy_verification_v1/report.md).

Saudi 2019 both-sex prevalence ASR is independently reproduced as **241.6821** (source bounds 198.0247–279.9642), versus 107.6 in the older published table: a **124.61% same-year source difference**. Rounding/transcription does not explain that contrast. Common full-age components, numerical standard weights and definitions remain unavailable for harmonization; its cause cannot be assigned to epidemiological growth or a particular release revision. Annual historical age–sex panels remain unavailable for cross-release forecast validation. [Release comparison](../release_sensitivity_v1/report.md).

The Saudi 2023 GBD-implied 80+ population remains 37.39% below GASTAT for males and 49.65% below for females. Residency, reference-date and vintage compatibility are unresolved. GBD-implied populations come from Number/Rate, rather than an independent population export. Newly available UN files provide 3,600 age–sex–year marginal quantiles for GCC populations in 2024–2028, but no shared trajectory identifiers. Their 95–99 and 100+ quantiles cannot be added to obtain a 95+ probability interval. Source ASR bounds, age-specific bounds and UN marginal quantiles do not supply joint disease–demographic trajectories or calibrated prediction intervals. They were not inserted into the frozen forecasts. [Source audit](../source_compatibility_v1/report.md), [population sensitivity](../population_sensitivity_v1/report.md), [current discrepancy ledger](source_status.csv).
"""

    mixture_section = f"""### Bounded equal-weight distribution mixture

The final exploratory experiment combined the original local statistical, pooled non-neural and adapted-TCN predictive distributions with fixed one-third weights. One model role applies to an entire age–sex–horizon trajectory, with paired historical population errors and count/share/ratio transformations retained. The 21–33 mixture atoms represent only 7–11 distinct overlapping historical origins. No models or weights were refitted. The twelve GCC country–outcome cases were committed before new scoring, but their evaluation years had already been inspected in earlier work; this remains post-inspection research.

Both mixture and component controls use inverse empirical CDF quantiles. Archived linear-quantile controls were reproduced separately; their numerical differences are not mixture improvements. Thus the CDF TCN coverage below differs from the original linear-quantile coverage reported earlier. The primary point comparison remains the single 2018→2023 endpoint; this table averages horizon-five results across origins 2014–2018.

Nominal 80% coverage at ages 45+; weighted interval score (WIS) uses the 50%/80% intervals. Negative WIS change means improvement relative to the matched TCN control. Width and WIS use rates per 100,000; changes are percentages. Each sex summary contains 55 dependent age–origin cells.

{rate_table}

Saudi prevalence WIS improved by 13.36% in males and 10.73% in females, with wider intervals. Female rate coverage fell from 61.82% to 58.18%. Female 80+ prevalence-burden-share coverage fell from three of five to two of five origins. These were the two failed guards out of 38; the candidate therefore failed its fixed joint decision rule. The 36 passing guards are not a success probability or evidence of adequate calibration. Male 80+ share coverage remained one of five for both outcomes.

The four oldest groups (80–84, 85–89, 90–94 and 95+) remain included. Their aggregate rate results comprise twenty dependent age–origin cells per sex:

{oldest_table}

For incidence, greater male 45+ coverage accompanied a 13.76% worse WIS; male 80+ rate coverage fell from 80% to 60%. Saudi incidence passed only 30 of 38 guards, and none of the twelve GCC country–outcome cases passed every guard. Both-sex 45+ count coverage was already five of five for the matched TCN and remained five of five with the mixture for both outcomes; aggregation coverage is not evidence that age composition is reliable. The 5% score tolerances and twofold width ceiling are exploratory research rules, not clinical thresholds or significance tests.

Retain the mixture as a sensitivity analysis. Do not replace the primary procedure, tune new weights on these years, exclude old ages, or claim repaired interval calibration. An independent computational audit replayed 277,200 rate interval rows and 510,300 derived-burden interval rows. Numerical validity establishes reproducibility, not external predictive validity. [Complete experiment](../distribution_mixture_v1/report.md), [rate comparisons](saudi_mixture_rate_comparison.csv), [derived-burden comparisons](saudi_mixture_burden_comparison.csv), [GCC decisions](gcc_mixture_decisions.csv), [quantile convention control](../distribution_mixture_v1/quantile_convention_effect.csv).
"""

    manuscript = (OLD / "manuscript_results.md").read_text()
    start = manuscript.index("## Design and evidence boundaries")
    manuscript = f"# Manuscript results draft\n\n**Proposed editorial title:** {TITLE}\n\n" + (
        "**Integration status, 30 September 2026:** Version 1.2 incorporates native regional ASR verification, "
        "the official location hierarchy and the completed bounded mixture experiment. Original results and "
        "the failed joint primary criterion are preserved. This editorial title does not amend the locked protocol.\n\n"
    ) + manuscript[start:]
    marker = "## Primary result: Saudi prevalence"
    manuscript = manuscript.replace(marker, source_section + "\n" + marker, 1)
    old_scope = ("Geographic labels and workbook values passed internal checks; native extraction, precise "
                 "standard weights and GBD boundary compatibility remain unverified.")
    assert manuscript.count(old_scope) == 1
    manuscript = manuscript.replace(old_scope,
        "The official hierarchy verifies the World Bank classification roster, and the native export verifies "
        "all overlapping ASR values for the seven regional locations. The remaining 196 locations, exact "
        "standard weights and population-boundary compatibility remain incompletely verified.")
    marker = "## Interpretation for the discussion"
    manuscript = manuscript.replace(marker, mixture_section + "\n" + marker, 1)
    manuscript = manuscript.split("## Table and figure placement map")[0]
    manuscript += ("\nThe [manuscript plan](manuscript_plan.md) defines the proposed framing, objectives, "
                   "figure/table placement, uncertainty language and remaining validation priorities.\n")
    save("manuscript_results.md", manuscript)

    report = f"""# Integrated Parkinson’s forecasting synthesis — verified sources and final mixture

Prepared 30 September 2026. This additive version supersedes the interpretation in version 1.1 where source-verification status has changed. Earlier reports, locked analyses, raw files and forecasts remain unchanged.

**Decision: move to manuscript assembly, retaining the failed joint primary comparison and the mixture as exploratory.** The completed analyses support selective transfer benefits and important reliability failures. They do not establish a generally superior forecasting model or adequate interval calibration.

## Primary result and study scope

The primary endpoint is Saudi GBD-modeled **age-specific prevalence at ages 45+**, evaluated separately by sex, averaging absolute log error (ALE) equally over eleven age bands at the 2018→2023 five-year endpoint. Full-age ASR is a separate supporting estimand. Saudi Arabia is the primary target; the other five GCC countries repeat the procedure. Incidence is the key secondary outcome. Mortality/disability remain supporting.

{primary_table}

TCN improves both male comparisons and the female local comparison but is 6.37% worse than the female non-neural comparator. The joint criterion remains unmet. This locally locked, retrospectively evaluated design is not externally registered or a blinded confirmatory trial. [Frozen contrasts](primary_contrasts.csv), [locked protocol](../../study_design/locked_v1/protocol.md).

## What the latest experiment adds

Nominal 80% rate coverage at ages 45+, horizon five, averaged over five overlapping origins; both methods use the same inverse empirical CDF quantiles. WIS is the 50%/80% weighted interval score. Negative WIS change is favorable.

{rate_table}

The prevalence mixture lowers WIS in both sexes, but female coverage worsens. Female 80+ prevalence-share coverage also declines from 3/5 to 2/5, while male 80+ share coverage stays at 1/5 for both outcomes. These two female failures prevent accepting the prevalence mixture under the fixed rule (36/38 guards passed). Saudi incidence passes 30/38 guards; none of twelve GCC cases passes all guards. Male incidence illustrates why coverage alone is insufficient: coverage increases while WIS worsens. Counts covered at all five origins do not validate age shares or biological burden.

![Saudi mixture and component rate coverage](../distribution_mixture_v1/saudi_coverage.png)

The mixture used saved complete trajectories, fixed equal weights and no new fits. All evaluation years had been inspected before this extension. Quantile-convention changes are documented separately and are not credited to pooling. Continue to display ages 80+ through the open 95+ band and all adverse findings. [Experiment report](../distribution_mixture_v1/report.md), [failed Saudi prevalence guards](saudi_prevalence_failed_guards.csv), [derived-burden table](saudi_mixture_burden_comparison.csv).

{source_section}

## Findings retained from the complete study

- Saudi incidence improves all four single-endpoint TCN comparisons; its female rolling error is nevertheless worse than the local comparator. Only Saudi incidence and Bahrain prevalence meet all four endpoint comparisons among twelve GCC settings. [Secondary results](../secondary_v1/report.md).
- Donor restriction and target adaptation can harm females. Historically similar donors are not reliably better donors. The fixed-checkpoint history experiment also shows no uniformly improving 15/20/29-year curve. [Donor study](../donor_comparisons_gpu_v1/report.md), [history budgets](../learning_curves_v1/report.md).
- Realized-population oracle diagnostics reduce some oldest-age count errors but are not operational improvements. Native-count reconciliation achieves arithmetic consistency without consistent accuracy gains. [Population study](../population_sensitivity_v1/report.md), [reconciliation](../demography_v1/native_count_report.md).
- Mortality/disability and earlier calibration/dynamic-model experiments retain substantial oldest-age failures. They do not rescue the primary result. [Supporting outcomes](../supporting_v1/report.md), [reliability experiments](../reliability_v1_3/report.md).
- The successful CPU full-age ASR recovery is the scientific global-donor comparison; the archived GPU neural results used persistence fallbacks. All 120 CPU seed fits succeeded. Global versus regional donors improve horizon-five TCN error in 8/24 dependent country–outcome–sex comparisons, helping Saudi males but harming females at that endpoint. [Recovered benchmark](../global_asr_cpu_recovery_v1/report.md).
- The frozen 2028 prevalence projections range from about 31,844 under a GBD-baseline/UN-growth scenario to 37,144 under unaligned UN medium population; corresponding incidence totals are 4,557 and 5,242. These are ages-45+ conditional scenarios made in 2026 from a 2023 disease-data cutoff. The scenario gap is not a probability interval, and the narrow conditional intervals omit full population and GBD estimation uncertainty. The mixture did not replace these projections. [Projection results](../projections_v1/report.md), [unchanged scenario table](saudi_2028_scenario_totals.csv).

## Manuscript position and next work

Use the proposed title **{TITLE}**. Emphasize sex-specific borrowing/adaptation harm, oldest-age composition failures despite acceptable aggregate errors, and separation of source, demographic and predictive uncertainty. No priority claim is justified merely by GBD 2023, sex stratification, five-year forecasts or model combination. The earlier literature review is a focused comparison, not a systematic novelty guarantee; its subjective ratings are not outcome measures. No new literature search or revised numerical study rating was performed for this synthesis.

The [updated results draft](manuscript_results.md) contains the complete findings. The [manuscript plan](manuscript_plan.md) supplies objective wording, claim boundaries and a figure/table map. Proceed with methods, discussion and supplement assembly from these frozen artifacts. Further weight, age-range or calibration selection on the inspected years would remain exploratory. Stronger validation requires compatible annual historical vintages or genuinely unexamined external/future outcomes; population reconciliation and coherent source trajectories remain separate evidence needs.

## Verification

The reporting script verifies existing source/result/protection hashes, regenerates comparative tables directly from frozen ledgers, checks the unchanged primary verdict and mixture failures, and verifies all local links. It performs no fitting, calibration or alteration of source uncertainty. The prior mixture audit, source checks and CPU recovery retain their own independent validation records. Run `/home/saif/agpu_env/bin/python -B scripts/report_study_synthesis_verified.py --verify-only` to check this package. [Validation record](validation.json).
"""
    save("report.md", report)
    save("manuscript_plan.md", PLAN.replace("{TITLE}", TITLE))

    # Every local evidence link is also recorded as an input identity. Numerical
    # comparisons above use prior manifest-verified files, not these new hashes alone.
    for name, content in list(outputs.items()):
        if name.endswith(".md"):
            for link in re.findall(r"\]\(([^)]+)\)", content.decode()):
                if link.startswith(("https://", "http://", "#")):
                    continue
                path = ((OUT / name).parent / link.split("#", 1)[0]).resolve()
                if path.is_file() and path.is_relative_to(ROOT):
                    sources[str(path.relative_to(ROOT))] = sha(path)
    for relative in ["results/primary_v1/primary_contrasts.csv", "results/source_asr_verification_v1/summary.json"]:
        sources[relative] = sha(ROOT / relative)
    # Link validation knows this final record will be written after all checks.
    links = check_links(dict(outputs, **{"validation.json": b"{}"}))
    check_hashes(preserved)
    validation = {
        "passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "reporter_sha256": sha(Path(__file__)), "source_sha256": sources,
        "preserved_files_checked": len(preserved), "file_links_checked": links,
        "original_primary_unchanged": True, "primary_joint_success": False,
        "mixture_joint_success": False, "gcc_mixture_joint_passes": 0,
        "source_verification_scope_locations": 7, "global_workbook_locations": 203,
        "rate_comparison_rows": len(rate_pairs), "burden_comparison_rows": len(burden_pairs),
        "models_fitted": 0, "forecast_intervals_changed": False,
        "prior_manuscript_inherited_with_source_and_mixture_updates": True,
        "artifact_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()},
    }
    save("validation.json", json.dumps(validation, indent=2) + "\n")
    OUT.mkdir()
    for name, content in outputs.items():
        with (OUT / name).open("xb") as handle:
            handle.write(content)
    print(json.dumps({"passed": True, "output": str(OUT), "artifacts": len(outputs),
                      "preserved_files_checked": len(preserved), "file_links_checked": links}, indent=2))


PLAN = """# Manuscript assembly plan and claim boundaries

Prepared 30 September 2026. This is an editorial plan based on completed analyses, not a new statistical protocol or external registration.

## Proposed title and objectives

**{TITLE}**

Primary objective: determine whether regional pretraining with limited Saudi adaptation improves five-year forecasts of male and female GBD-estimated age-specific Parkinson’s prevalence at ages 45+, compared with Saudi-only statistical and pooled non-neural models. Preserve the original four-comparison joint criterion and the 2018→2023 endpoint. Equal weighting of age-specific errors is not age standardization.

Secondary objectives:

1. Evaluate borrowing and adaptation harm under all-regional, GCC-only and historically similar donor choices, separately by sex; assess global donors only in the separate full-age ASR benchmark.
2. Assess error, interval width, coverage, directional misses and WIS across compatible historical forecast origins, with explicit 80+ diagnostics and coherent counts, age shares and male/female rate ratios. Present later calibration and mixture work as exploratory.
3. Apply identical procedures to incidence as the key secondary outcome and all six GCC countries, keeping Saudi Arabia primary. Treat mortality/disability, target-history budgets and conditional 2024–2028 projections as supporting analyses.
4. Document source comparability and available release-point differences. State that sex-specific forecast stability across releases remains untested, rather than presenting unavailable validation as an achieved objective.

These statements clarify reporting of the [locked design](../../study_design/locked_v1/protocol.md). They do not change its endpoints after evaluation. The updated [results draft](manuscript_results.md) preserves adverse findings and analysis priority.

## Main argument and novelty boundaries

The joint superiority hypothesis fails. Male primary benefits coexist with female non-neural superiority; donor similarity, more history and probabilistic pooling give selective gains. Total-count accuracy can coexist with poor oldest-age burden composition. The mixture improves prevalence WIS but does not meet its joint reliability rule.

The contribution is the combined evaluation of limited adaptation, sex-specific negative transfer, age-composition reliability and demographic/source sensitivity in Saudi Arabia with matched GCC benchmarks. Do not claim the first PD forecast, GBD 2023 analysis, sex-stratified study, ensemble, temporal evaluation or Gulf comparison. The [focused literature review](../../study_design/literature_review/decision_review_2026-09-30/decision.md) and [published-study comparison](../../study_design/literature_review/decision_review_2026-09-30/pd_comparators.md) support positioning; neither is an exhaustive novelty review. Recheck bibliographic metadata and new publications at submission.

## Proposed main tables and figures

| Placement | Content and available source | Required caption qualification |
|---|---|---|
| Table 1 | Data/estimand inventory from [locked protocol](../../study_design/locked_v1/protocol.md), [native ASR verification](../source_asr_verification_v1/report.md), [hierarchy](../source_hierarchy_verification_v1/report.md) | Seven regional age-specific locations versus 203 World Bank-classified full-age ASR locations; source bounds distinct from predictive intervals. Assemble this source table during manuscript preparation. |
| Table 2 | [Four primary comparisons](primary_contrasts.csv) and [incidence/GCC contrasts](../secondary_v1/endpoint_contrasts.csv) | Primary Saudi prevalence first; no joint success; incidence/GCC explicitly secondary. |
| Table 3 | [Mixture rate comparisons](saudi_mixture_rate_comparison.csv) and [burden comparisons](saudi_mixture_burden_comparison.csv) | Exploratory; matched inverse-CDF controls; five overlapping origins; report female/share failures. |
| Figure 1 | [Primary age errors](../primary_v1/primary_age_errors.svg) | Every age through 95+ retained; single five-year endpoint, not rolling average. |
| Figure 2 | [GCC endpoint comparisons](../secondary_v1/endpoint_comparisons.svg) and [donor harm](../donor_comparisons_gpu_v1/donor_endpoint_comparison.svg) | Original primary CPU comparisons separate from same-device GPU donor sensitivity. |
| Figure 3 | [Population sources](../population_sensitivity_v1/saudi_population_sources_2023.svg) and [80+ share scenarios](../population_sensitivity_v1/saudi_80plus_share_population_scenarios.svg) | Realized-population oracle is a diagnostic; source differences are unresolved. |
| Figure 4 | [Mixture rate coverage](../distribution_mixture_v1/saudi_coverage.svg), paired with Table 3 | Show WIS/width and share failures alongside coverage; this is not a calibration success figure. |

Keep the main narrative focused. Move fixed-budget learning curves, mortality/disability and component accounting, native-count reconciliation, all earlier calibration/dynamic alternatives, global ASR recovery, release comparisons, and conditional projections to a complete supplement. The [earlier synthesis](../study_synthesis_v1_1/manuscript_results.md) links those artifacts but its source-status wording is superseded here. The [scenario figure](saudi_projection_scenarios.svg) and [totals](saudi_2028_scenario_totals.csv) remain unchanged. Include all model-family results, device/fallback audits, quantile-convention controls and dated amendment chronology. Final numbering depends on the journal.

## Language that the evidence supports

| Topic | Defensible statement | Unsupported extension |
|---|---|---|
| Transfer benefit | Error improved for particular sexes, outcomes and comparators. | TCN consistently outperforms all statistical/non-neural methods. |
| Mixture | Prevalence WIS fell, with female coverage/share deterioration and failure of the fixed rule. | Intervals are now reliable, or 36/38 guards represent a 95% success probability. |
| Source verification | All 4,284 overlapping regional ASR values match the new native export to rounding. | All 203 locations or all historical workbook construction steps are independently authenticated. |
| Release difference | Native newer Saudi 2019 ASR exceeds an older published value by 124.61%. | This measures disease growth or proves a specific revision mechanism. |
| Population | Conditional population substitutions explain a component of signed forecast error. | Population error is causally established or the stated percentage is variance explained. |
| Biology | Sex and age heterogeneity motivates hypotheses and demographic planning research. | Hormonal, genetic, migration or exposure mechanisms have been identified. |
| Uncertainty | Conditional retrospective forecast intervals have measured coverage limitations. | Marginal GBD/UN bounds or seed spread provide total disease-burden uncertainty. |
| Clinical scope | Modeled aggregate burden forecasts can be compared and stress-tested. | Individual diagnosis, survival, progression or validated service staffing can be inferred. |

## Methods and discussion assembly checklist

Describe the GBD-modeled data and source hierarchy before models. Define ALE, WIS and every denominator/age-share target. Give model features, source/target exclusions, chronological tuning/adaptation, five fixed seeds, failure handling and software/device details from the locked configuration and manifests. Separate frozen primary evaluation, supporting comparisons and every post-inspection amendment. Explain local locking and prior outcome inspection; do not call the study preregistered or blinded.

Explain centered joint residual blocks, their small overlapping origin count, original linear quantiles, matched inverse-CDF controls for the mixture, and coherent nonlinear burden transformations. Keep predicted point means separate from predictive medians. Report 50%/80% WIS as principal and 95% intervals as sparse-tail sensitivity. Do not infer significance from ages, countries, seeds or draws as independent samples. Explicitly acknowledge that reconstructed same-vintage GBD outcomes are the verification targets, not independently observed clinical truth.

In the discussion, lead with the failed joint primary result, then explain selective benefits and composition failures. Report the resolved regional source checks before remaining denominator, joint-draw and release-harmonization gaps. The new source export does not warrant correcting old values or rerunning unaffected models. The old subjective study ratings should not appear as study results, acceptance probabilities or journal rankings.

## Next research that would change the evidence

Finish manuscript and supplement assembly using the frozen results. Broad model or interval-weight searches on these same years are not the next priority. A new hierarchical model remains a possible separately justified future study, not an unfinished requirement for this manuscript.

Stronger claims require compatible annual historical GBD releases for vintage robustness, or genuinely unexamined external observations/future outcomes with definitions specified before scoring. A future prospective study must record the actual issuance date, data release/cutoff, code and source hashes, frozen forecasts, verification source and scoring calendar. Existing 2024–2028 projections were generated in 2026 from a 2023 disease-data cutoff; do not relabel them as forecasts issued in 2023.

Coherent age–sex population trajectories, joint GBD source draws and reconciled population definitions could support better uncertainty propagation. If only marginal quantiles remain available, a separate dependence-sensitivity protocol is required; summing quantiles, independent endpoint sampling or narrowing the age range cannot resolve that evidence gap.
"""


def verify_only():
    validation = json.loads((OUT / "validation.json").read_text())
    assert validation["passed"] and validation["reporter_sha256"] == sha(Path(__file__))
    check_hashes(validation["source_sha256"])
    for name, digest in validation["artifact_sha256"].items():
        assert sha(OUT / name) == digest, name
    files = {name: (OUT / name).read_bytes() for name in validation["artifact_sha256"]}
    files["validation.json"] = (OUT / "validation.json").read_bytes()
    assert check_links(files) == validation["file_links_checked"]
    print(json.dumps({"passed": True, "read_only_verification": True,
                      "source_files": len(validation["source_sha256"]),
                      "artifacts": len(validation["artifact_sha256"]),
                      "local_links": validation["file_links_checked"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify_only() if args.verify_only else build()
