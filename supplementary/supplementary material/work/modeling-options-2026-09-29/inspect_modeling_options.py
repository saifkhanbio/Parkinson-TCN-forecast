"""Describe feasible forecasting targets; fit no forecasting models."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / "supporting_data/2026-09-26/raw"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def workbook_cells(path, sheet_name):
    """Read saved values from an unchanged workbook without an Excel dependency."""
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(item.itertext())
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))
            ]
        relationships = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        }
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next(s for s in workbook.find("m:sheets", NS) if s.attrib["name"] == sheet_name)
        target = relationships[sheet.attrib[f"{{{REL_NS}}}id"]]
        member = target.lstrip("/") if target.startswith("/") else "xl/" + target
        cells = {}
        for cell in ET.fromstring(archive.read(member)).findall(".//m:sheetData/m:row/m:c", NS):
            value = cell.find("m:v", NS)
            if value is not None:
                cells[cell.attrib["r"]] = strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text
        return cells


def main():
    source = next((ROOT / "More data").rglob("*.csv"))
    data = pd.read_csv(source)
    keys = ["location_name", "sex_name", "age_name", "year", "metric_name", "measure_name"]
    assert not data.duplicated(keys).any()
    assert data["val"].gt(0).all()
    wide = data.pivot(index=keys[:-1], columns="measure_name", values="val").reset_index()
    dalys = "DALYs (Disability-Adjusted Life Years)"
    ylds = "YLDs (Years Lived with Disability)"
    ylls = "YLLs (Years of Life Lost)"
    common = wide[wide.year.ge(1990)].copy()
    assert np.allclose(common[dalys], common[ylds] + common[ylls], rtol=1e-12, atol=1e-9)

    saudi = data[
        data.location_name.eq("Saudi Arabia") & data.year.eq(2023) & data.metric_name.eq("Number")
    ].copy()
    saudi["age_start"] = saudi.age_name.str.extract(r"^(\d+)").astype(int)
    summaries = []
    for (sex, measure), group in saudi.groupby(["sex_name", "measure_name"]):
        total = group.val.sum()
        summaries.append({
            "sex": sex, "measure": measure, "year": 2023,
            "count_45plus": total,
            "share_65plus_among_45plus": group.loc[group.age_start.ge(65), "val"].sum() / total,
            "share_80plus_among_45plus": group.loc[group.age_start.ge(80), "val"].sum() / total,
        })
    pd.DataFrame(summaries).to_csv(OUT / "saudi_2023_burden_composition.csv", index=False)

    rates = common[common.metric_name.eq("Rate")].copy()
    rates["yld_per_prevalent_case"] = rates[ylds] / rates.Prevalence
    ratio_summary = rates.groupby(["age_name", "sex_name"])["yld_per_prevalent_case"].agg(
        ["count", "mean", "std", "min", "max"]
    ).reset_index()
    ratio_summary["coefficient_of_variation"] = ratio_summary["std"] / ratio_summary["mean"]
    ratio_summary.to_csv(OUT / "yld_prevalence_relationship.csv", index=False)

    rehabilitation = []
    rehab_inputs = [(2023, "43.", "BCDEFGH"), (2024, "4-55", "BCDEFGHI")]
    for edition, sheet, columns in rehab_inputs:
        path = RAW / f"saudi_moh_yearbook_{edition}.xlsx"
        cells = workbook_cells(path, sheet)
        for column in columns:
            rehabilitation.append({
                "source_file": path.name, "edition": edition, "sheet": sheet,
                "source_cell": column + "7", "year": int(cells[column + "4"].removesuffix("G")),
                "reported_medical_rehabilitation_cases": float(cells[column + "7"]),
                "scope": "MOH; all conditions; source terminology retained; not PD-specific",
            })
    rehab = pd.DataFrame(rehabilitation)
    assert rehab.groupby("year").reported_medical_rehabilitation_cases.nunique().eq(1).all()
    rehab.to_csv(OUT / "moh_rehabilitation_context_by_edition.csv", index=False)

    correction_scope = {}
    with ZipFile(RAW / "WPP2024_CSV_files_update.zip") as archive:
        for member in archive.namelist():
            with archive.open(member) as stream:
                correction_scope[member] = pd.read_csv(stream, usecols=["Location"]).Location.unique().tolist()

    global_data = pd.read_csv(ROOT / "age_standard/parkinsons_ml_ready_wide.csv")
    assert not global_data.duplicated(["location_id", "year"]).any()
    used = [source, ROOT / "age_standard/parkinsons_ml_ready_wide.csv",
            RAW / "WPP2024_CSV_files_update.zip"]
    used += [RAW / f"saudi_moh_yearbook_{year}.xlsx" for year in [2023, 2024]]
    report = {
        "inspection_date": "2026-09-29", "models_fitted": False,
        "scope_decision": "Survival analyses and modeled survival scenarios excluded by user.",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in used},
        "regional_rows": len(data), "regional_countries": sorted(data.location_name.unique()),
        "global_rows": len(global_data), "global_locations": int(global_data.location_id.nunique()),
        "global_year_range": [int(global_data.year.min()), int(global_data.year.max())],
        "maximum_daly_identity_error": float((common[dalys] - common[ylds] - common[ylls]).abs().max()),
        "yld_prevalence_cv_across_countries_years_within_age_sex": {
            "minimum": float(ratio_summary.coefficient_of_variation.min()),
            "maximum": float(ratio_summary.coefficient_of_variation.max()),
        },
        "wpp_correction_locations": correction_scope,
        "rehabilitation_years": sorted(rehab.year.unique().astype(int).tolist()),
        "rehabilitation_overlapping_editions_agree": True,
        "notes": [
            "GBD estimates are not observed patient records or independent count samples.",
            "YLD/prevalence describes modeled aggregate disability burden, not measured patient severity.",
            "Source interval bounds were not summed or treated as joint uncertainty draws.",
            "Rehabilitation cases/visits terminology and 2023-region/2024-cluster geography need harmonization.",
        ],
    }
    (OUT / "feasibility_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "models_fitted": False, "regional_rows": len(data),
                      "global_locations": report["global_locations"], "rehabilitation_years": report["rehabilitation_years"]}))


if __name__ == "__main__":
    main()
