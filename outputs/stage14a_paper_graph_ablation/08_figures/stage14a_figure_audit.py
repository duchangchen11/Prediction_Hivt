"""Export/source checks after the six PNGs have received visual inspection."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_protocol"))
from stage14a_common import *
from PIL import Image


def main():
    folder = ROOT / "08_figures"
    manifest = read_json(folder / "stage14a_figure_manifest.json")
    assert manifest["Status"] == "PASS" and manifest["FamilyCount"] == 6
    assert not manifest["IndependentOfficialTestClaim"]
    assert manifest["NoFitOrNewInference"]
    assert sha256(folder / "stage14a_figure_contract.json") == manifest["FigureContractSHA256"]
    for path, digest in manifest["SourceDataSHA256"].items():
        assert sha256(ROOT / path) == digest
    identity = read_json(ROOT / "05_evaluation/stage14a_identity_audit.json")
    assert identity["Status"] == "PASS"
    gate = read_json(ROOT / "04_checkpoints/stage14a_all_frozen.json")
    assert gate["Status"] == "FROZEN_ALL_COMPLETE" and len(gate["Checkpoints"]) == 6
    assert sha256(ROOT / "04_checkpoints/stage14a_all_frozen.json") == manifest["FrozenGateSHA256"]
    rows = []
    for number, family in enumerate(manifest["Families"], 1):
        files = {}
        text = ""
        for record in family["Files"]:
            path = ROOT / record["Path"]
            assert path.stat().st_size == record["Bytes"] and sha256(path) == record["SHA256"]
            files[record["Format"]] = record
            if record["Format"] == "svg":
                xml = ET.parse(path)
                elements = [element for element in xml.iter() if element.tag.endswith("}text")]
                assert len(elements) > 10
                text = "\n".join("".join(element.itertext()) for element in elements)
                assert "internal ranking OOF" in text and "not an independent official test" in text
                assert "n=" in text and "m" in text
                text_count = len(elements)
            elif record["Format"] == "pdf":
                data = path.read_bytes()
                assert data.startswith(b"%PDF-")
                assert b"/FontFile2" in data and b"/CIDFontType2" in data
                assert b"/Subtype /Type3" not in data
            elif record["Format"] == "png":
                with Image.open(path) as preview:
                    assert preview.format == "PNG" and min(preview.size) >= 400
                    pixels = list(preview.size)
        assert set(files) == {"svg", "pdf", "png"}
        if number <= 5:
            source_path = folder / f"stage14a_fig{number}_source_data.csv"
            source = pd.read_csv(source_path)
            assert len(source) > 0 and "Count" in source.columns
            source_sha = sha256(source_path)
        else:
            source_path = folder / "stage14a_case_manifest.json"
            cases = read_json(source_path)
            assert cases["Status"] == "PASS" and cases["AllModelsUseUnchangedGeometry"]
            assert not cases["RerankerOrPredictorInferencePerformed"]
            assert all(case.get("CandidateGeometryUnchanged", True) for case in cases["Cases"])
            source_sha = sha256(source_path)
        rows.append({
            "Figure": number, "Family": family["Family"], "Status": "PASS",
            "PNGViewedWithViewImage": True,
            "VisualLayout": "PASS: titles, axis labels, legends and footnotes readable; no clipping or superimposed titles",
            "UnitsAndActualSampleCounts": "PASS",
            "InternalOOFAndPredictorOverlapDisclosure": "PASS",
            "SVGEditableText": True, "SVGTextElements": text_count,
            "PDFEmbeddedTrueType": True, "PDFFontTypeSetting": 42,
            "PNGDimensionsPixels": pixels,
            "SourceData": str(source_path.relative_to(ROOT)), "SourceSHA256": source_sha,
            "Exports": files,
        })
    assert len(rows) == 6
    atomic_json(folder / "stage14a_figure_audit.json", {
        "Stage": "Stage14A", "Status": "PASS", "FigureFamilies": 6,
        "Backend": "Python / matplotlib exclusively",
        "VisualQA": "All six PNG previews inspected with view_image after rendering and layout repair; final case annotations avoid the GT endpoint region.",
        "LayoutRepairsOnly": [
            "Use a single title per axis and wrap long delta-axis labels",
            "Include all moving/stopped/parked vehicle states in the table, with sample counts and exploratory ParkedVehicle contrast",
            "Use square equal-metric case coordinate domains, leaving separate legend/footnote space",
            "Place case score annotations in the less occupied upper corner",
        ],
        "DataOrEvaluationMetricsChanged": False, "CaseSelectionContractChanged": False,
        "ModelFittingOrInferencePerformed": False, "ScientificDecisionMade": False,
        "IndependentOfficialTestClaim": False,
        "BonferroniFamily4IntervalsRetained": True,
        "ParkedVehiclePossibleHarmDisclosed": True,
        "CasesShowImprovementAndFailure": True,
        "CandidateCoordinatesUnchanged": True,
        "Figures": rows,
        "FigureContractSHA256": manifest["FigureContractSHA256"],
        "FigureManifestSHA256": sha256(folder / "stage14a_figure_manifest.json"),
        "FrozenGateSHA256": manifest["FrozenGateSHA256"],
        "IdentityAuditSHA256": manifest["IdentityAuditSHA256"],
        "SourceDataSHA256": manifest["SourceDataSHA256"],
    })
    print("STAGE14A_FIGURE_VISUAL_AND_EXPORT_QA_PASS", flush=True)


if __name__ == "__main__":
    main()
