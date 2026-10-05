"""Validate PNG/PDF/SVG exports and numerical/source/marker traceability."""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
STAGE3_ROOT = ROOT.parent / 'stage3_multitype_hivt'
sys.path.insert(0, str(STAGE3_ROOT / '00_manifest'))
sys.path.insert(0, str(STAGE3_ROOT / '04_evaluation'))
sys.path.insert(0, str(ROOT / '00_manifest'))
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3b_common import atomic_json, read_json, sha256
from PIL import Image


def main():
    checks = []
    for directory in ('05_figures', '03_type_interaction'):
        for path in sorted((ROOT / directory).glob('stage4a_*_audit.json')):
            audit = read_json(path)
            if 'exports_SHA256' not in audit:
                continue
            assert audit['status'] == 'PASS'
            for source, digest in audit.get('source_SHA256', {}).items():
                assert sha256(ROOT / source) == digest
            if 'source_json' in audit:
                assert sha256(ROOT / audit['source_json']) == audit['source_sha256']
            assert {Path(name).suffix for name in audit['exports_SHA256']} == {'.png', '.pdf', '.svg'}
            row = {'figure': path.name.removesuffix('_audit.json'), 'audit_sha256': sha256(path)}
            for name, digest in audit['exports_SHA256'].items():
                export = ROOT / name; assert sha256(export) == digest
                if export.suffix == '.png':
                    with Image.open(export) as im:
                        assert im.width >= 1000 and im.height >= 900
                        row['PNG_pixels'] = [im.width, im.height]; row['PNG_dpi'] = list(im.info.get('dpi', [])); im.verify()
                elif export.suffix == '.svg':
                    svg = ET.parse(export)
                    texts = svg.findall('.//{http://www.w3.org/2000/svg}text')
                    assert len(texts) > 0, 'SVG editable text missing'
                    row['SVG_editable_text_nodes'] = len(texts)
                elif export.suffix == '.pdf':
                    fonts = subprocess.check_output(['pdffonts', str(export)], text=True)
                    assert 'TrueType' in fonts, 'PDF TrueType fonts missing'
                    text = subprocess.check_output(['pdftotext', str(export), '-'], text=True)
                    assert len(text.strip()) > 20, 'PDF selectable text missing'
                    row['PDF_text_selectable'] = True; row['PDF_TrueType_font'] = True
            if 'panels' in audit:
                assert len(audit['panels']) == 2 and audit['same_xy_limits'] and audit['equal_aspect']
                assert audit['same_actor_and_GT'] and audit['neighbor_eligibility_uses_t0_only']
                for panel in audit['panels']:
                    assert max(panel['metric_absolute_differences_m'].values()) < 1e-4
                    assert len(panel['future_trajectory_styles']) == 3
                    assert all(style['marker_count'] == 12 for style in panel['future_trajectory_styles'])
                row['paired_case_numeric_and_marker_audits'] = 'PASS'
            checks.append(row)
    names = {row['figure'] for row in checks}
    assert {'stage4a_loss_curve', 'stage4a_val_fde_curve', 'stage4a_training_comparison',
            'stage4a_fde_ablation', 'stage4a_interaction_context_fde', 'stage4a_type_pair_bias_heatmap',
            'stage4a_vehicle_improvement_case_001', 'stage4a_pedestrian_improvement_case_001',
            'stage4a_degradation_case_001'} <= names
    manifest = read_json(ROOT / '04_evaluation/stage4a_qualitative_case_manifest.json')
    assert manifest['status'] == 'PASS' and manifest['case_count'] >= 3
    assert all(item['name'] in names for item in manifest['figures'])
    atomic_json(ROOT / '00_manifest/stage4a_export_QA.json', {
        'status': 'PASS', 'backend': 'python', 'figure_count': len(checks), 'figures': checks,
        'all_exports_editable_and_valid': True, 'all_source_hashes_verified': True,
        'manual_visual_review_required_for_final_delivery': True})
    print('STAGE4A_EXPORT_QA=PASS', len(checks), flush=True)


if __name__ == '__main__':
    main()
