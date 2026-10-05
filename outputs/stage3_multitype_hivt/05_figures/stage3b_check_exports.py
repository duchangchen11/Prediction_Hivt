"""Automated editable-export QA; manual image review is recorded separately."""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3b_common import read_json,atomic_json,sha256
from PIL import Image

def main():
    rows=[]
    for directory in ('05_figures','03_type_embedding'):
        for path in sorted((ROOT/directory).glob('stage3b_*_audit.json')):
            audit=read_json(path);assert audit['status']=='PASS'
            hashes=audit['exports_SHA256'];extensions={Path(p).suffix for p in hashes}
            assert extensions=={'.png','.pdf','.svg'}
            stem=path.name.removesuffix('_audit.json');row={'figure':stem,'audit_sha256':sha256(path)}
            for name,digest in hashes.items():
                export=ROOT/name;assert sha256(export)==digest
                if export.suffix=='.png':
                    with Image.open(export) as im:
                        assert im.width>=1000 and im.height>=900
                        row['PNG_pixels']=[im.width,im.height];row['PNG_dpi']=list(im.info.get('dpi',[]));im.verify()
                elif export.suffix=='.svg':
                    tree=ET.parse(export);texts=tree.findall('.//{http://www.w3.org/2000/svg}text')
                    assert len(texts)>0,'SVG text outlined or absent'
                    row['SVG_editable_text_nodes']=len(texts)
                elif export.suffix=='.pdf':
                    fonts=subprocess.check_output(['pdffonts',str(export)],text=True)
                    assert 'CID TrueType' in fonts or 'TrueType' in fonts,'PDF text missing TrueType fonts'
                    text=subprocess.check_output(['pdftotext',str(export),'-'],text=True)
                    assert len(text.strip())>20,'PDF text is not selectable'
                    row['PDF_text_selectable']=True;row['PDF_TrueType_font']=True
            if 'panels' in audit:
                assert len(audit['panels'])==2 and audit['same_xy_limits'] and audit['equal_aspect']
                for p in audit['panels']:
                    assert max(p['metric_absolute_differences_m'].values())<1e-4
                    assert len(p['future_trajectory_styles'])==3
                    assert all(s['marker_count']==12 for s in p['future_trajectory_styles'])
                row['paired_case_numeric_and_marker_audits']='PASS'
            rows.append(row)
    assert len(rows)>=10
    atomic_json(ROOT/'00_manifest/stage3b_export_QA.json',{'status':'PASS','backend':'python',
        'figure_count':len(rows),'figures':rows,'all_exports_editable_and_valid':True,
        'manual_visual_review_required_for_final_delivery':True})
    print('STAGE3B_EXPORT_QA=PASS',len(rows),flush=True)

if __name__=='__main__':main()
