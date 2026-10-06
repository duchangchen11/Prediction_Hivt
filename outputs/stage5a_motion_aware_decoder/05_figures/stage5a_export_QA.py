"""Check all eight PNG/PDF/SVG bundles, source hashes and numerical plot audits."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import subprocess
import re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage5a_common import atomic_json, read_json, sha256
from PIL import Image


def main():
    quantitative=('stage5a_main_fde_comparison','stage5a_motion_group_comparison',
                  'stage5a_router_distribution','stage5a_pedestrian_motion_bins')
    cases=read_json(ROOT/'04_evaluation/stage5a_qualitative_case_manifest.json')['figures']
    names=quantitative+tuple(r['name'] for r in cases)
    assert len(names)==8
    records=[]
    for name in names:
        audit=read_json(ROOT/'05_figures'/(name+'_audit.json'));assert audit['status']=='PASS'
        for rel,digest in audit['exports_SHA256'].items():assert sha256(ROOT/rel)==digest
        png=ROOT/'05_figures'/(name+'.png')
        with Image.open(png) as im:
            size=im.size;dpi=im.info.get('dpi');im.verify()
        assert dpi and all(abs(d-300)<1 for d in dpi)
        svg=ET.parse(ROOT/'05_figures'/(name+'.svg'))
        text_nodes=svg.findall('.//{http://www.w3.org/2000/svg}text');assert len(text_nodes)>10
        pdf=ROOT/'05_figures'/(name+'.pdf')
        pdf_info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        assert re.search(r'^Pages:\s+1\s*$',pdf_info,re.MULTILINE)
        pdf_text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
        assert len(pdf_text.strip())>100
        font_output=subprocess.check_output(['pdffonts',str(pdf)],text=True)
        fonts=[line for line in font_output.splitlines()[2:] if line.strip()]
        assert fonts and all('Type 3' not in line for line in fonts)
        if name in quantitative:
            assert sha256(ROOT/audit['source_csv'])==audit['source_csv_sha256']
            if name=='stage5a_router_distribution':
                assert sha256(ROOT/'06_tables/stage5a_router_statistics.csv')==audit['statistics_csv_sha256']
            else:
                assert sha256(ROOT/'04_evaluation/stage5a_bootstrap_ci.json')==audit['bootstrap_json_sha256']
            if name=='stage5a_router_distribution':assert all(x['artist_max_abs_diff']==0 for x in audit['density_checks'])
            else:assert audit['bar_artist_max_abs_diff']==0
        else:
            assert audit['same_xy_limits'] and audit['equal_aspect'] and not audit['trajectory_edits']
            assert sha256(ROOT/audit['source_json'])==audit['source_sha256']
        records.append({'name':name,'PNG_size':size,'PNG_dpi':dpi,'SVG_editable_text_nodes':len(text_nodes),
                        'PDF_fonts':fonts,'PDF_selectable_text':True,'PNG_SHA256':sha256(png),'status':'PASS'})
    atomic_json(ROOT/'00_manifest/stage5a_export_QA.json',{'status':'PASS','bundle_count':8,
        'formats':['PNG300dpi','editable PDF','editable SVG'],'figure_records':records,
        'PDF_checker':'existing Poppler pdfinfo / pdftotext / pdffonts',
        'numerical_integrity':'artist arrays exactly match source data; display annotations rounded only',
        'manual_visual_inspection_still_required':True})
    print('STAGE5A_EXPORT_QA_PASS')


if __name__=='__main__':main()
