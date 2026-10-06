"""Numerical source and editable export checks for all eight figure bundles."""
from pathlib import Path
import sys,subprocess,re,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage6a_common import atomic_json,read_json,sha256
from PIL import Image

def main():
    quantitative=('stage6a_top1fde_comparison','stage6a_oracle_gap','stage6a_hit_rate','stage6a_interaction_ablation')
    cases=read_json(ROOT/'04_evaluation/stage6a_qualitative_case_manifest.json')['figures']
    names=quantitative+tuple(r['name'] for r in cases);assert len(names)==8;records=[]
    for name in names:
        audit=read_json(ROOT/'05_figures'/(name+'_audit.json'));assert audit['status']=='PASS'
        for path,digest in audit['exports_SHA256'].items():assert sha256(ROOT/path)==digest
        if name in quantitative:
            assert sha256(ROOT/audit['source_csv'])==audit['source_csv_sha256']
            assert sha256(ROOT/audit['bootstrap_csv'])==audit['bootstrap_csv_sha256']
            assert audit['artist_max_abs_diff']==0
        else:
            assert sha256(ROOT/audit['source_json'])==audit['source_sha256']
            assert audit['same_xy_limits'] and audit['equal_aspect'] and not audit['trajectory_edits']
            assert all(p['metric_difference_m']<1e-4 and p['all_six_candidates_exact'] for p in audit['panels'])
            assert all(p['probability_artist_max_abs_diff']==0 for p in audit['probability_checks'])
        png=ROOT/'05_figures'/(name+'.png')
        with Image.open(png) as im:size=im.size;dpi=im.info.get('dpi');im.verify()
        assert dpi and all(abs(d-300)<1 for d in dpi)
        svg=ET.parse(ROOT/'05_figures'/(name+'.svg'));texts=svg.findall('.//{http://www.w3.org/2000/svg}text');assert len(texts)>10
        pdf=ROOT/'05_figures'/(name+'.pdf');info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        assert re.search(r'^Pages:\s+1\s*$',info,re.MULTILINE)
        text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True);assert len(text.strip())>100
        font_output=subprocess.check_output(['pdffonts',str(pdf)],text=True)
        fonts=[x for x in font_output.splitlines()[2:] if x.strip()];assert fonts and all('Type 3' not in x for x in fonts)
        records.append({'name':name,'status':'PASS','PNG_size':size,'PNG_dpi':dpi,'PNG_SHA256':sha256(png),
            'SVG_editable_text_nodes':len(texts),'PDF_selectable_text':True,'PDF_fonts':fonts})
    atomic_json(ROOT/'00_manifest/stage6a_export_QA.json',{'status':'PASS','bundle_count':8,'figure_records':records,
        'formats':['PNG300dpi','editable PDF','editable SVG'],'PDF_checker':'existing Poppler tools',
        'numerical_integrity':'plot artist coordinates and probabilities match full-precision source; annotation rounding only',
        'manual_visual_inspection_still_required':True})
    print('STAGE6A_EXPORT_QA_PASS')

if __name__=='__main__':main()
