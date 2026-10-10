"""Reproducible source/font/export checks; visual review is explicitly separate."""
from pathlib import Path
import sys,argparse,subprocess,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage17_common import *
from PIL import Image
def main(visual):
 protect();p=ROOT/'08_figures/stage17_figure_source_receipt.json';v=read_json(p)
 for name,h in v['SourceSHA256'].items():assert sha256(ROOT/name)==h
 counts={};pdfchecks={};pngchecks={}
 for stem in ('stage17_external_method_comparison','stage17_main_performance_table'):
  svg=ROOT/'08_figures'/(stem+'.svg');counts[svg.name]=svg.read_text().count('<text');assert counts[svg.name]>10
  pdf=svg.with_suffix('.pdf');text=subprocess.check_output(['pdffonts',str(pdf)],text=True);assert 'TrueType' in text and 'yes' in text;pdfchecks[pdf.name]=text
  png=svg.with_suffix('.png')
  with Image.open(png) as im:
   dpi=im.info['dpi'];assert min(dpi)>=399 and im.width>2000 and im.height>1000;pngchecks[png.name]={'Size':[im.width,im.height],'DPI':list(dpi)}
 v.update({'VisualReview':'PASS: final PNGs inspected; readable labels/legend/counts, no clipping or overlap' if visual else 'PENDING_MANUAL_REVIEW',
  'FinalPlotSourceSHA256':sha256(ROOT/'08_figures/stage17_figures.py'),'ExportSHA256':{str(q.relative_to(ROOT)):sha256(q) for q in (ROOT/'08_figures').iterdir() if q.suffix in ('.svg','.pdf','.png')},
  'SVGTextCounts':counts,'PDFTrueTypeFontInspection':pdfchecks,'PNGInspection':pngchecks,'VisualReviewAutomated':False})
 atomic_json(p,v)
 audit=ROOT/'00_manifest/stage17_final_audit.json'
 if audit.exists():
  a=read_json(audit);a['FiguresSourceReceiptSHA256']=sha256(p);a['FinalFigureVisualReview']='PASS' if visual else 'PENDING';atomic_json(audit,a)
 print('EXPORT_QA_PASS', 'VISUAL_REVIEW_CONFIRMED' if visual else 'MANUAL_VISUAL_REVIEW_PENDING',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--visual-review-confirmed',action='store_true');a=p.parse_args();main(a.visual_review_confirmed)
