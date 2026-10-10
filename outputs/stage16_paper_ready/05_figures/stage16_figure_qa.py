"""Figure source integrity, editable vector text, PDF dimensions and case audits."""
from pathlib import Path
import sys,xml.etree.ElementTree as ET,subprocess,re
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage16_common import *
from PIL import Image
def main():
 manifest=read_json(ROOT/'05_figures/stage16_figure_manifest.json');rows=[]
 for fig in manifest['Figures']:
  name=fig['Figure'];svg=ROOT/f'05_figures/{name}.svg';pdf=ROOT/f'05_figures/{name}.pdf';png=ROOT/f'05_figures/{name}.png'
  for ext,path in [('svg',svg),('pdf',pdf),('png',png)]:assert sha256(path)==fig['Exports'][ext]['sha256']
  tree=ET.parse(svg);texts=[n.text or '' for n in tree.iter() if n.tag.endswith('}text')];assert len(texts)>5,'SVG outlined text'
  plain=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True);assert len(plain)>100,'PDF text unavailable'
  info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);assert re.search(r'Pages:\s+1',info)
  w,h=Image.open(png).size;assert w>2000 and h>1000
  for source in fig['SourceFiles']:
   path=ROOT/source if (ROOT/source).exists() else PROJECT/source
   assert path.exists(),source
   assert sha256(path)==fig['SourceSHA256'][source],source
  if name not in ('fig1_architecture','fig2_future_interaction_graph'):
   assert 'Custom scene-isolated' in plain and 'nuScenes' in plain and 'actor windows' in plain,name
  rows.append({'Figure':name,'EditableSVGTextElements':len(texts),'SelectablePDFText':True,'PDFSinglePage':True,'PNGPixels':(w,h),'SourceFilesExist':True,'SourceSHA256Match':True,'SourceValuesNotReconstructed':True})
 cases=read_json(ROOT/'07_cases/stage16_case_manifest.json')
 for c in cases['Cases']:
  role=OLD/f"05_candidate_interface/cache/fold{c['Fold']}/OuterTest";signed=read_json(role/'manifest.json')
  entry=next(x for x in signed['Contexts'] if x['scene_token']==c['SceneToken'])
  assert entry['sha256']==c['SourceContextSHA256']==sha256(PROJECT/c['SourceContextPath'])
  assert sha256(ROOT/c['LocalCoordinateFile'])==c['CoordinateFileSHA256']
 # Independently verify drawn metric-table means against immutable raw evaluation.
 f,m,p,z=frozen_eval();tables=pd.read_csv(ROOT/'06_source_data/stage16_seven_model_metrics.csv');pooled=tables[tables.Fold=='Pooled']
 for group,mask in groups(f).items():
  for j,name in enumerate(MODELS):
   row=pooled[(pooled.Group==group)&(pooled.Model==name)].iloc[0];assert row.Count==mask.sum()
   for field in ('Top1FDE','Top1ADE','minFDE6','HitRate'):assert abs(row[field]-m[mask,j,FIELDS.index(field)].mean())<1e-9
 review_path=ROOT/'05_figures/stage16_visual_review.json'
 visual_pass=False
 if review_path.exists():
  review=read_json(review_path)
  visual_pass=review['Status']=='PASS' and len(review['PNG_SHA256'])==8 and all(review['PNG_SHA256'].get(fig['Figure'])==fig['Exports']['png']['sha256'] for fig in manifest['Figures'])
 atomic_json(ROOT/'05_figures/stage16_figure_qa.json',{'Status':'PASS_DATA_VECTOR_EXPORT_CHECKS','Backend':'Python/matplotlib','Figures':rows,'AllCaseContextsMatchOriginalRecordedSHA':True,'AllDrawnFrozenAggregateValuesMatchRawArrays':True,'VisualQAStatus':'PASS_ALL8_REVIEWED' if visual_pass else 'PENDING_FINAL_REVIEW','VisualReview':'All eight PNG exports visually inspected after layout revisions; text, arrow routing, axis labels, source denominators and S2 panel-title wrap reviewed.' if visual_pass else 'Final visual receipt missing or exports changed; review current PNGs.','DatasetLabel':'custom internal CV, never official test','PNG_DPI':400,'RawCoordinatesUploaded':False})
 print('PASS',len(rows),'figure export/source checks; original-manifest real-case context SHA and metric values exact')
if __name__=='__main__':main()
