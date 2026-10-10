"""Retrieve only the nine small pinned public source files required for code audit."""
from pathlib import Path
import json,hashlib,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def main():
 for filename in ('TNT-Trajectory-Prediction_code_audit.json','R-RNet_code_audit.json'):
  manifest=json.loads((ROOT/'02_literature'/filename).read_text())
  repo=manifest['repository'];commit=manifest['commit'];assert len(commit)==40
  for name,rec in manifest['code'].items():
   path=ROOT/'02_literature/cache'/repo.split('/')[1]/name
   if path.exists():data=path.read_bytes()
   else:
    url=f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
    data=urllib.request.urlopen(url,timeout=30).read();assert hashlib.sha256(data).hexdigest()==rec['sha256']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
   assert len(data)==rec['bytes'] and hashlib.sha256(data).hexdigest()==rec['sha256']
  print('PINNED_SOURCE_PASS',repo,commit)
if __name__=='__main__':main()
