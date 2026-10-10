"""Restore small pinned audit files into Stage17; never update a source version."""
from pathlib import Path
import json,hashlib,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def main():
 manifests=[json.loads((ROOT/'01_source/stage17_original_scoring_source.json').read_text()),json.loads((ROOT/'01_source/stage17_extra_source_manifest.json').read_text())]
 records={**manifests[0]['code'],**manifests[1]};repo=manifests[0]['repository'];commit=manifests[0]['commit']
 for name,rec in records.items():
  path=ROOT/'01_source/cache'/name
  if path.exists():data=path.read_bytes()
  else:
   data=urllib.request.urlopen(f'https://raw.githubusercontent.com/{repo}/{commit}/{name}',timeout=30).read()
   assert hashlib.sha256(data).hexdigest()==rec['sha256'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  assert hashlib.sha256(data).hexdigest()==rec['sha256'] and len(data)==rec['bytes']
 print('PINNED_STAGE17_SOURCE_RESTORE_PASS',commit)
if __name__=='__main__':main()
