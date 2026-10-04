"""Prepare exact Git trees for authorized app delivery without putting credentials in shell."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]


def git(*args):return subprocess.check_output(["git",*args],cwd=PROJECT)


def main():
    p=argparse.ArgumentParser();p.add_argument("--base",required=True);p.add_argument("--label",required=True);a=p.parse_args()
    assert a.label.replace("_","").isalnum()
    parent=a.base;commits=[]
    for commit in git("rev-list","--reverse",a.base+"..HEAD").decode().splitlines():
        entries=[]
        for name in git("diff","--name-only",parent,commit).decode().splitlines():
            raw=git("show",commit+":"+name);sha=git("rev-parse",commit+":"+name).decode().strip()
            try:content=raw.decode("utf-8");encoding="utf-8"
            except UnicodeDecodeError:content=base64.b64encode(raw).decode();encoding="base64"
            entries.append({"path":name,"sha":sha,"mode":"100644","type":"blob","content":content,"encoding":encoding})
        commits.append({"local_commit":commit,"parent_local":parent,"base_tree":git("rev-parse",parent+"^{tree}").decode().strip(),
                        "expected_tree":git("rev-parse",commit+"^{tree}").decode().strip(),"message":git("log","-1","--format=%B",commit).decode().strip(),"entries":entries})
        parent=commit
    payload={"repository":"duchangchen11/Prediction_Hivt","base_commit":a.base,"branch":"stage2c/trainval-vehicle-baseline","commits":commits}
    path=ROOT/"02_preprocessed/stage2c_metadata_cache"/f"stage2c_git_payload_{a.label}.json"
    path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(payload,ensure_ascii=True,separators=(",",":")))
    print(json.dumps({"path":str(path),"bytes":path.stat().st_size,"commits":len(commits),"entries":[len(c['entries']) for c in commits]}))


if __name__=="__main__":main()
