#!/usr/bin/env bash
set -u
run() { printf '\n$'; printf ' %q' "$@"; printf '\n'; "$@" 2>&1; }
run pwd
run whoami
run uname -a
run nvidia-smi
run which python
run python --version
run which pip
run pip --version
run which conda
run bash -c 'test ! -e /home/lrj/traffic_prediction || ls -ld /home/lrj/traffic_prediction'
for path in /home/lrj /home/lrj/data /home/lrj/dataset /home/lrj/datasets /media/lrj; do
  if [ -d "$path" ]; then
    run find "$path" -maxdepth 5 \( -name v1.0-mini -o -name v1.0-trainval -o -name scene.json \)
  else
    printf '%s: not present\n' "$path"
  fi
done
run find /home/lrj -maxdepth 3 -type d \( -iname '*hivt*' -o -iname '*nuscenes*' -o -iname '*trajectory*' -o -iname '*prediction*' -o -iname '*motion*' -o -iname '*forecast*' \)
run find /home/lrj -maxdepth 4 -type d -name .git
run bash -c 'for p in /home/lrj/miniconda3/bin/conda /home/lrj/anaconda3/bin/conda /opt/conda/bin/conda; do test ! -x "$p" || "$p" env list; done'
