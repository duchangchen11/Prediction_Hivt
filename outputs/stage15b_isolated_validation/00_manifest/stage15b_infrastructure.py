"""Logging-only scalar serialization fix; frozen computation/RNG sources untouched."""
from pathlib import Path
import sys,json,runpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
_original=json.JSONEncoder.default

def numpy_scalar_default(self,value):
 if isinstance(value,np.generic):return value.item()
 return _original(self,value)

json.JSONEncoder.default=numpy_scalar_default

if __name__=='__main__':
 entry=Path(sys.argv[1]).resolve();assert entry.is_relative_to(ROOT) and entry.suffix=='.py'
 sys.argv=[str(entry),*sys.argv[2:]]
 runpy.run_path(str(entry),run_name='__main__')
