"""Inspect an existing environment without changing dependencies."""
import importlib
import json
import sys

try:
    from importlib.metadata import version as distribution_version
except ImportError:
    from pkg_resources import get_distribution
    distribution_version = lambda name: get_distribution(name).version

result = {"python": sys.version, "executable": sys.executable, "modules": {}}
for name in ["torch", "numpy", "scipy", "sklearn", "pandas", "matplotlib", "nuscenes", "torch_geometric", "pytorch_lightning", "lightning", "yaml"]:
    try:
        mod = importlib.import_module(name)
        record = {"ok": True, "version": getattr(mod, "__version__", "unknown")}
        if name == "nuscenes":
            from nuscenes.nuscenes import NuScenes
            record["version"] = distribution_version("nuscenes-devkit")
        if name == "torch":
            record["cuda_available"] = mod.cuda.is_available()
            record["cuda_runtime"] = mod.version.cuda
            if record["cuda_available"]:
                record["gpu"] = mod.cuda.get_device_name(0)
                record["cuda_compute"] = float((mod.ones(2, device="cuda") * 3).sum().item())
        result["modules"][name] = record
    except Exception as exc:
        result["modules"][name] = {"ok": False, "error": repr(exc)}
print(json.dumps(result, indent=2))
