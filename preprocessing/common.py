from pathlib import Path
import json

import yaml
from nuscenes.nuscenes import NuScenes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TYPE_NAMES = ("vehicle", "pedestrian", "bicycle")


def load_config(path=None):
    with open(path or PROJECT_ROOT / "configs/local_paths.yaml") as handle:
        return yaml.safe_load(handle)


def load_nuscenes(config):
    return NuScenes(version=config["nuscenes_version"],
                    dataroot=config["nuscenes_root"], verbose=False)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def category_mapping(nusc):
    """Derive disjoint groups from the loaded taxonomy, including parked actors.

    Bicycle takes precedence over vehicle. All remaining vehicle.* (including
    motorcycles) and human.pedestrian.* are retained at this initial stage.
    """
    names = {c["name"] for c in nusc.category}
    if "vehicle.bicycle" not in names:
        raise ValueError("Loaded taxonomy does not contain vehicle.bicycle")
    mapping = {}
    for name in sorted(names):
        if name == "vehicle.bicycle":
            mapping[name] = 2
        elif name.startswith("vehicle."):
            mapping[name] = 0
        elif name.startswith("human.pedestrian."):
            mapping[name] = 1
    return mapping


def scene_samples(nusc, scene):
    """Traverse and validate both directions of the timestamp-ordered chain."""
    samples, seen = [], set()
    token, previous = scene["first_sample_token"], ""
    while token:
        if token in seen:
            raise ValueError("Cycle in sample chain")
        seen.add(token)
        sample = nusc.get("sample", token)
        if sample["scene_token"] != scene["token"] or sample["prev"] != previous:
            raise ValueError("Inconsistent scene / prev link")
        if samples and sample["timestamp"] <= samples[-1]["timestamp"]:
            raise ValueError("Non-increasing sample timestamps")
        samples.append(sample)
        previous, token = token, sample["next"]
    if not samples or samples[-1]["token"] != scene["last_sample_token"]:
        raise ValueError("Scene last_sample_token mismatch")
    if len(samples) != scene["nbr_samples"]:
        raise ValueError("Scene sample count mismatch")
    return samples
