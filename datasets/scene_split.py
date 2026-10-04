"""Deterministic scene-first smoke-test split, not an official benchmark split."""
import random


def validate_scene_split(split, all_tokens=None):
    train, val, test = (set(split[name]) for name in ("train", "val", "test"))
    assert train.isdisjoint(val), "train/val scene leakage"
    assert train.isdisjoint(test), "train/test scene leakage"
    assert val.isdisjoint(test), "val/test scene leakage"
    assert all(len(split[name]) == len(set(split[name])) for name in split), "duplicate scene"
    if all_tokens is not None:
        assert train | val | test == set(all_tokens), "scene coverage mismatch"


def split_scenes(scene_tokens, ratios=(0.6, 0.2, 0.2), seed=42):
    tokens = list(scene_tokens)
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate scene tokens")
    if len(tokens) < 3:
        raise ValueError("At least three scenes required")
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios)-1) > 1e-9:
        raise ValueError("Three positive ratios summing to one required")
    tokens.sort()
    random.Random(seed).shuffle(tokens)
    n_train, n_val = int(len(tokens)*ratios[0]), int(len(tokens)*ratios[1])
    if min(n_train, n_val, len(tokens)-n_train-n_val) < 1:
        raise ValueError("Requested ratios produce an empty split")
    split = {"train": tokens[:n_train], "val": tokens[n_train:n_train+n_val], "test": tokens[n_train+n_val:]}
    validate_scene_split(split, tokens)
    return split


def window_anchors(nusc, scene_tokens, th=5, tf=12, max_per_scene=3):
    """Enumerate anchors only inside scenes assigned to this split."""
    from preprocessing.common import scene_samples
    import numpy as np

    if th < 2 or tf < 1 or max_per_scene < 1:
        raise ValueError("Invalid window sizes / max_per_scene")
    for token in scene_tokens:
        scene = nusc.get("scene", token)
        samples = scene_samples(nusc, scene)
        valid = list(range(th-1, len(samples)-tf))
        if not valid:
            continue
        selected = np.unique(np.linspace(0, len(valid)-1, min(max_per_scene, len(valid)), dtype=int))
        for index in selected:
            yield scene, valid[index], samples
