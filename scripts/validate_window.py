"""Validate the serialized sample against original nuScenes metadata."""
import argparse

import numpy as np
import torch

from preprocessing.build_one_window import tensor_statistics
from preprocessing.common import PROJECT_ROOT, category_mapping, load_config, load_nuscenes, write_json
from preprocessing.coordinates import ego_heading_to_global, ego_to_global, quaternion_yaw, wrap_angle


def validate(window, nusc):
    th, tf = window["th"], window["tf"]
    n = len(window["instance_tokens"])
    assert len(set(window["instance_tokens"])) == n
    expected = {"agent_pos": (n, th, 2), "agent_velocity": (n, th, 2), "agent_heading": (n, th),
                "agent_type": (n,), "history_mask": (n, th), "future_pos": (n, tf, 2),
                "future_mask": (n, tf), "ego_history": (th, 2), "ego_future": (tf, 2)}
    for key, shape in expected.items():
        assert window[key].shape == shape, key
    stats = tensor_statistics(window)
    local = torch.cat([window["agent_pos"], window["future_pos"]], dim=1).numpy()
    local_heading = torch.cat([window["agent_heading"], window["future_heading"]], dim=1).numpy()
    mask = torch.cat([window["history_mask"], window["future_mask"]], dim=1).numpy()
    assert np.all(local[~mask] == 0), "Invalid position padding must be finite zero"
    origin, yaw = window["ego_origin_global"].numpy(), window["ego_yaw_global"].item()
    restored = ego_to_global(local, origin, yaw)
    restored_heading = ego_heading_to_global(local_heading.astype(np.float64), yaw)
    position_errors, heading_errors, checked = [], [], 0
    mapping = category_mapping(nusc)
    current = nusc.get("sample", window["sample_token"])
    current_tokens = {nusc.get("sample_annotation", t)["instance_token"] for t in current["anns"]
                      if nusc.get("sample_annotation", t)["category_name"] in mapping}
    assert current_tokens == set(window["instance_tokens"]), "All selected current actors must be retained"
    for i, instance in enumerate(window["instance_tokens"]):
        previous = None
        for j, sample_token in enumerate(window["sample_tokens"]):
            sample = nusc.get("sample", sample_token)
            matches = [nusc.get("sample_annotation", t) for t in sample["anns"]
                       if nusc.get("sample_annotation", t)["instance_token"] == instance]
            assert bool(matches) == bool(mask[i, j]), "Mask disagrees with metadata"
            if not matches:
                assert window["annotation_tokens"][i][j] == ""
                continue
            assert len(matches) == 1
            ann = matches[0]
            assert ann["token"] == window["annotation_tokens"][i][j]
            assert ann["instance_token"] == instance and ann["sample_token"] == sample_token
            assert mapping[ann["category_name"]] == int(window["agent_type"][i])
            if previous is not None:
                assert previous["next"] == ann["token"] and ann["prev"] == previous["token"]
            previous = ann
            position_errors.append(float(np.max(np.abs(restored[i, j]-np.array(ann["translation"][:2])))))
            heading_errors.append(float(abs(wrap_angle(restored_heading[i, j]-quaternion_yaw(ann["rotation"])))))
            checked += 1
    ego_local = torch.cat([window["ego_history"], window["ego_future"]]).numpy()
    ego_restored = ego_to_global(ego_local, origin, yaw)
    ego_heading = torch.cat([window["ego_history_heading"], window["ego_future_heading"]]).numpy().astype(np.float64)
    ego_heading_restored = ego_heading_to_global(ego_heading, yaw)
    ego_errors, ego_heading_errors = [], []
    for j, token in enumerate(window["sample_tokens"]):
        sample = nusc.get("sample", token)
        assert sample["scene_token"] == window["scene_token"]
        assert sample["timestamp"] == int(window["sample_timestamps_us"][j])
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", sd["ego_pose_token"])
        ego_errors.append(float(np.max(np.abs(ego_restored[j]-ego["translation"][:2]))))
        ego_heading_errors.append(float(abs(wrap_angle(ego_heading_restored[j]-quaternion_yaw(ego["rotation"])))))
    assert max(position_errors) < 1e-4
    assert max(heading_errors) < 1e-6
    assert max(ego_errors) < 1e-4
    assert max(ego_heading_errors) < 1e-6
    np.testing.assert_allclose(ego_local[th-1], [0, 0], atol=1e-7)
    assert abs(ego_heading[th-1]) < 1e-7
    return {"status": "PASS", "annotations_checked": checked,
            "source_instance_and_mask_check": "PASS", "source_annotation_chain_check": "PASS",
            "agent_position_max_error_m": max(position_errors), "agent_heading_max_error_rad": max(heading_errors),
            "ego_position_max_error_m": max(ego_errors), "ego_heading_max_error_rad": max(ego_heading_errors),
            "nan_count": sum(s["nan_count"] for s in stats.values()), "inf_count": sum(s["inf_count"] for s in stats.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROJECT_ROOT / "outputs/debug/one_window.pt"))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    result = validate(torch.load(args.input, weights_only=True, map_location="cpu"), load_nuscenes(load_config(args.config)))
    write_json(PROJECT_ROOT / "outputs/reports/serialized_window_validation.json", result)
    print(result)


if __name__ == "__main__":
    main()
