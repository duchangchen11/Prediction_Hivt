"""Align inherited diagnostic metadata with measured VAL, proving all training state unchanged."""
import hashlib
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3a_plus_common import BEST_MANIFEST, read_json, atomic_json, sha256, verify_freeze
import torch


def state_equal(a, b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def model_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    verify_freeze(); m = read_json(BEST_MANIFEST); assert m['status'] == 'COMPLETE'
    if not m['best_refreshed']:
        atomic_json(ROOT / '00_manifest/stage3a_plus_checkpoint_metadata_audit.json', {
            'status': 'PASS', 'original_best_retained': True, 'metadata_correction_needed': False,
            'model_optimizer_RNG_and_cursor_changed': False})
        print('CHECKPOINT_METADATA=PASS; original retained', flush=True); return
    path = ROOT / m['final_best']['relative_path']
    assert path.name == 'stage3a_plus_best_overall_minfde.pt' and sha256(path) == m['final_best']['sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    measured_path = ROOT / '04_evaluation/stage3a_plus_best_val_metrics.json'
    measured = read_json(measured_path); full = measured['metrics']['full_horizon']
    assert abs(saved['metadata']['validation_FDE'] - full['overall']['minFDE6']) < 1e-10
    before = sha256(path); digest = model_digest(saved['state_dict'])
    expected = {'moving_ADE': full['vehicle.moving']['minADE6'], 'moving_FDE': full['vehicle.moving']['minFDE6']}
    corrected = {key: {'before': saved['metadata'][key], 'after': value} for key, value in expected.items() if saved['metadata'][key] != value}
    metadata = dict(saved['metadata']); metadata.update(expected)
    if corrected:
        payload = {**saved, 'metadata': metadata}; temp = path.with_suffix('.pt.tmp')
        torch.save(payload, temp)
        restored = torch.load(temp, map_location='cpu', weights_only=False)
        assert all(state_equal(saved[k], restored[k]) for k in saved if k != 'metadata')
        assert model_digest(restored['state_dict']) == digest
        os.replace(temp, path)
        measured['checkpoint_metadata'] = metadata; atomic_json(measured_path, measured)
        m['final_best']['sha256'] = sha256(path)
    m['final_best']['model_state_content_sha256'] = digest
    m['checkpoint_metadata_finalization'] = {'inherited_moving_diagnostics_aligned_with_selected_VAL': True,
                                            'model_optimizer_RNG_and_cursor_unchanged': True}
    atomic_json(BEST_MANIFEST, m)
    atomic_json(ROOT / '00_manifest/stage3a_plus_checkpoint_metadata_audit.json', {
        'status': 'PASS', 'original_best_retained': False, 'corrected_diagnostic_fields': corrected,
        'checkpoint_file_sha256_before': before, 'checkpoint_file_sha256_after': sha256(path),
        'model_state_content_sha256': digest, 'all_nonmetadata_payload_fields_bitwise_identical': True,
        'model_optimizer_RNG_and_cursor_changed': False,
        'reason': 'Extension checkpoints inherited parent moving_ADE/FDE diagnostics; official selected-VAL JSON is authoritative.'})
    verify_freeze(); print('CHECKPOINT_METADATA=PASS; training state unchanged', list(corrected), flush=True)


if __name__ == '__main__': main()
