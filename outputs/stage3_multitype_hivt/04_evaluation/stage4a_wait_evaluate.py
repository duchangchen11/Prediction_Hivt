"""Resume-safe Stage4A evaluator watcher; never runs concurrently with training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
EVALUATOR = ROOT / "04_evaluation/stage4a_evaluate.py"
SUMMARY = ROOT / "03_type_interaction/stage4a_training_summary.json"
BEST = ROOT / "07_checkpoints/stage4a_best_overall_minfde.pt"
STATE = ROOT / "00_manifest/stage4a_evaluation_pipeline_state.json"
LOG = ROOT / "08_logs/stage4a_final_evaluation.log"

OUTPUTS = {
    "fresh-val": ("04_evaluation/stage4a_actor_errors.csv", "04_evaluation/stage4a_metrics.json"),
    "analyze": ("04_evaluation/stage4a_pairing_audit.json", "04_evaluation/stage4a_analysis_completion.json",
        "04_evaluation/stage4a_nontrivial_motion_metrics.json", "04_evaluation/stage4a_type_interaction_ablation.json",
        "04_evaluation/stage4a_bootstrap_ci.json", "06_tables/stage4a_main_results.csv",
        "06_tables/stage4a_nontrivial_motion_metrics.csv", "06_tables/stage4a_interaction_subgroup_metrics.csv",
        "06_tables/stage4a_type_interaction_ablation.csv", "06_tables/stage4a_bootstrap_ci.csv"),
    "diagnostics": ("04_evaluation/stage4a_relation_bias_statistics.json", "06_tables/stage4a_relation_bias_statistics.csv",
        "06_tables/stage4a_relation_bias_pair_aggregate.csv"),
    "efficiency": ("04_evaluation/stage4a_efficiency_audit.json", "06_tables/stage4a_efficiency.csv",
        "06_tables/stage4a_efficiency_batch_timings.csv"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save(state):
    temp = STATE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, STATE)


def pid_active(pid):
    path = Path("/proc") / str(pid)
    if not path.exists():
        return False
    try:
        # A zombie has exited and cannot occupy the GPU.
        raw = (path / "stat").read_text()
        return raw.split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def training_processes():
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
            if command and b"python" in Path(os.fsdecode(command[0])).name.encode():
                if any(Path(os.fsdecode(arg)).name == "stage4a_train.py" for arg in command[1:] if arg):
                    if pid_active(int(entry.name)):
                        result.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def verify_completed(state):
    for stage in state.get("completed_stages", []):
        recorded = state["stages"][stage]
        assert recorded["status"] == "PASS" and recorded["returncode"] == 0
        for relative, expected in recorded["outputs_SHA256"].items():
            assert sha256(ROOT / relative) == expected, "Completed-stage artifact changed: " + relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-pid", type=int, required=True)
    args = parser.parse_args()
    previous = read_json(STATE) if STATE.exists() else {}
    state = {**previous, "status": "WAITING", "training_PID": args.training_pid,
        "watcher_started_UTC": now(), "watcher_source_SHA256": sha256(Path(__file__)),
        "evaluator_source_SHA256": sha256(EVALUATOR), "completed_stages": previous.get("completed_stages", []),
        "stages": previous.get("stages", {}), "waiting_policy": "poll 15s; GPU work only after training PID exit and training summary COMPLETE",
        "formal_training_and_final_evaluation_GPU_concurrent": False}
    for key in ("failed_stage", "failure", "failed_UTC"):
        state.pop(key, None)
    save(state)
    try:
        verify_completed(state)
        print("STAGE4A_FINAL_EVALUATION_WAITING_FOR_TRAINING_PID", args.training_pid, flush=True)
        while pid_active(args.training_pid):
            time.sleep(15)
        assert SUMMARY.exists(), "Training exited without completion summary; stop"
        training = read_json(SUMMARY)
        assert training.get("status") == "COMPLETE", "Training exited without COMPLETE status; stop"
        assert not training_processes(), "Another Stage4A training process is active; stop"
        checkpoint = sha256(BEST)
        assert training["checkpoint_sha256"] == checkpoint
        if state.get("checkpoint_SHA256"):
            assert state["checkpoint_SHA256"] == checkpoint, "Checkpoint changed since prior evaluator run"
        state.update(status="RUNNING", training_exited_UTC=now(), checkpoint_SHA256=checkpoint,
            training_summary_SHA256=sha256(SUMMARY), training_COMPLETE_verified=True,
            no_other_Stage4A_training_process=True)
        save(state)
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", buffering=1) as handle:
            handle.write("\nSTAGE4A FINAL EVALUATION RUN " + now() + "\n")
            for stage, outputs in OUTPUTS.items():
                if stage in state["completed_stages"]:
                    print("STAGE4A_FINAL_EVALUATION_RESUME_SKIP", stage, flush=True)
                    continue
                assert not training_processes(), "Stage4A training appeared during final evaluation; stop"
                state["active_stage"] = stage
                state["stages"][stage] = {"status": "RUNNING", "started_UTC": now(),
                    "evaluator_source_SHA256": sha256(EVALUATOR)}
                save(state)
                command = [sys.executable, str(EVALUATOR), "--" + stage]
                handle.write("\nCOMMAND " + json.dumps(command) + " UTC=" + now() + "\n")
                print("STAGE4A_FINAL_EVALUATION_STAGE_START", stage, flush=True)
                process = subprocess.run(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT, check=False)
                state["stages"][stage]["returncode"] = process.returncode
                assert process.returncode == 0, "Evaluation stage failed: " + stage
                for relative in outputs:
                    assert (ROOT / relative).is_file(), "Missing stage artifact: " + relative
                state["stages"][stage].update(status="PASS", completed_UTC=now(),
                    outputs_SHA256={relative: sha256(ROOT / relative) for relative in outputs})
                state["completed_stages"].append(stage)
                state.pop("active_stage", None)
                save(state)
                print("STAGE4A_FINAL_EVALUATION_STAGE_PASS", stage, flush=True)
        verify_completed(state)
        assert state["completed_stages"] == list(OUTPUTS)
        state.update(status="PASS", completed_UTC=now(), all_evaluation_subprocesses_exited=True,
            evaluation_log_SHA256=sha256(LOG), evaluator_source_SHA256=sha256(EVALUATOR))
        save(state)
        print("STAGE4A_FINAL_EVALUATION_PIPELINE=PASS", flush=True)
    except Exception as error:
        state.update(status="FAIL", failure=str(error), failed_UTC=now())
        if state.get("active_stage"):
            state["failed_stage"] = state["active_stage"]
            state["stages"][state["active_stage"]]["status"] = "FAIL"
        save(state)
        print("STAGE4A_FINAL_EVALUATION_PIPELINE=FAIL", str(error), flush=True)
        raise


if __name__ == "__main__":
    main()
