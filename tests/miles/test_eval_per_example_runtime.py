import json
from types import SimpleNamespace

from miles.ray.rollout import metrics
from miles.utils.types import Sample

from open_instruct.miles.rollout import eval_logging


def test_hook_preserves_native_aggregate_and_persists_compact_trials(monkeypatch, tmp_path):
    logs = []
    monkeypatch.setattr(eval_logging.tracking, "log", lambda args, value, **kw: logs.append(value))
    args = SimpleNamespace(
        save=str(tmp_path / "checkpoints"),
        wandb_always_use_train_step=False,
        eval_datasets=[],
        n_samples_per_eval_prompt=2,
        log_passrate=False,
        custom_eval_rollout_log_function_path="open_instruct.miles.rollout.eval_logging.log_per_example",
    )
    samples = [Sample(index=i, prompt="first" if i < 2 else "second") for i in range(4)]
    data = {"aime_2025": {"samples": samples, "rewards": [1, 0, 1, 1]}}
    # Native metrics beyond rewards expect generation fields; exercise the logging
    # contract while the separate refresh suite covers complete generated samples.
    monkeypatch.setattr(metrics, "_compute_metrics_from_samples", lambda *a: {})
    for extra in (None, {}):
        result = metrics.log_eval_rollout_data(99, args, data, extra)
        assert result["eval/aime_2025"] == 0.75
        assert result["eval/step"] == 99
    compact = [json.loads(row["eval/per_example"]) for row in logs if "eval/per_example" in row]
    assert all([p["correct"] for p in event["aime_2025"]] == [1, 2] for event in compact)
    assert len(compact) == 2
    records = [
        json.loads(line)
        for path in (tmp_path / "eval_per_example").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]["eval_step"] == 99
