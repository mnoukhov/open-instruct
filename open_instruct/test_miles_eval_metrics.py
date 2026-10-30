from types import SimpleNamespace

import pytest

from open_instruct.miles.rollout import eval_metrics


def test_correct_counts_failures_and_stable_identity_across_order():
    samples = [SimpleNamespace(prompt="p1", index=i) for i in range(3)]
    samples += [SimpleNamespace(prompt="p2", index=i) for i in range(3, 6)]
    rewards = [1, 0, None, 1, 1, 0]
    rows = eval_metrics.problem_results(samples, rewards, 3)
    assert [(r["problem"], r["correct"], r["trials"], r["failed_trials"]) for r in rows] == [
        (1, 1, 3, 1),
        (2, 2, 3, 0),
    ]
    assert rows[0]["accuracy"] == pytest.approx(1 / 3)
    assert eval_metrics.problem_results(samples[::-1], rewards[::-1], 3) == rows
    assert "prompt" not in rows[0]


def test_rejects_scaled_rewards_and_misaligned_samples():
    sample = SimpleNamespace(prompt="p", index=0)
    with pytest.raises(ValueError, match="binary"):
        eval_metrics.problem_results([sample], [10], 32)
    with pytest.raises(ValueError, match="align"):
        eval_metrics.problem_results([sample], [], 32)
