"""Accounting checks for preempted and resumed experiment histories."""

import json

import pytest
from scripts.analysis import ngu_dashboard as dashboard


def test_compute_counts_abandoned_work_and_clips_at_evaluation():
    intervals = [{"start": 0, "end": 3600, "gpus": 4}, {"start": 7200, "end": 10800, "gpus": 8}]
    assert dashboard.compute_at(intervals, 9000) == {"gpu_hours": 8, "active_hours": 1.5}
    assert dashboard.compute_at(intervals, -1) == {"gpu_hours": 0, "active_hours": 0}


def test_latest_step_survives_rollback_but_compute_clock_never_resets():
    intervals = [{"start": 0, "end": 3600, "gpus": 4}]
    rows = [
        {"eval/step": 99, dashboard.METRIC: 0.2, "_timestamp": 1000, "run": "before"},
        {"eval/step": 99, dashboard.METRIC: 0.25, "_timestamp": 2000, "run": "resumed"},
        {"eval/step": 199, dashboard.METRIC: 0.24, "_timestamp": 3000, "run": "resumed"},
    ]
    points, issues = dashboard.evaluation_points(rows, intervals, 0)
    assert not issues
    assert [p["accuracy"] for p in points] == [25, 24]
    assert points[0]["run"] == "resumed"
    assert points[0]["gpu_hours"] == pytest.approx(2000 * 4 / 3600)


def test_unverified_reward_scale_is_flagged_instead_of_plotted_as_accuracy():
    rows = [{"eval/step": 0, dashboard.METRIC: 2.0, "_timestamp": 10, "run": "old"}]
    points, issues = dashboard.evaluation_points(rows, [{"start": 0, "end": 100, "gpus": 4}], 0)
    assert points == []
    assert "scale needs verification" in issues[0]


def test_latest_replica_replaces_preempted_attempt_without_hiding_other_replica():
    jobs = [
        {"id": "old", "replicaRank": 0, "status": {"created": "2026-01-01", "exited": "2026-01-02", "exitCode": 143}},
        {"id": "new", "replicaRank": 0, "status": {"created": "2026-01-03"}},
        {"id": "peer", "replicaRank": 1, "status": {"created": "2026-01-01", "started": "2026-01-01"}},
    ]
    latest = dashboard.latest_jobs([{"id": "task", "jobs": jobs}])
    assert {j["id"] for j in latest} == {"new", "peer"}
    assert {dashboard.job_state(j) for j in latest} == {"queued", "running"}


def test_attempt_intervals_deduplicate_and_include_running_attempt():
    job = {"id": "job", "requests": {"gpuCount": 4}, "status": {"started": "1970-01-01T00:00:10Z"}}
    intervals = dashboard.intervals_from_tasks([{"jobs": [job, job]}], 100)
    assert intervals == [{"start": 10, "end": 100, "gpus": 4, "job": "job"}]


def test_per_problem_rows_join_by_attempt_and_step_after_rollback():
    rows = [
        {"eval/step": 99, dashboard.METRIC: 0.5, "_timestamp": 100, "run": "old"},
        {
            "eval/step": 99,
            "eval/per_example": json.dumps({"aime_2025": [{"correct": 1}]}),
            "_timestamp": 99,
            "run": "old",
        },
        {"eval/step": 99, dashboard.METRIC: 1, "_timestamp": 200, "run": "new"},
        {
            "eval/step": 99,
            "eval/per_example": json.dumps({"aime_2025": [{"correct": 2}]}),
            "_timestamp": 199,
            "run": "new",
        },
    ]
    points, issues = dashboard.evaluation_points(rows, [{"start": 0, "end": 300, "gpus": 4}], 0)
    assert not issues
    assert points[0]["run"] == "new"
    assert points[0]["problems"] == [{"correct": 2}]
