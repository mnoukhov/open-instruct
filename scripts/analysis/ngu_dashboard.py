"""Local, read-only AIME dashboard using authenticated W&B and Beaker clients.

uv run python scripts/analysis/ngu_dashboard.py --port 8766
--once --export /path/dashboard.html creates an offline snapshot instead.
"""

import argparse
import concurrent.futures
import datetime
import http.server
import json
import math
import pathlib
import subprocess
import threading
import time
import urllib.parse

import wandb

ROOT = pathlib.Path(__file__).resolve().parents[2]
HTML = pathlib.Path(__file__).with_suffix(".html")
METRIC = "eval/aime_2025"
UTC = datetime.timezone.utc


def timestamp(value):
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def latest_jobs(tasks):
    """A preempted attempt must not hide a queued/running replacement replica."""
    latest = {}
    for task in tasks:
        for job in task.get("jobs", []):
            execution = job.get("execution", {})
            key = (task["id"], job.get("replicaRank", execution.get("replicaRank", 0)))
            if key not in latest or job["status"]["created"] > latest[key]["status"]["created"]:
                latest[key] = job
    return list(latest.values())


def job_state(job):
    status = job["status"]
    if status.get("exited"):
        return (
            "completed"
            if status.get("exitCode") == 0
            else "preempted"
            if status.get("canceledCode") == 1
            else "failed"
        )
    if status.get("failed") or status.get("canceled"):
        return "failed"
    return "running" if status.get("started") else "starting" if status.get("scheduled") else "queued"


def intervals_from_tasks(tasks, now):
    intervals, seen = [], set()
    for task in tasks:
        for job in task.get("jobs", []):
            status = job["status"]
            if job["id"] in seen or not status.get("started"):
                continue
            seen.add(job["id"])
            end = status.get("exited") or status.get("failed") or status.get("canceled")
            intervals.append(
                {
                    "start": timestamp(status["started"]),
                    "end": timestamp(end) if end else now,
                    "gpus": job["requests"]["gpuCount"],
                    "job": job["id"],
                }
            )
    return intervals


def compute_at(intervals, at):
    """Count all work before an eval, including abandoned work and teardown."""
    seconds = [max(0.0, min(at, item["end"]) - item["start"]) for item in intervals]
    return {
        "gpu_hours": sum(s * i["gpus"] for s, i in zip(seconds, intervals, strict=True)) / 3600,
        "active_hours": sum(seconds) / 3600,
    }


def evaluation_points(rows, intervals, submitted):
    retained = {}
    issues = []
    for row in rows:
        if not all(key in row for key in ("eval/step", METRIC, "_timestamp")):
            continue
        score = row[METRIC]
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            issues.append(f"Non-probability AIME reward at step {row['eval/step']}; scale needs verification")
            continue
        step = int(row["eval/step"])
        if step not in retained or row["_timestamp"] > retained[step]["_timestamp"]:
            retained[step] = row
    points = []
    for row in sorted(retained.values(), key=lambda r: r["_timestamp"]):
        at = row["_timestamp"]
        if not intervals or at < min(i["start"] for i in intervals) - 60:
            issues.append(f"Evaluation step {row['eval/step']} predates mapped Beaker runtime")
            continue
        points.append(
            {
                "step": int(row["eval/step"]),
                "accuracy": row[METRIC] * 100,
                "timestamp": at,
                "wall_hours": (at - submitted) / 3600,
                "run": row["run"],
                **compute_at(intervals, at),
            }
        )
    return points, sorted(set(issues))


def beaker_tasks(experiment):
    result = subprocess.run(
        ["beaker", "experiment", "tasks", experiment, "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def run_evaluations(project, run, cache, force):
    path = cache / f"{run.id}.json"
    # Ended attempts can receive delayed uploads. Re-read hourly, and on force.
    ttl = 240 if run.state == "running" else 3600
    if not force and path.exists() and time.time() - path.stat().st_mtime < ttl:
        return json.loads(path.read_text())
    rows = [
        dict(row, run=run.id) for row in run.scan_history(keys=["eval/step", METRIC, "_timestamp"], page_size=1000)
    ]
    path.write_text(json.dumps(rows))
    return rows


def collect_series(project, spec, cache, force):
    api = wandb.Api(timeout=60)
    runs = list(api.runs(project, filters={"group": spec["group"]}))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [task for result in pool.map(beaker_tasks, spec["experiments"]) for task in result]
    jobs = [j for task in tasks for j in task.get("jobs", [])]
    if not jobs:
        raise ValueError("No Beaker attempts found")
    submitted = min(timestamp(j["status"]["created"]) for j in jobs)
    now = time.time()
    intervals = intervals_from_tasks(tasks, now)
    rows = [row for run in runs for row in run_evaluations(project, run, cache, force)]
    points, issues = evaluation_points(rows, intervals, submitted)
    # A series may combine a failed initial submission and its successful rerun.
    current_tasks = beaker_tasks(spec["experiments"][-1]) if len(spec["experiments"]) > 1 else tasks
    states = [job_state(j) for j in latest_jobs(current_tasks)]
    state = next((s for s in ["failed", "running", "starting", "queued", "preempted"] if s in states), "completed")
    return {
        **spec,
        "state": state,
        "attempts": len(jobs),
        "wandb_attempts": len(runs),
        "preemptions": sum(j["status"].get("canceledCode") == 1 for j in jobs),
        "submitted": submitted,
        "points": points,
        "issues": issues,
        "compute": compute_at(intervals, now),
        "wall_hours": (now - submitted) / 3600,
    }


class Dashboard:
    def __init__(self, registry, cache):
        self.registry = registry
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = cache / "dashboard.json"
        self.data: dict | None = json.loads(self.snapshot_path.read_text()) if self.snapshot_path.exists() else None
        self.lock = threading.Lock()
        self.refreshing = False
        self.error = None

    def refresh(self, force=False):
        with self.lock:
            if self.refreshing:
                return
            self.refreshing = True
        try:
            errors, series = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    pool.submit(collect_series, self.registry["project"], spec, self.cache, force): spec
                    for spec in self.registry["runs"]
                }
                for future in concurrent.futures.as_completed(futures):
                    spec = futures[future]
                    try:
                        series.append(future.result())
                    except Exception as exc:
                        errors.append(f"{spec['label']}: {type(exc).__name__}; data refresh failed")
                        # Keep the last good series visible, explicitly marked stale.
                        old = next((r for r in (self.data or {}).get("series", []) if r["id"] == spec["id"]), None)
                        if old:
                            series.append({**old, "stale": True})
            order = {r["id"]: i for i, r in enumerate(self.registry["runs"])}
            series.sort(key=lambda r: order[r["id"]])
            if not series:
                raise RuntimeError("All source refreshes failed; check Beaker/W&B authentication")
            result = {
                "updated": time.time(),
                "project": self.registry["project"],
                "series": series,
                "errors": errors,
                "metric": METRIC,
                "refresh_seconds": 300,
            }
            self.snapshot_path.write_text(json.dumps(result))
            self.data, self.error = result, None
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.refreshing = False

    def payload(self):
        return {"data": self.data, "refreshing": self.refreshing, "error": self.error}


def handler_for(dashboard):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path == "/api/data":
                force = urllib.parse.parse_qs(url.query).get("force") == ["1"]
                age = time.time() - (dashboard.data or {}).get("updated", 0)
                if force or age > 300:
                    threading.Thread(target=dashboard.refresh, kwargs={"force": force}, daemon=True).start()
                content = json.dumps(dashboard.payload()).encode()
                mime = "application/json"
            elif url.path == "/":
                content, mime = HTML.read_bytes(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=pathlib.Path, default=ROOT / "configs/ngu/dashboard_runs.json")
    parser.add_argument(
        "--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/open-instruct/ngu-dashboard"
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--export", type=pathlib.Path)
    args = parser.parse_args()
    dashboard = Dashboard(json.loads(args.registry.read_text()), args.cache)
    if args.once:
        dashboard.refresh(force=True)
        if dashboard.error or dashboard.data is None:
            raise RuntimeError(dashboard.error)
    else:
        threading.Thread(target=dashboard.refresh, daemon=True).start()
    if args.export:
        if dashboard.data is None:
            raise ValueError("Use --once with --export")
        payload = json.dumps(dashboard.payload()).replace("<", "\\u003c")
        args.export.write_text(HTML.read_text().replace("/*SNAPSHOT*/", f"window.NGU_SNAPSHOT = {payload};"))
        print(f"Exported {args.export}")
    if not args.once:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(dashboard))
        print(f"AIME dashboard: http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
