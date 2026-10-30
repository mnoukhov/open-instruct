"""Opt-in MILES eval hook; preserves native aggregate logging and return values."""

import json
import time
import uuid
from pathlib import Path

from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking

from open_instruct import logger_utils
from open_instruct.miles.rollout import eval_metrics

logger = logger_utils.setup_logger(__name__)
ATTEMPT = uuid.uuid4().hex


def log_per_example(rollout_id, args, data, extra_metrics=None):
    step = compute_rollout_step(args, rollout_id)
    datasets = {}
    for name, values in data.items():
        if name != "aime_2025":
            continue
        config = next((cfg for cfg in args.eval_datasets if cfg.name == name), None)
        trials = config.n_samples_per_eval_prompt if config else args.n_samples_per_eval_prompt
        try:
            datasets[name] = eval_metrics.problem_results(values["samples"], values["rewards"], trials)
        except (KeyError, ValueError, TypeError):
            logger.exception("Per-example eval data unavailable for %s at step %s", name, step)
    if not datasets:
        return False
    metrics = {"eval/per_example": json.dumps(datasets, separators=(",", ":"))}
    for name, problems in datasets.items():
        for problem in problems:
            prefix = f"eval/{name}/problem_{problem['problem']:02d}"
            metrics.update({f"{prefix}/{key}": problem[key] for key in ("accuracy", "correct", "trials")})
    # The hook's False return preserves native aggregate metrics and metric checkers.
    # Native evals may pass None; log a separate event in that case, joined by eval/step.
    if extra_metrics is not None:
        extra_metrics.update(metrics)
    else:
        tracking.log(args, {"eval/step": step, **metrics}, step_key="eval/step")
    try:
        path = Path(args.save).parent / "eval_per_example" / f"attempt-{ATTEMPT}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "eval_step": step,
                        "rollout_id": rollout_id,
                        "timestamp": time.time(),
                        "datasets": datasets,
                    }
                )
                + "\n"
            )
    except OSError:
        logger.exception("Could not persist per-example eval file at step %s; W&B metrics retained", step)
    return False
