"""Compact binary evaluation statistics with stable identities across attempts."""

import hashlib
import json
import math


def problem_results(samples, rewards, samples_per_prompt):
    if len(samples) != len(rewards) or samples_per_prompt < 1:
        raise ValueError("Evaluation samples/rewards must align and trial count must be positive")
    groups = {}
    for sample, reward in zip(samples, rewards, strict=True):
        if reward is not None and (
            not isinstance(reward, (int, float)) or not math.isfinite(reward) or reward not in (0, 1)
        ):
            raise ValueError("Per-problem correctness requires binary rewards (or None for failed trials)")
        prompt = json.dumps(sample.prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        identity = hashlib.sha256(prompt.encode()).hexdigest()
        group = groups.setdefault(
            identity,
            {
                "id": identity,
                "problem": sample.index // samples_per_prompt + 1,
                "correct": 0,
                "trials": 0,
                "failed_trials": 0,
            },
        )
        group["correct"] += int(reward == 1)
        group["trials"] += 1
        group["failed_trials"] += int(reward is None)
    return [
        dict(group, accuracy=group["correct"] / group["trials"])
        for group in sorted(groups.values(), key=lambda group: group["problem"])
    ]
