# NGU comparison recipes

These opt-in Deepscaler recipes use Qwen3-4B-Base with the NGU runtime integration.
See the [experiment plan and run inventory](EXPERIMENTS.md) for the fixed-K
baselines, NGU comparison arm and evaluation-over-time protocol.
Prepare the data on a host with dataset access:

```bash
uv run python scripts/data/miles/prepare_deepscaler.py /path/to/prepared-data
```

Set the model, data, output, workspace and secret paths in your run copy before
launching. Keep experiment-specific copies in ignored `runs/`.

- `qwen3_4b_base_deepscaler_baseline.toml` is the original no-NGU comparison arm.
- `qwen3_4b_base_deepscaler_ngu.toml` uses probabilistic retry (0.875).
- `qwen3_4b_base_deepscaler_ngu_budget4_k8n16.toml` uses 16 prompts, eight samples
  per attempt and up to four attempts (32 samples per chain).

NGU supports barrier and refresh publication. These recipes retain their original
barrier settings; select `core.publication_mode = "refresh"` and
`miles.use_miles_router = true` for refresh. See
[NGU controls and metrics](../../docs/miles/data-and-evaluation.md#never-give-up-and-rejected-groups).

## AIME / compute dashboard

From a CPU environment containing `wandb`, with authenticated W&B and Beaker CLI
clients, run:

```bash
uv run --no-sync python scripts/analysis/ngu_dashboard.py --port 8766
```

Open `http://127.0.0.1:8766`. The read-only local dashboard refreshes every five
minutes while open; **Refresh data** forces a history refresh. It compares the
four-GPU baseline/NGU refresh pair by default, with optional eight-GPU historical
series. Select active GPU-hours, active job-hours, submission wall time or
training steps, and raw or best-so-far accuracy. GPU model, allocation, trainer /
generator split and publication mode remain visible.

Active GPU-hours sum each Beaker attempt's started-container runtime multiplied
by assigned GPUs, clipped at each evaluation timestamp. They include evaluation,
setup and work lost to rollback, but exclude queue time and image pull before
container start; they do not measure GPU utilization or FLOPs. Repeated eval
steps retain the latest timestamp across all W&B attempts in the run group.

Run definitions live in [dashboard_runs.json](dashboard_runs.json), including the
October 30 K=8 and K=16 baseline submissions with 3h minimum runtime. No credentials
are written into the page. Cached evaluation data is stored under
`~/.cache/open-instruct/ngu-dashboard/`. Source failures keep the last successful
series with an explicit stale/error indicator. Stop the server with Ctrl-C.

For an offline, self-contained snapshot:

```bash
uv run --no-sync python scripts/analysis/ngu_dashboard.py --once --export /path/to/aime-compute.html
```

### Per-problem AIME accuracy

The K=8 and K=16 baselines opt into this MILES callback:

```toml
[miles]
custom_eval_rollout_log_function_path = "open_instruct.miles.rollout.eval_logging.log_per_example"
```

Each evaluation logs correct/trials/accuracy for every AIME problem, plus compact
`eval/per_example` JSON for the dashboard. It also appends to
`<run root>/eval_per_example/attempt-<id>.jsonl`. No full responses or token arrays
are captured. Native aggregate metrics and metric-checker results are preserved.
Missing rewards count as incorrect, matching native aggregate evaluation.

The dashboard's per-problem heatmap shows correct/trials for each evaluated
checkpoint. P01, P02, etc. refer to the prepared eval dataset's row order, rather
than official exam numbering. Prompt SHA-256 identifiers provide stable matching
across attempts. Heatmap records are joined to the retained eval by attempt and
step, so checkpoint rollbacks do not mix old and new per-problem results. Older
runs without the callback display an explicit no-data state.
