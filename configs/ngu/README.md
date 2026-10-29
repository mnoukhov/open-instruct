# NGU comparison recipes

These opt-in Deepscaler recipes use Qwen3-4B-Base with the NGU runtime integration.
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
