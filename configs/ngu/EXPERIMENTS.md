# NGU: evaluation performance over time

## Question

Does Never Give Up (NGU) reach better evaluation performance sooner than spending
a fixed number of samples on every prompt? Compare Qwen3-4B-Base on the same
Deepscaler training data, with N=16 and the following four arms.

| Arm | N | K per attempt | Maximum samples per chain | Handling of rejected groups |
| --- | ---: | ---: | ---: | --- |
| Baseline K=8 | 16 | 8 | 8 | Ordinary baseline; no NGU accumulation |
| Baseline K=16 | 16 | 16 | 16 | Ordinary baseline; no NGU accumulation |
| Baseline K=32 | 16 | 32 | 32 | Ordinary baseline; no NGU accumulation |
| NGU K=8, max 32 | 16 | 8 | 32 | `never_give_up`, `ngu_max_budget=4` |

K is completions per prompt per attempt. The NGU budget includes the first
attempt: four attempts at K=8 allow at most 32 completions. N is the configured
rollout prompt count. NGU merges attempts, so a retained training collection can
contain fewer than 16 distinct groups after fitting its sample budget.

## Shared setup

- Model: Qwen3-4B-Base, starting from the same base checkpoint.
- Data: the existing `deepscaler-10k-base-01` training split and verifier config.
  Record immutable dataset/checkpoint provenance with each launch.
- Hardware for the new comparison: four Holmes GPUs, one trainer and three
  generators; record the actual GPU model and placement.
- Publication: refresh; fully asynchronous execution; max weight staleness 4;
  TIS correction with clip 2.0; zero-standard-deviation group filtering.
- Hold optimizer, sampling, sequence limits, precision and seed fixed across
  arms. Current recipe uses learning rate 1e-6, training temperature/top-p 1/1,
  response limit 8192 and context limit 10240.
- Keep the native sample budget N×K: 128/256/512 for the three baselines and
  128 retained samples for NGU K=8. Thus an optimizer update is not an equal
  amount of work across arms. NGU can also spend samples on failed chains that
  do not enter training. Report actual generated tokens and GPU time.
- Target 1,000 training collections; evaluate at startup and every 100;
  checkpoint every 25 and retain the latest three. Use a fresh output root and
  W&B group for each independent run; retries of one run share its group.
- Use the same scheduling policy across arms. Current launches use zero minimum
  runtime and auto-resume; scheduling gaps and checkpoint rollback affect the
  observed wall time.

## Primary comparison

Plot achieved **AIME 2025 mean correctness against elapsed wall-clock hours**.
Use the same evaluation prompts, verifier, 32 completions per prompt,
temperature 1.0, top-p 0.95 and response limit 8192 for every arm. Mean
correctness over those draws estimates single-sample accuracy; it is not pass@32.
Retain the exact W&B metric name and scale in exported data.

For each independent run:

1. Retrieve every W&B attempt in its group and its Beaker attempts. Verify the
   latest attempt separately for each task and replica; superseded preemptions
   are not terminal failures.
2. Deduplicate evaluation/training steps by the latest logged timestamp across
   checkpoint rollbacks. Use the evaluation's completion timestamp and the
   evaluated policy step/version, not the most recent training row.
3. Start the main clock at the first Beaker attempt's actual start. Include
   initialization, evaluation, restarts, repeated work and subsequent scheduling
   gaps. Also report submission-to-result time, including the initial queue.
4. Show a second plot against cumulative allocated GPU-hours, summing actual
   started-to-ended attempt durations × allocated GPUs. This includes failed
   attempts, setup, evaluation and work later rolled back. It is not a direct
   GPU-utilization measure or pure training time.
5. Show raw evaluation points and optionally best-so-far curves; label
   best-so-far explicitly. Compare at shared observed time budgets and report
   time to predeclared accuracy thresholds. Do not extrapolate beyond an arm's
   last evaluation or select a winner from a single noisy maximum.

Plot step and generated-token axes as diagnostics. Record confidence intervals
and replicate seeds before drawing a final conclusion; one seed is exploratory.
Historical 2-trainer/6-generator or barrier runs belong in separate panels from
the new four-GPU refresh comparison. Runtime, image and hardware changes also
prevent attributing every historical speed difference to NGU.

## Existing runs

Inventory checked on 2026-10-29. W&B project:
[`ai2-llm/open_instruct_internal`](https://forge.coreweave.com/wandb/ai2-llm/open_instruct_internal).
Run states below distinguish useful partial learning from successful completion.

### Hardware and publication matrix

Do not combine these rows into one hardware-matched timing comparison. GPU models
below come from the named attempts' downloaded W&B `wandb-metadata.json`; verify
every resumed attempt when exporting a complete history.

| Series | GPU model | Total GPUs | Trainers | Generators | Publication | Evidence |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Original N=16/K=32 baseline | NVIDIA B300 SXM6 AC | 8 | 4 | 4 | Barrier | `2zy3r5wj` hardware; group configs |
| Original N=16/K=8 NGU max 32 | NVIDIA B300 SXM6 AC | 8 | 4 | 4 | Barrier | `504ocdww` hardware; group configs |
| October 19 N=16/K=32 baseline | NVIDIA B300 SXM6 AC | 8 | 2 | 6 | Refresh | `1alfosor` hardware; group configs |
| October 19 N=16/K=8 NGU max 32 | NVIDIA B300 SXM6 AC | 8 | 2 | 6 | Refresh | `15g1vf6l` hardware; group configs |
| New N=16/K=8 NGU max 32 | Holmes requested; actual model pending placement | 4 | 1 | 3 | Refresh | [Beaker 01M64J5HXN9DXHF864S531HAT8](https://beaker.org/ex/01M64J5HXN9DXHF864S531HAT8); submitted, queued |
| New N=16/K=32 baseline | Holmes requested; actual model pending placement | 4 | 1 | 3 | Refresh | [Beaker 01M64J5V5PA5G2CTWCJ2Y9NYRD](https://beaker.org/ex/01M64J5V5PA5G2CTWCJ2Y9NYRD); submitted, awaiting placement |

The additional K=32 barrier metrics series uses eight GPUs split 2+6. Its actual
GPU model has not been independently checked in this inventory. The failed
October 27 four-GPU job used 1+3 and refresh but produced no training data.

Both October 29 submissions use immutable image `01M64J2YXFNKB92M1WKZTG597N`,
built on Hammond from Open Instruct `76ec0b7c492fcd692ea4e9a7ae0e72af675b3235`
and MILES `cdac4bf18ff7a81137a33e06f8415f19ad914583`, with the current main
binary base `01M5ZYQZVF7DDJ1VR619KRSS42`. The runtime's generated argument
snapshot matches the launcher. Validation: 281 config/launch tests, 137 rollout
runtime tests, and 27 refresh integration tests. CPU runtime tests disabled the
optional Megatron import because the CPU build host has no CUDA driver; GPU
startup/training is not yet verified. The rendered Holmes jobs have no explicit
`LD_LIBRARY_PATH` override.

### Learning evidence

| Arm / variant | Existing evidence | Use in the comparison |
| --- | --- | --- |
| Baseline K=8, N=16 | No matching plain baseline identified yet | Missing arm; needs a new four-GPU refresh run |
| Baseline K=16, N=16 | No matching plain baseline identified | Missing arm. The earlier `baseline_r01` and `deepscaler-miles_*` plain K=16 runs used **N=8**, not N=16; K=16 NGU runs are also a different arm |
| Baseline K=32, N=16, original, 4+4 GPUs, barrier | [Initial W&B 2zy3r5wj](https://forge.coreweave.com/wandb/ai2-llm/open_instruct_internal/runs/2zy3r5wj); [finished attempt p0fd6l05](https://forge.coreweave.com/wandb/ai2-llm/open_instruct_internal/runs/p0fd6l05); group `qwen3-4b-base-deepscaler_baseline_k32n16` | Four W&B attempts; completed through step 999, final logged AIME correctness 21.35%. Historical eight-GPU barrier reference |
| Baseline K=32, N=16, 2+6 GPUs, barrier | [Beaker 01M50BF6EZWDPEGFCENVXVC53X](https://beaker.org/ex/01M50BF6EZWDPEGFCENVXVC53X); group `qwen3-4b-base-deepscaler_baseline_k32n16_t2g6_metrics` | Historical fixed-K baseline; barrier and GPU allocation differ from the new series |
| Baseline K=32, N=16, 2+6 GPUs, refresh | [Beaker 01M5AQZ17MQKNP51G3SVQAE8NT](https://beaker.org/ex/01M5AQZ17MQKNP51G3SVQAE8NT); [W&B 1alfosor](https://forge.coreweave.com/wandb/ai2-llm/open_instruct_internal/runs/1alfosor); group `qwen3-4b-base-deepscaler_baseline_refresh_k32n16_t2g6_20261019` | 52 W&B attempts; usable partial learning, evaluations through step 700 (22.81% in that attempt's summary). Latest Beaker attempt failed during startup, before executing training |
| NGU K=8, max 32, N=16, original, 4+4 GPUs, barrier | [Initial W&B ahhqdtrr](https://forge.coreweave.com/wandb/ai2-llm/open_instruct_internal/runs/ahhqdtrr); [finished attempt 504ocdww](https://forge.coreweave.com/wandb/ai2-llm/open_instruct_internal/runs/504ocdww); group `qwen3-4b-base-deepscaler_ngu_budget4_k8n16` | Two W&B attempts; completed through step 999, final logged AIME correctness 20.42%. Historical eight-GPU barrier reference |
| NGU K=8, max 32, N=16, 2+6 GPUs, refresh | [Beaker 01M5AT1ZWBT5TGB8SP02RTGMXG](https://beaker.org/ex/01M5AT1ZWBT5TGB8SP02RTGMXG); [finished W&B 15g1vf6l](https://forge.coreweave.com/wandb/ai2-llm/open_instruct_internal/runs/15g1vf6l); group `qwen3-4b-base-deepscaler_ngu_refresh_budget4_k8n16_t2g6_20261019` | 12 W&B attempts; Beaker completed successfully, evaluations through step 999 (19.48%). Closest existing NGU arm, but eight GPUs and older runtime |
| NGU K=8, max 32, N=16, 1+3 GPUs, Holmes | [Failed attempt 01M5ZF84PHZFJP6Q6FP8K4FS50](https://beaker.org/ex/01M5ZF84PHZFJP6Q6FP8K4FS50) | CUDA Error 803 before training; no learning evidence. Replacement submitted October 29; see hardware matrix |

The initial NGU refresh submission
[01M5AQZ4X838QW92GEE0JN9TAG](https://beaker.org/ex/01M5AQZ4X838QW92GEE0JN9TAG)
preceded the successful rerun; retrieve the whole shared W&B group when forming
its history. K=16 NGU max-48/max-64 runs and later CUDA startup failures are
outside the requested four arms.

The saved October 19 comparison is preliminary: at step 499 it recorded 21.46%
for baseline K=32 and 19.17% for NGU K=8; NGU reached that evaluation earlier.
That is not a matched-time or replicated conclusion. Re-export all attempts
before using those numbers in a final plot.

The live inventory inspected 150 matching W&B runs/attempts, checking configured
`rollout_batch_size`, `n_samples_per_prompt`, handler, budget and GPU allocation,
rather than classifying runs from their names alone. Summary maxima establish
available evaluation coverage; they do not replace timestamp-deduplicated full
history. In particular, the saved step-499 baseline analysis is older than the
live step-700 coverage above.

## Next experiments

1. Verify GPU startup and learning for the submitted four-GPU Holmes NGU
   K=8/max-32 and baseline K=32 refresh pair.
2. Run plain baselines K=8 and K=16 with the same four-GPU refresh setup and
   source/runtime pins. These two launches are planned, not yet authorized by
   writing this document.
3. Export evaluation points, timestamps, generated-token counts and attempt
   lifetimes into one comparison dataset. Produce wall-time and GPU-hour plots.
4. Add group-size diagnostics: trained sizes 8/16/24/32 and a separate max-32
   failed-chain category. Training sizes follow pruning/trimming; producer
   failure counts use observation windows and should not be equated with unique
   trained prompts.
