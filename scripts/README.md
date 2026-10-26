# Script conventions

These conventions apply to scripts throughout this directory. Investigation-
specific workflow guidance lives in [`investigate/README.md`](investigate/README.md).

## Readability and state

- **Document function contracts.** Describe inputs, outputs, assumptions, and
  state a function reads or changes. Include tensor shapes and meanings where
  they help readers understand the operation.
- **Document hooks.** Explain which module input or output a hook reads, what it
  captures, how it transforms or reduces that data, and whether it changes the
  model's output.
- **Describe shared dictionary state.** At initialization, document its keys,
  value types, and whether values are per-sample or accumulated. Update that
  description when the schema changes. Add shapes and device/dtype details for
  tensor values when relevant.
- **Make data flow traceable.** Keep preprocessing, transformations, model
  execution, validation, and result storage easy to follow. Avoid unnecessary
  intermediate state and comments that only restate the code.
- **Clarify side effects and cleanup.** Identify code that registers hooks,
  mutates shared state, writes files, or displays plots. Keep cleanup visible,
  especially hook removal in `finally` blocks.

## Shared configuration

Use the shared configs by default, with CLI paths available to override them:

- `configs/model.yaml`: model ID and local directory, model family, QIG source,
  dtype, device map, and attention implementation. Common keys are `model_id`,
  `model_dir`, `model_family`, `implementation_source_dir`, `torch_dtype`,
  `device_map`, and `attn_implementation`.
- `configs/dataset.yaml`: data/cache locations and the standard calibration
  subset. Common keys are `data_root`, `raw_dir`, `cache_dir`, `processed_dir`,
  `calibration_samples`, `calibration_seed`, `calibration_source`, and
  `calibration_output_dir`.

Treat YAML paths as repository-relative unless already absolute. Use the shared
config/path helpers from `attention_quantization.config` and
`attention_quantization.data` rather than duplicating path or dataset logic.
CLI values take precedence over YAML. Options such as `--samples` and `--seed`
should override their corresponding config values; otherwise use the configured
value and document any script-specific fallback.

## Tensor device safety

Do not assume masks returned by a multimodal adapter share a device. In
particular, `attention_mask`, `vision_mask`, and `labels` may be split between
CPU and GPU. Before combining masks with boolean operations, detach and move
them to one common device (CPU is convenient for bookkeeping). Before using a
mask to index activations or attention weights, move that mask to the indexed
tensor's device. Keep the original adapter tensors on their expected devices
when passing inputs to the model. This avoids errors such as `Expected all
tensors to be on the same device`.

## Timing expensive operations

Before implementing a script, identify operations likely to dominate runtime or
resource use: dataset/model downloads and loading, preprocessing, model forwards,
large tensor computations, CPU/GPU transfers, disk reads/writes, and serialization.
Measure these stages separately with a monotonic clock and log their elapsed time
at `extensive` verbosity. For long-running work, include periodic progress at
`normal` verbosity so a run does not appear stalled.

Include enough context to guide the next compute decision: item or layer range,
shape or byte count, device or storage tier, completed/total work, and effective
throughput when meaningful. Keep timings for unlike stages separate (for example,
GPU-to-host copy versus disk flush) instead of reporting one combined duration.
Synchronize CUDA around wall-clock measurements of asynchronous GPU work, or use
CUDA events, so reported times include the operation being measured. Avoid adding
extra synchronization to normal execution unless timing is enabled or the
synchronization is required for correct progress reporting.

For expensive third-party calls, do not report only one opaque total. Instrument
the distinct phases at the narrowest stable boundaries available (for example,
tokenizer/config loading, checkpoint tensor loading, device dispatch, and adapter
initialization). If the dependency exposes no suitable boundaries, log a useful
input inventory before the call (file count and bytes, shard sizes, source and
destination filesystem free space, device limits) and periodic elapsed-time
heartbeats while it runs. Log resource and output sizes afterward. Keep
diagnostic scans lightweight and time them separately; avoid repeatedly
recursing through large directories just to produce heartbeats.

When `--log-level extensive` is enabled, finish the run with a timing summary
for each measured process or stage. Report total elapsed time and, for repeated
work, the operation count and useful aggregate statistics such as average,
minimum, and maximum duration. Keep separate stages separate in the summary
(for example, model forward, GPU-to-host transfer, and disk write/flush) so the
results show where runtime went. Omit stages that did not run, and report a
clear partial-run summary on failure or interruption when practical.

## Common command-line options

Use consistent options when they apply; a script does not need to expose an
option that is irrelevant to its task:

- `--model-config` and `--dataset-config` default to the repository's shared
  YAML files. CLI values should override config values.
- `--samples` and `--seed` select a reproducible subset when sampling applies.
- `--device` is appropriate when the script supports choosing a device.
- Use an output option such as `--output-dir` or `--save-dir` when writing
  results. Keep defaults predictable and report the resolved output path.
- `--log-level` uses `none`, `normal`, and `extensive`. `normal` is the default
  and reports concise progress; `extensive` adds resolved settings, shapes,
  capture counts, metrics, and timings; `none` suppresses routine progress.
- `--quiet-warnings` may silence noisy dependency warnings, but must not hide
  errors or exceptions.
- `--timing` shows timing while keeping normal verbosity. Extensive logging
  enables timing automatically; `none` suppresses timing too.

Add task-specific options only for real user choices. Give each a useful help
string, validate numeric ranges before expensive work, and avoid hard-coded
sample counts, model paths, or output locations that obscure a run's purpose.
