# Investigation script conventions

This directory contains exploratory and diagnostic scripts for the LLaVA/QIG
workflow. New investigation scripts should follow the common structure below so
they are easy to run, compare, and maintain.

## The five phases

Use these numbered section headings in `main()` so the workflow is easy to scan
and familiar across investigation scripts:

1. **PREPARE CALIBRATION DATASET** — load the intended data source, select the
   requested reproducible subset, check that required local images exist, and
   report the sample count and seed. For quantized comparisons, prepare one
   shared set of inputs for both models.
2. **LOAD MODEL** — load the configured base LLaVA model. When the analysis
   compares quantization, also load the quantized checkpoint and make its role
   explicit. The base model is the reference; the quantized model is the model
   under investigation. Do not require a quantized checkpoint for base-only
   analysis.
3. **REGISTER HOOKS** — register only the hooks needed to capture the
   investigation's measurements (for example, attention weights or activations)
   and report how many hooks were attached. Remove hooks in a `finally` block.
4. **FORWARD CALIBRATION SAMPLES** — preprocess and run the selected examples
   with gradients disabled. Use consistent inputs, sample order, labels, and
   masks across compared models. Move retained measurements to CPU promptly
   and release per-sample/model memory when practical.
5. **ANALYSE CAPTURED DATA** — compute the investigation-specific statistics,
   plots, or comparisons; save outputs in documented formats and locations; and
   print a concise completion message and output path. The existing attention
   script calls this phase `ANALYSE CAPTURED ATTENTION`, and the PCA script calls
   it `ANALYSE ACTIVATIONS`; use a specific noun where it improves clarity.

Read CLI arguments and configs before phase 1, and validate them before loading
large resources. These setup steps do not replace or renumber the five phases.

## Configuration files

Use these shared configs by default, with CLI paths available to override them:

- `configs/model.yaml`: model ID and local directory, model family, QIG source,
  dtype, device map, and attention implementation. Common keys are
  `model_id`, `model_dir`, `model_family`, `implementation_source_dir`,
  `torch_dtype`, `device_map`, and `attn_implementation`.
- `configs/dataset.yaml`: data/cache locations and the standard calibration
  subset. Common keys are `data_root`, `raw_dir`, `cache_dir`, `processed_dir`,
  `calibration_samples`, `calibration_seed`, `calibration_source`, and
  `calibration_output_dir`.

Treat paths in YAML as repository-relative unless the path is already absolute.
Use the shared config/path helpers from `attention_quantization.config` and
`attention_quantization.data` instead of duplicating path or dataset logic.
`--samples` and `--seed` should normally override the corresponding dataset
config values. A CLI value takes precedence over YAML; otherwise use the
configured value and document any script-specific fallback.

## Common command-line options

Unless an investigation has a clear reason to differ, provide:

- `--model-config` and `--dataset-config`, defaulting to the repository's
  `configs/model.yaml` and `configs/dataset.yaml`.
- `--samples` and `--seed` to control the subset and make runs repeatable.
- `--device` or the configured device setting when device selection is needed.
- An output path option such as `--output-dir` or `--save-dir` when results are
  written. Default outputs should be predictable and reported at completion.
- `--log-level` with three levels: `none` suppresses routine progress and
  timing output, `normal` prints concise phase/sample milestones, and
  `extensive` prints resolved model/data settings, input and mask shapes, hook
  and capture counts, per-sample metrics, and timing output. Use `normal` as the
  default.
- `--quiet-warnings` to silence warning messages from Python and dependencies
  when they overwhelm useful output. It must not suppress errors or exceptions.
- `--timing` to show timing lines while keeping normal verbosity. It is enabled
  automatically by `--log-level extensive`; `--log-level none` suppresses the
  timing lines as well.
- `--quantized-model` when a quantized checkpoint is an actual input. Require
  it for comparisons that cannot run without it, validate its metadata before
  model loading, and use `--base-model` as an optional reference override when
  the base path normally comes from `model.yaml`.

Add task-specific options only for real analysis choices (for example, batch
size, layer/token selection, plot scope, or the metric being correlated). Give
each option a useful help string and validate numeric ranges before starting
inference. Avoid hiding the purpose of a run behind hard-coded sample counts,
model paths, or output locations.

## Current scripts

- `investigate_attention.py`: captures attention/IGA information and can display
  or save attention maps.
- `investigate_activation_pca.py`: captures layer activations and plots
  image-token versus text-token PCA.
- `iga_error_correlation.py`: compares base and quantized layer errors and
  correlates them with an answer-level divergence/loss metric.

All scripts should be runnable from the repository root, derive the root from
`Path(__file__)` (two parent levels from this directory), and keep analysis code
here rather than adding more top-level entries under `scripts/`.
