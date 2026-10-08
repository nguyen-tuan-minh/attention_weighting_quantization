# Investigation script conventions

This directory contains exploratory and diagnostic scripts for the LLaVA/QIG
workflow. New investigation scripts should follow the common structure below so
they are easy to run, compare, and maintain.

## The five phases

Keep `main()` organized into these five visible phases, using numbered section
comments where the workflow is long:

1. **Read options and configuration.** Parse CLI arguments, load the model and
   dataset YAML files, apply CLI overrides, resolve repository-relative paths,
   and validate values before loading large resources.
2. **Select and prepare data.** Load the intended dataset split, select a
   reproducible sample subset, and use locally available images where possible.
   Print the selected count and seed. Keep preprocessing consistent across
   models being compared.
3. **Load model(s).** Load the base LLaVA model using the configured QIG
   implementation. If the investigation concerns quantization, also load the
   requested quantized model. The base model is the reference for hidden-state
   error, logits, or other comparisons; the quantized model is the model under
   investigation. Do not require a quantized checkpoint for an analysis that
   only examines the base model.
4. **Capture and compute.** Attach only the hooks needed for the investigation,
   run inference with gradients disabled, and move retained measurements to CPU
   promptly. Remove hooks after the forward pass and release model/batch memory
   when practical. Keep sample and token masks explicit, especially assistant
   answer masks and image-token masks.
5. **Analyze and save results.** Compute summary statistics after data capture.
   Write machine-readable results (CSV/JSON or tensor files) and figures to a
   documented output directory. Print a concise progress summary and the output
   path.

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
- `--timing` when the script has expensive setup or per-sample work.
- `--quantized-model` when a quantized checkpoint is an actual input. Require
  it for comparisons that cannot run without it, validate its metadata before
  model loading, and use `--base-model` as an optional reference override when
  the base path normally comes from `model.yaml`.

Add task-specific options only for real analysis choices (for example, batch
size, layer/token selection, plot scope, or the metric being correlated). Give
each option a useful help string and validate numeric ranges before starting
inference. Avoid hiding the purpose of a run behind hard-coded sample counts,
model paths, or output locations.

## Quantized-model comparisons

For a base-versus-quantized analysis, make the roles and metric direction
explicit in the CLI help, output metadata, and README/example command. Use the
same inputs, labels, token masks, and sample order for both models. Record both
model identities and the metric definition in the summary output. For example,
`iga_error_correlation.py` supports a required quantized checkpoint and a base
reference, then lets the user choose `KL(base || quantized)` or the CE change
`quantized CE - base CE` with `--correlation-target`.

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
