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

Phase 4 normally runs inference without gradients. A gradient investigation can
enable gradients deliberately; document which activations require gradients and
which loss is differentiated so this exception is clear.

Read CLI arguments and configs before phase 1, and validate them before loading
large resources. These setup steps do not replace or renumber the five phases.

## Readability conventions

Organize functions, tensor transformations, and shared state so readers can
follow each phase and trace how inputs become saved results:

- **Group phase functions together.** Keep each phase's related function
  definitions in one area of the file, under a clear, prominent numbered
  comment such as `# N. PHASE NAME`. Make it easy to find a phase's
  implementation without searching through unrelated helpers.
- **Document function contracts.** Give functions clear docstrings describing
  their inputs, outputs, assumptions, and any state they read or change. State
  tensor shapes and meanings where they matter.
- **Document each hook's contract when hooks are used.** Explain which module
  output it reads, what it captures, how it reduces or transforms the data, and
  what it returns to the model.
- **Document dictionary state at initialization and updates.** For each
  dictionary, describe its keys and value types when it is initialized. Use the
  same format when its contents are reset or updated: `variable_name: type,
  shape if tensor, description`. State whether values are per-sample or
  accumulated across samples.
- **Keep phase flows explicit.** Show the ordered steps within each phase. For
  forward phases, make preprocessing, mask creation, model forward, capture
  validation, and result storage easy to distinguish, including which work
  happens once per sample.
- **Use one variable comment format.** At phase handoffs, tensor/dictionary
  initialization, and tensor/dictionary reassignment, comment each variable on
  its own line as `variable_name: type, shape if tensor, description`. For
  example: `attention_mask: torch.Tensor, CPU bool [batch, sequence], valid
  token positions`. Include the shape and dimension meanings for tensors, and
  the key/value schema for dictionaries. Update the comment whenever the
  variable's type, shape, device, dtype, or meaning changes.
- **Clarify side effects and cleanup.** Identify functions that register hooks,
  mutate shared state, write files, or display plots. Keep cleanup visible,
  especially hook removal in `finally` blocks.

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

## Tensor device safety

Do not assume masks returned by the multimodal adapter share a device. In
particular, `attention_mask`, `vision_mask`, and `labels` may be split between
CPU and GPU. Before combining masks with boolean operations, detach and move
them to one common device (CPU is convenient for bookkeeping). Before using a
mask to index activations or attention weights, move that mask to the indexed
tensor's device. Keep the original adapter tensors on their expected devices
when passing inputs to the model. This avoids errors such as `Expected all
tensors to be on the same device`.

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
  or save attention maps. Use `--true-attention-percent` to label image-key
  softmax weights as percentages of attention over all valid keys. Use
  `--top-iga-percent 0.1` for a binary map retaining the top 10% of image
  tokens by IGA score independently in each layer; this mode uses nearest-neighbor resizing
  and a discrete colorbar instead of the logarithmic attention scale.
- `investigate_activation_pca.py`: captures layer activations and plots
  image-token versus text-token PCA.
- `investigate_assistant_attention.py`: measures assistant answer-token attention
  to image and non-image text keys, then saves per-layer percentages as a
  stacked graph and CSV.
- `investigate_grad_important.py`: computes assistant answer-token CE one token
  at a time, backpropagates each token loss to a selected activation at every
  decoder layer, sums squared gradients over hidden dimensions and answer tokens for each image
  token, and displays per-layer image-token heatmaps for each sample, overlaid
  on the source image by default. Use `--heatmap-only` to show the source image
  in a separate reference panel beside the plain patch-grid heatmaps. The
  default color scale is logarithmic; use `--scale linear` for a normal linear
  scale. It displays by default and saves nothing unless `--save-dir` is
  supplied.
- `iga_error_correlation.py`: compares base and quantized layer errors and
  correlates them with an answer-level divergence/loss metric.

For `investigate_assistant_attention.py`, assistant query positions come from
the non-ignored answer labels. Image keys come from `vision_mask`; text keys are
all valid non-image positions. The graph reports each group's share of the
combined image-plus-text attention mass, averaged over heads, answer queries,
and samples. The CSV also records the number of head/query rows contributing to
each layer.

Example:

```bash
python scripts/investigate/investigate_assistant_attention.py \
  --samples 128 \
  --log-level extensive \
  --quiet-warnings
```

Display only the gradient heatmaps for two samples (no files are written):

```bash
python scripts/investigate/investigate_grad_important.py \
  --samples 2 \
  --heatmap-only \
  --scale linear
```

Differentiate with respect to each layer's post-normalization activation instead
of its pre-normalization block input:

```bash
python scripts/investigate/investigate_grad_important.py \
  --samples 2 \
  --capture-point after_layer_norm
```

Save the same per-sample figures without opening windows:

```bash
python scripts/investigate/investigate_grad_important.py \
  --samples 2 \
  --save-dir outputs/grad_important \
  --no-display
```

For each assistant answer token, this script computes that token's causal
cross-entropy from the preceding-position logits and obtains gradients with
respect to the [batch, sequence, hidden] input of each decoder block. The
per-layer score for an image token is the sum of squared gradient values over
hidden dimensions and all assistant answer tokens. Since each answer token
requires a separate gradient calculation, this analysis can be slow for long
answers and retains the forward graph until all answer-token gradients are
computed.

All scripts should be runnable from the repository root, derive the root from
`Path(__file__)` (two parent levels from this directory), and keep analysis code
here rather than adding more top-level entries under `scripts/`.
