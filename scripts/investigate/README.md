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

### Interactive terminal convention

The gradient investigation's interactive terminal has two screens:

- **All-token screen** — shows the generated assistant answer as numbered
  tokens with their probabilities. Selecting a position opens its candidate
  screen. This screen also owns answer regeneration, initial-question editing,
  and the choice of how many candidates to display for a position. Commands
  include `n`, `n k count`, and `n p mass` for selection; `r` or `r s` to
  regenerate greedily or stochastically; and `e` to replace the initial
  question and regenerate.
- **Candidate screen** — shows the top candidate tokens and probabilities for
  the selected answer position. Candidate inspection, gradient plots,
  continuation comparisons, and replacing the current token belong to this
  screen. Use `n g` to graph the image and text shares per layer, computed by
  taking the square root of each modality's summed squared token gradients and
  normalizing those two values to 100%. Inspecting a candidate keeps the user
  at the same position; replacing
  it regenerates the rest of the answer and returns to the all-token screen;
  `b` returns without replacing it.
- Entering `q` from either screen exits the investigation program after hook
  cleanup; it does not just leave the current screen or continue with another
  sample. The timeout counts idle time at each terminal prompt and resets when
  the next prompt is shown. If the terminal receives no input for the configured
  inactivity timeout, the program exits cleanly as well. Set
  `interactive_timeout_minutes` in `configs/investigate.yaml` (default: `10`);
  use `--investigate-config PATH` to load a different investigation config.

Plots and large-image views are visualizations opened from the candidate
screen, not additional terminal screens. Keep the terminal flow organized
around these two screens as their features grow.

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

## Resource and memory conventions

For new investigations that accumulate large CPU-side measurements, prefer a
run that finishes using disk space over one that exhausts system RAM. Provide a
disk-offload option and use a configurable RAM reserve threshold. Interpret a
10% threshold as **less than 10% of system RAM available** (about 90% used),
not 10% utilization. Check available memory before retaining the next large
detached result, so offload begins before a large allocation pushes the
process over the limit.

- Use `--offload-to-disk` to enable threshold-based spilling,
  `--ram-reserve-percent` to set the available-RAM threshold (default `10`),
  and `--offload-dir` to select the destination. Scripts that normally retain
  large captures should enable offload by default and offer an explicit opt-out;
  small scripts may leave it opt-in.
- Offload completed, detached measurements or sample chunks; do not move a
  live autograd graph to disk. Keep the current sample's tensors in memory only
  as long as the computation needs them, then write the reduced results and
  release their references.
- Make the reserve percentage and offload directory configurable through the
  script config and/or CLI. Use predictable per-script output paths and record
  enough metadata to recover tensor shapes, dtypes, layers, and sample IDs.
- When no offload directory is specified, use a unique run-specific temporary
  directory and remove it during normal cleanup after analysis. Preserve a
  user-specified `--offload-dir`; never recursively delete a shared or
  pre-existing directory. Keep final graphs, CSVs, and other requested outputs
  outside the temporary offload directory.
- Write bounded chunks instead of collecting the full dataset and spilling it
  only at the end. If analysis needs the spilled data, read it in chunks or use
  memory mapping rather than loading the complete store back into RAM.
- Check that the destination is writable and has space before starting a large
  run. If offload cannot proceed, give a clear error with the path and a
  practical alternative such as reducing samples or selecting fewer layers;
  never silently keep accumulating CPU tensors until the process is killed.
- Log when offloading starts and how many samples/chunks have been written.
  Extensive logging may include available and total RAM; normal logging should
  stay concise. `--quiet-warnings` must not hide offload errors.

Offload is for retained analysis data, not a substitute for reducing a live
model forward or gradient graph that exceeds available memory. For those cases,
use smaller batches or process one sample at a time. Existing investigation
scripts do not need to be changed just to adopt this convention; apply it when
adding or substantially extending a script that retains enough data to risk
RAM exhaustion.

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
- `investigate_overlap.py`: for each selected COCO ShareGPT4V pair, first
  greedily generates an assistant answer with no gradients or attention-map
  retention, then frees that generation cache before a gradient-enabled,
  teacher-forced pass. It compares per-image-token IGA (attention averaged over
  generated assistant queries and heads) against the root-sum-square gradient
  of the mean generated-answer CE with respect to each selected layer's value
  projection. It reports the intersection count between each metric's top
  `--top-percent` image tokens (ceil of the requested fraction), and saves a
  per-sample/layer CSV plus a graph of mean overlap count.
- `investigate_grad_important.py`: computes assistant answer-token CE one token
  at a time, backpropagates each token loss to a selected activation at every
  decoder layer, sums squared gradients over hidden dimensions and answer tokens for each image
  and valid non-image text token, and displays separate per-layer figures for
  image-patch maps and text-token scores with decoded labels. Each modality has
  its own colorbar and scale, so image and text colors show relative levels
  within that modality and their raw colors must not be compared across the two.
  Use `--heatmap-only` to show the source image in a separate reference panel
  beside the plain patch-grid heatmaps. The default color scale is logarithmic;
  use `--scale linear` for a normal linear scale. Use
  `--score-normalization percent` to normalize image and text tokens separately
  within each layer, so each modality's scores sum to 100% in that layer. It
  displays by default and saves nothing unless `--save-dir` is
  supplied. By default it scores the dataset's assistant answer. Use
  `--answer-mode generate` to first greedily generate a reply from the user
  prompt, then score that generated reply; the sentence is printed in both
  normal and extensive logging. Generation stops at the model's EOS token or
  the `--max-new-tokens` cap (default 128); there is no custom semantic stopping
  criterion. Use `--answer-mode interactive` for the two-screen token browser.
  On the all-token screen, enter `n` for the default top 10 candidates at
  position n, `n k count` for top-k, or `n p mass` for candidates covering a
  cumulative probability mass such as `0.9` or `90%`. Enter `r` to regenerate
  greedily, `r s` to regenerate with stochastic nucleus sampling, or `e` to
  replace the initial question and choose greedy or stochastic generation.
  The display command `n p mass` uses the untempered model probabilities;
  `--top-p` controls sampling during `r s` or stochastic question regeneration.
  Stochastic regeneration uses `--temperature` (default 0.8) and `--top-p`
  (default 0.9). On the candidate screen, n is the candidate rank: enter `n`
  or `n i` for image scores, `n t` for all prior text-token scores, or `n t c`
  for the latest c prior text tokens. Enter `n + c` to print the next c greedy
  tokens after candidate n is selected, alongside the original token and its
  original continuation. This branch view prints text only and does not plot
  gradients. Enter `n replace` to use candidate n as the current token and
  greedily regenerate the rest of the answer up to the original total token
  cap. Add a positive count, `n replace c`, to cap the newly generated suffix
  at c tokens. This returns to the all-token screen with the updated answer.
  The other candidate commands plot the selected candidate in a separate image
  or text figure. After an inspection, the interface stays at the same
  position; use `h` for candidate-screen help, `b` to return to the
  token list, `i` to open the input image in a large window, or `q` to finish.
  `h` on the all-token screen prints its command summary. These gradient maps
  are diagnostic clues, not a standalone
  hallucination test.
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

Compare the top 10% of IGA and value-cache gradient scores on 128 greedy
generated COCO answers, for decoder layers 22–25:

```bash
python scripts/investigate/investigate_overlap.py \
  --samples 128 \
  --layers 22 23 24 25 \
  --top-percent 10 \
  --log-level extensive \
  --quiet-warnings
```

The graph and companion CSV default to `data/processed/investigate_overlap.png`
and `.csv`. Use `--output PATH` to change the destination, and `--no-display`
to save the graph without opening a window. Greedy answers are generated one at
a time before the scoring forward to keep the generation KV cache out of the
gradient pass.

Display only the gradient heatmaps for two samples (no files are written):

```bash
python scripts/investigate/investigate_grad_important.py \
  --samples 2 \
  --heatmap-only \
  --scale linear \
  --score-normalization percent
```

Differentiate with respect to each layer's post-normalization activation instead
of its pre-normalization block input:

```bash
python scripts/investigate/investigate_grad_important.py \
  --samples 2 \
  --capture-point after_layer_norm
```

Generate the assistant answer instead of scoring the dataset example:

```bash
python scripts/investigate/investigate_grad_important.py \
  --samples 2 \
  --answer-mode generate \
  --max-new-tokens 128
```

Generate a complete answer, then inspect alternatives at selected token
positions and optionally view a candidate's image gradient map:

```bash
python scripts/investigate/investigate_grad_important.py \
  --samples 1 \
  --answer-mode interactive
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
per-layer score for an image or valid non-image text token is the sum of squared
gradient values over hidden dimensions and all assistant answer tokens. The
figure gives image and text their own color scales because their score magnitudes
can differ; percent mode normalizes each type within its own layer. Since each answer token
requires a separate gradient calculation, this analysis can be slow for long
answers and retains the forward graph until all answer-token gradients are
computed.

All scripts should be runnable from the repository root, derive the root from
`Path(__file__)` (two parent levels from this directory), and keep analysis code
here rather than adding more top-level entries under `scripts/`.
