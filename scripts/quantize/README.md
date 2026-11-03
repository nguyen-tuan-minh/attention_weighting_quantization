# Quantization scripts

## Disk-backed GPTQ for LLaVA 1.5

The scripts support both LLaVA 1.5 7B and 13B checkpoints. Select
`--model-config configs/model-llava-v1.5-13b.yaml` to use the 13B model in a
quantization run; the same config option is available in evaluation and
investigation scripts. The LLaVA loader resolves the correct upstream model
identifier from the checkpoint config/path, including for renamed local
checkpoint directories.

### Layer selection convention

Quantization scripts should allow targeting a subset of decoder layers while
defaulting to all layers. Layer numbers exposed to users are 1-based and should
accept comma-separated individual numbers and inclusive ranges. For example,
`--layers 1-4,8` selects layers 1, 2, 3, 4, and 8. The script validates the
selection against the model's decoder layer count. Subset runs compute Hessians
and quantize only selected layers, while unselected layers keep their base
weights in the standalone output. Record the selected layer numbers in output
metadata. GPTQ implements this as `--layers`; omitting the option quantizes
every decoder layer.

`quantize_gptq.py` is a standalone implementation of layerwise GPTQ for the
configured LLaVA 1.5 checkpoint. It does not call QIG's GPTQ quantizer and does
not modify LMMS-Eval. Accelerate handles CPU and disk weight offload: those are
storage locations, and offloaded decoder weights are staged onto GPU 0 for
forward execution. The script logs the storage map separately from the
execution device and reports the size of the disk-offload cache. `--max-model-memory-gpu`
and `--max-model-memory-cpu` constrain automatic placement; when weights do not fit
within those budgets, Accelerate can spill them to the disk cache in
`--offload-dir` (or the output directory's temporary work folder). Disk access
can make forwards substantially slower than GPU-resident weights. The script
accumulates exact float32 Hessians on GPU in group passes. For each group of
selected decoder layers, it loads that group's Hessians once, forwards all
calibration samples while keeping the group on GPU, then saves the group once.
It repeats model forwards for each group. With all 32 layers and the default
`--hessian-group-layers 7`, this is 5 passes over the samples (640 forwards for
128 samples), with a final partial group. This reduces Hessian disk writes from
once per sample to once per group, at the cost of rereading offloaded model
weights during repeated forwards. `--hessian-group-layers` controls the number
of adjacent decoder layers in a group; larger groups reduce the number of
passes and Hessian saves, but need more VRAM for active Hessians and forward
workspace. Extensive logs report planned forwards, per-group progress, and
load/save/forward timings. GPU and CPU placement limits
default to unset, so Accelerate chooses placement automatically. If you cap
only one tier, the script fills the other tier with Accelerate's detected
available capacity, so a GPU-only cap still allows CPU placement before disk
offload. Set a tier to `0GB` to prohibit placement there. If you set
`--max-model-memory-gpu`, leave enough VRAM beyond the resident weights for the active
Hessian group and forward workspace; reduce `--hessian-group-layers` if GPU
allocation fails.
All model and Hessian memory limits accept `KB`, `MB`, `GB`, `TB` and binary
`KiB`, `MiB`, `GiB`, `TiB` units. A bare number means decimal `GB`, so `10`
equals `10GB`. `--max-hessian-memory-gpu` caps the active GPU accumulation group. Completed
Hessians are placed in RAM up to `--max-hessian-memory-cpu`, with the remainder
stored on disk up to `--max-hessian-memory-disk`. Each limit defaults to unset
(no explicit cap); disk free space is checked before allocation. Quantized
outputs are not used to collect Hessians.
`--hessian-token-scope all` (the default) uses every valid image and text token
position. Set it to `text` to exclude image positions; this uses the adapter's
vision mask and attention mask, so padding is excluded as well.

`--hessian-weighting uniform` (the default) gives every included token equal
weight. Set `--hessian-weighting iga` to use the repository's IGA definition:
for each layer, average post-softmax attention over heads and valid text queries
after the image, producing one score per image token. Normalize image-token
scores to mean 1, use them as weights for image-token activation rows, and keep
valid text-token rows at weight 1 when accumulating `X'WX`. This requires
`--hessian-token-scope all`; IGA cannot weight image-token rows when the text-only
scope excludes them. The script requests attention weights for the calibration
forward, reduces each layer's matrix immediately, and drops the full matrix
instead of retaining matrices from all layers. This adds IGA computation and
temporary attention memory. The selected weighting is recorded in
`gptq_metadata.json`. To enable IGA, pass
`--hessian-token-scope all --hessian-weighting iga`.

Set `--hessian-weighting grad` to weight each included token by its gradient
importance. For each calibration example, the script takes assistant-answer
cross-entropy gradients with respect to each selected decoder block's input,
sums squared gradients across hidden features and answer-token targets, then
takes the square root to obtain the token score. Scores are normalized to mean
1 over the selected token scope before weighting that layer's `X'WX` Hessians.
This follows the investigation's `block_input` gradient capture convention and
works with either `--hessian-token-scope all` or `text`. It does not request or
retain attention matrices. Gradient weighting runs in two phases: grouped
gradient-enabled forwards compute and store only per-token weights, then a
second set of grouped inference forwards recomputes Linear activations and
accumulates the weighted Hessians on GPU. This avoids staging every selected
Linear activation in CPU RAM, at the cost of replaying the calibration
forwards. Gradient scoring computes one gradient per assistant answer token,
so it can use substantially more memory and compute than uniform or IGA
weighting. Logs report the gradient-weight collection and Hessian accumulation
phases separately. The weighting mode is recorded in `gptq_metadata.json`.

Example using gradient weights:

```bash
.venv/bin/python scripts/quantize/quantize_gptq.py \
  --output-dir outputs/quantized/llava-gptq-w4g128-grad \
  --samples 128 \
  --seed 42 \
  --bits 4 \
  --group-size 128 \
  --hessian-token-scope text \
  --hessian-weighting grad \
  --hessian-group-layers 4 \
  --log-level extensive
```

Disk-backed Hessians share one contiguous `hessians.f32` file. When an active
layer group is entirely disk-backed, the script reads and writes that group's
Hessians as one contiguous chunk rather than issuing separate transfers for
each Linear matrix. Mixed RAM/disk groups use per-matrix transfers. Disk flush
latency can vary by backing filesystem; group save timing separates GPU-to-host
copy, assignment into the memory-mapped file, and flushing that mapping.

Example for a 12 GB RAM / 15 GB GPU machine (keeps Hessian backing off CPU and
uses four-layer GPU groups to leave room for model weights and forward workspace):

```bash
.venv/bin/python scripts/quantize/quantize_gptq.py \
  --output-dir outputs/quantized/llava-gptq-w4g128 \
  --samples 128 \
  --seed 42 \
  --bits 4 \
  --group-size 128 \
  --hessian-token-scope text \
  --hessian-group-layers 4 \
  --max-hessian-memory-gpu 4GB \
  --max-hessian-memory-cpu 0GB \
  --max-hessian-memory-disk 30GB \
  --max-model-memory-gpu 7GB \
  --max-model-memory-cpu 7GB \
  --log-level extensive
```

Without `--samples` or `--seed`, the script uses `calibration_samples` and
`calibration_seed` from the dataset config (currently 2 and null). Use
`--log-level none|normal|extensive` for routine, concise, or diagnostic output;
`--timing` adds the total runtime at normal verbosity, and
`--quiet-warnings` suppresses Python and dependency warnings without hiding
errors. This script requires CUDA for execution, so it does not expose a
`--device` option.

The script checks free disk space for disk-backed Hessians before calibration.
Allow additional free space for the model disk-offload cache when the automatic
device map places weights on disk; its size is reported after model loading.
It does not write calibration activations to disk. For example,
`--max-hessian-memory-cpu 0` stores all persistent Hessians on disk. The active
group is still temporarily present on GPU while being updated, and host memory
is used briefly to transfer data between GPU and disk. For example,
`--max-hessian-memory-cpu 500MB` keeps up to 500 MB of stored Hessians in system
RAM and places the remainder on disk, subject to the disk limit. Disk-backed Hessians and the
temporary model cache are removed after success. Pass
`--keep-hessians` to retain disk-backed Hessians, or `--offload-dir` to choose a
scratch location with sufficient free space. RAM-backed Hessians are released
when quantized and are not retained after the process exits.

Before sampling, the script downloads the configured calibration image source
if the local image subset is missing or incomplete. For COCO, the default
`configs/dataset.yaml` setting downloads the full `train2017.zip` archive,
extracts up to 1,024 images, then removes the archive. Change `max_images` in
that config to control the extracted subset; existing extracted images are kept.
Calibration samples are selected from ShareGPT4V records whose image files are
present in that extracted subset. Local paths and file-existence checks are
refreshed against the current `raw_dir`, rather than reused from a previous
machine's dataset cache.

### Quantized output format convention

The default `--output-format standalone` saves a complete checkpoint in the
serialization format expected by the configured base model and inference
loader. For LLaVA 1.5, that means `config.json`, PyTorch `.bin` weight shards
(`pytorch_model*.bin`), tokenizer files, and the image processor. Validate the
config and at least one expected weight shard after export, before reporting
success. Do not silently switch serialization formats.

The output is a **standalone LLaVA checkpoint**. `--output-dir` contains the
model config, sharded FP16 weights, tokenizer, image processor, and
`gptq_metadata.json`; load it directly with `scripts/evaluate_model.py
--pretrained <output-dir>`. Selected GPTQ weights are dequantized to FP16 when
the checkpoint is saved, while unselected layers retain their original base
weights. This matches the current LLaVA inference adapter's checkpoint
interface, but it is not a compact 4-bit checkpoint or a low-bit inference
kernel. Packed layer files are temporary and removed after export. Loading the
base model again and serializing the complete checkpoint adds time and needs
disk space close to the original FP16 checkpoint size.

Pass `--output-format packed-overlay` to keep a compact GPTQ artifact instead
of writing a complete FP16 model. It stores packed 2/3/4/8-bit codes and their
scales/zero-points under `layers/`, with `gptq_metadata.json`; it does not
reload and reserialize the base checkpoint. For evaluation, pair it with the
same base checkpoint:

```bash
.venv/bin/python scripts/evaluate_model.py \
  --pretrained models/llava-v1.5-7b \
  --overlay-dir outputs/quantized/llava-gptq-w4g128-packed \
  --tasks pope
```

This packed format reduces the quantized artifact's disk size and preserves
the requested low-bit codes. The current LMMS-Eval LLaVA adapter expands those
codes to the model's regular dtype when applying the overlay, so it does not
provide low-bit inference or reduced runtime weight memory. A backend with
GPTQ low-bit kernels and a compatible LLaVA loader is still needed for that.
`standalone` remains the default because it is directly loadable by the
existing LLaVA checkpoint loader.
