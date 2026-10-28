# Quantization scripts

## Disk-backed GPTQ for LLaVA 1.5

### Layer selection convention

Quantization scripts should allow targeting a subset of decoder layers while
defaulting to all layers. Layer numbers exposed to users are 1-based and should
accept comma-separated individual numbers and inclusive ranges. For example,
`--layers 1-4,8` selects layers 1, 2, 3, 4, and 8. The script validates the
selection against the model's decoder layer count. Subset runs should compute
statistics and write overlays only for selected layers, and record the selected
layer numbers in output metadata. GPTQ implements this as `--layers`; omitting
the option quantizes every decoder layer.

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
default to unset, so Accelerate chooses placement automatically. If you set
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

The output is a **standalone LLaVA checkpoint**. `--output-dir` contains the
model config, sharded FP16 weights, tokenizer, image processor, and
`gptq_metadata.json`; load it directly with `scripts/evaluate_model.py
--pretrained <output-dir>`. Selected GPTQ weights are dequantized to FP16 when
the checkpoint is saved, while unselected layers retain their original base
weights. This works with the current LLaVA inference adapter but is not a
compact 4-bit checkpoint or a low-bit inference kernel. Packed overlay files
are temporary and removed after export. Loading the base model again and
serializing the complete checkpoint adds time and needs disk space close to the
original FP16 checkpoint size.
