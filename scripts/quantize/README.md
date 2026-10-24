# Quantization scripts

## Disk-backed GPTQ for LLaVA 1.5

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
accumulates full float32 Hessians on GPU while forwarding each calibration
sample through the model once. At each group of decoder layers, it loads that
group's accumulated Hessians from configured RAM/disk backing to GPU, updates
them, then writes them back before continuing. `--hessian-group-layers` is the
number of decoder layers paged to GPU at once; larger groups need more VRAM.
Because groups are loaded and saved for every sample, disk-backed Hessians can
make calibration substantially slower. GPU and CPU placement limits
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

Example on a 15 GiB GPU:

```bash
.venv/bin/python scripts/quantize/quantize_gptq.py \
  --output-dir outputs/quantized/llava-gptq-w4g128 \
  --samples 128 \
  --seed 42 \
  --bits 4 \
  --group-size 128 \
  --hessian-token-scope text \
  --max-hessian-memory-gpu 7GB \
  --max-hessian-memory-cpu 500MB \
  --max-hessian-memory-disk 30GB \
  --max-model-memory-gpu 10GB \
  --max-model-memory-cpu 2GB \
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

The output is a **packed weight overlay**, not a standalone Hugging Face
checkpoint. It stores quantized decoder Linear weights and their scales/zero
points under `layers/`; the original model checkpoint stays untouched. Load
the base model using the same LLaVA implementation, then attach the overlay:

```python
from scripts.quantize.quantize_gptq import apply_overlay

handles = apply_overlay(model, "outputs/quantized/llava-gptq-w4g128/layers")
```

The hook dequantizes a layer's packed weights after the model's offload hook
loads that layer. This provides a reference path for inspecting/evaluating the
quantized weights; it is not a low-bit inference kernel. The current script is
limited to LLaVA 1.5 models exposing `model.layers` and uses dense GPTQ Hessian
factorization, so one Hessian and its factorization workspace must fit on the
GPU even though model weights are partially offloaded and Hessians are split
between RAM and disk.
