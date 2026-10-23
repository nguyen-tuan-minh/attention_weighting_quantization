# Quantization scripts

## Disk-backed GPTQ for LLaVA 1.5

`quantize_gptq.py` is a standalone implementation of layerwise GPTQ for the
configured LLaVA 1.5 checkpoint. It does not call QIG's GPTQ quantizer and does
not modify LMMS-Eval. Accelerate places model weights across GPU, CPU, and the
specified disk offload directory. It accumulates each layer's Hessian online
during unquantized base-model forwards; activations are discarded and never
saved to disk. Hessian matrices use a configurable RAM budget, with the rest in
disk-backed float32 memmaps; each matrix is copied to the GPU only when its
layer is quantized. Quantized outputs are not used to collect later Hessians.
`--hessian-token-scope all` (the default) uses every valid image and text token
position. Set it to `text` to exclude image positions; this uses the adapter's
vision mask and attention mask, so padding is excluded as well.

Example on a 15 GiB GPU:

```bash
.venv/bin/python scripts/quantize/quantize_gptq.py \
  --output-dir outputs/quantized/llava-gptq-w4g128 \
  --samples 128 \
  --bits 4 \
  --group-size 128 \
  --hessian-token-scope text \
  --hessian-ram-gib 0.5 \
  --max-memory-gpu 10GiB \
  --max-memory-cpu 2GiB
```

The script checks free disk space for disk-backed Hessians before calibration.
It does not write calibration activations to disk. The Hessian RAM budget
defaults to `None` (unlimited), so all Hessians stay in RAM unless you set
`--hessian-ram-gib`. For example, `--hessian-ram-gib 0.5` limits Hessians in
RAM to 0.5 GiB and stores the remainder on disk. Disk-backed Hessians and the
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
