# Quantization scripts

## Disk-backed GPTQ for LLaVA 1.5

`quantize_gptq.py` is a standalone implementation of layerwise GPTQ for the
configured LLaVA 1.5 checkpoint. It does not call QIG's GPTQ quantizer and does
not modify LMMS-Eval. Accelerate places model weights across GPU, CPU, and the
specified disk offload directory. It first saves the **original base-model
input activation for every decoder layer** to disk. Each layer's Hessian is
then accumulated from those saved activations in a disk-backed float32 memmap;
outputs from quantized layers are not propagated into later layers. Quantization
loads one linear's Hessian at a time onto the GPU for GPTQ factorization.

Example on a 15 GiB GPU:

```bash
.venv/bin/python scripts/quantize/quantize_gptq.py \
  --output-dir outputs/quantized/llava-gptq-w4g128 \
  --samples 128 \
  --bits 4 \
  --group-size 128 \
  --max-memory-gpu 10GiB \
  --max-memory-cpu 2GiB
```

The script checks free disk space before collecting each layer's Hessians. The
original activation records, temporary model cache, and (by default) Hessians
are stored under the scratch directory. These temporary files are removed after
success. Pass `--keep-hessians` to retain Hessians, or `--offload-dir` to choose
a scratch location with sufficient free space. Calibration activations can
also consume substantial disk space, especially with many samples and image
tokens.

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
GPU even though the full checkpoint and Hessian set stay on disk.
