# Model inspection

`LlavaInspector` is the public entry point for the LLaVA implementation with
decoder layers at `model.model.layers`. Model execution remains explicit.

```python
from attention_quantization.inspection import LlavaInspector

inspector = LlavaInspector(model)
print(inspector.capture_targets())  # target name -> supported layer indices

with inspector.capture(
    targets=["attention.query", "attention.key", "input_norm.output"],
    layers=[0, 5],
    storage="cpu",
) as capture:
    outputs = model(**inputs)

tensor = capture.results[0].tensors[(0, "attention.key")][0]
```

Each result represents one model forward. Tensor keys are `(layer, target)`;
values are lists of module invocations within that forward. All batch items
are preserved. Q/K/V targets are raw projection outputs, before head reshaping,
RoPE, or grouped-query head repetition.

## Aggregate measurements

Collections available through `capture(collection=...)`:

| Collection | Required options | Measurement |
| --- | --- | --- |
| `iga` | `query_mask`, `image_mask` | Mean attention over selected queries and heads |
| `attention_mass` | `query_mask`, `image_mask`, `text_mask` | Summed image/text attention and head/query row count |
| `gradient_importance` | Scalar objective supplied later | Per-token gradient L2 norm |
| `overlap` | `query_mask`, `image_mask`, scalar objective | IGA, image-token gradient L2 norms, top-token intersection fraction |

Masks accept `[sequence]` or `[batch, sequence]` boolean arrays. IGA and
importance results are lists of per-sample vectors, supporting different token
counts. An optional `image_mask` restricts gradient importance to image tokens.
Gradient collections default to `gradient_target="attention.value"`.
Overlap uses `ceil(top_fraction * image_tokens)` tokens, with
`top_fraction=0.1` by default; intersection size is divided by that count.

```python
with inspector.capture(
    collection="overlap", layers=[5, 10],
    query_mask=assistant_mask, image_mask=image_mask, storage="cpu",
) as capture:
    outputs = model(**inputs)
    capture.compute_gradients(loss_function(outputs))

scores = capture.results[0].measurements[(5, "iga")][0]
overlap = capture.results[0].measurements[(5, "overlap")][0]
```

Attention collection requires eager attention with `output_attentions=True`.
Configure those on the model before the forward. The inspector reports an
error if a requested tensor is unavailable. It leaves module outputs intact,
so model-returned attention matrices can still consume memory.

## Gradients, storage, and lifecycle

Use `gradients=True` with arbitrary tensor targets to differentiate those
captures. `compute_gradients(loss, forward=-1, retain_graph=False)` uses
`torch.autograd.grad`, leaving parameter `.grad` fields untouched. Inputs and
the forward must enable gradients. Multiple objectives on one forward require
`retain_graph=True` for all but the final objective. Repeated gradient results
are appended in objective order, then module-call order.

Storage is `cpu` (detached copies), `live` (original tensors), or `disk`
(detached tensor files; requires `directory`). Gradient captures default to
`live`; others default to `cpu`. Gradient sessions retain private graph
references even with CPU/disk storage until differentiation or scope exit.
Live results retain their own graphs until the caller releases them or calls
`capture.clear()`. Disk files belong to the caller and are never deleted.

The context manager removes hooks on success or error. `capture.enabled=False`
temporarily suspends capture. `capture.clear()` releases stored result and
private graph references. Concurrent sessions on one inspector are rejected;
use one inspector per model and avoid concurrent forwards in a session.
Module checkpoint recomputation and compiled/fused paths require model-specific
support and should not be used with this initial implementation.

## Adding a model

Subclass `core.inspector.ModelInspector`. Implement `targets()` with
`CaptureTarget` objects, `tensor_collector()`, and `gradient_collector()`.
Each target binds a semantic name and layer index to a module input/output
extractor. Provide model-specific collector subclasses when extraction,
instrumentation, or gradient semantics differ. Override `collection_collector`
for model-specific composite wiring. Generic capture also supports additional
target names supplied by a new inspector.

Shared composite collectors coordinate registration and rollback. The default
composition partitions resolved targets into attention and other captures so
dependencies are registered once. Session-level reducers produce the aggregate
measurements after extraction and differentiation.

Run behavioral tests in an environment with PyTorch installed:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Shared infrastructure tests live in `tests/inspection/shared/`; LLaVA-specific
tests live in `tests/inspection/llava/`. Give each future inspector its own
test package, following `tests/README.md`.
