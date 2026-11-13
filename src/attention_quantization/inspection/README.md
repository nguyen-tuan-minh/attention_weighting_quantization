# Model inspection

The public API consists of a model inspector and collector definitions. The
inspector resolves model-specific tensor locations. Collectors own masks,
analysis settings, tensor dependencies, and gradient requirements. Sessions own
hook lifetime, capture scopes, storage, and cleanup.

## Capture collector classes or instances

```python
from attention_quantization.inspection import (
    LlavaInspector, AttentionWeightsCollector, ValueCollector, TensorCapture,
)

inspector = LlavaInspector(model)
print(inspector.capture_targets())  # semantic name -> available layer indices

with inspector.capture(
    targets=[AttentionWeightsCollector, ValueCollector,
             TensorCapture("input_norm.output")],
    layers=[0, 5], storage="cpu",
) as session:
    outputs = model(**inputs)

record = session.results[0]
values = record.tensors[(5, "attention.value")][0]
```

Pass a class when its constructor takes no required arguments, or an instance
when configuration is needed. `TensorCapture` accepts one semantic name or a
sequence of names. Multiple collectors share one capture of each tensor target.
Q/K/V targets are raw projection outputs, before head reshaping, RoPE, and
head repetition. This LLaVA inspector expects decoder blocks at
`model.model.layers`.

## Configure analysis on collectors

```python
from attention_quantization.inspection import IGACollector, AttentionMassCollector

iga = IGACollector(query_mask=assistant_mask, image_mask=image_mask)
mass = AttentionMassCollector(
    query_mask=assistant_mask, image_mask=image_mask, text_mask=text_mask,
)

with inspector.capture(
    targets=[iga, mass], layers=[5, 10], release_attention=True,
) as session:
    outputs = model(**inputs)

record = session.results[0]
image_scores = record.collections[iga.name].measurements[iga.measurement_key(5)][0][0]
modality_mass = record.collections[mass.name].measurements[mass.measurement_key(5)][0][0]
```

Masks accept `[sequence]` or `[batch, sequence]`. Each collector owns its masks;
none are passed to `capture()` or `session.forward()`. Constructors copy masks
and each model forward snapshots collector configuration, so changing a collector
later cannot change an earlier forward's analysis.

| Collector | Configuration | Measurement |
| --- | --- | --- |
| `TensorCapture` | Semantic targets; optional `gradients=True` | Raw tensors and optional derivatives |
| `AttentionWeightsCollector` | Optional `gradients=True` | Full attention weights |
| `ValueCollector` | Optional `gradients=True` | Value-projection outputs |
| `IGACollector` | `query_mask`, `image_mask` | Mean attention over selected queries and heads |
| `AttentionMassCollector` | Query, image, and text masks | Modality sums and head/query row count |
| `GradientImportanceCollector` | Target, optional image mask, reduction, accumulation | Per-token gradient scores |
| `OverlapCollector` | Query/image masks, gradient settings, top fraction | IGA, gradient importance, and top-token intersection fraction |

Collectors default to their class name as their result namespace. Give explicit
unique names when using two instances of the same class:

```python
first = IGACollector(query_mask=prompt_mask, image_mask=image_mask, name="prompt")
second = IGACollector(query_mask=answer_mask, image_mask=image_mask, name="answer")
session = inspector.capture(targets=[first, second])
```

### Measurement labels

`name` identifies a collector's namespace; `label` identifies its primary
measurement within that namespace. Every built-in measurement collector accepts
`label`. Use `collector.measurement_key(layer)` to access it without repeating
a string literal. `collector.labels` lists all its declared outputs.

```python
iga = IGACollector(
    query_mask=assistant_mask, image_mask=image_mask,
    name="answer_analysis", label="patch_relevance",
)
with inspector.capture(targets=[iga], layers=[5]) as session:
    outputs = model(**inputs)
scores = session.results[0].collections[iga.name].measurements[
    iga.measurement_key(5)
][0][0]
```

Overlap also accepts `iga_label` and `importance_label` for its child
measurements. Its output and both child outputs must have distinct labels because
they share the composite result namespace. Dependency lookups use the child's
declared output key, so renaming is safe:

```python
from attention_quantization.inspection import OverlapCollector

collector = OverlapCollector(
    query_mask=assistant_mask, image_mask=image_mask,
    iga_label="attention_scores",
    importance_label="sensitivity",
    label="agreement",
)
# Keys are collector.iga.measurement_key(layer),
# collector.importance.measurement_key(layer), and collector.measurement_key(layer).
```

Labels are snapshotted with the rest of the collector configuration. When accessing
an older forward after changing a label, use
`record.collections[name].collector.measurement_key(layer)` to reference that
forward's declaration. Invalid labels and collisions are rejected before capture.

## Gradients

```python
from attention_quantization.inspection import GradientImportanceCollector

importance = GradientImportanceCollector(
    gradient_target="attention.value", image_mask=image_mask,
    reduction="sum_squared", accumulate=True,
)
with inspector.capture(targets=[importance], layers=[5, 10], storage="cpu") as session:
    outputs = model(**inputs)
    losses = compute_answer_token_losses(outputs)
    for index, loss in enumerate(losses):
        session.compute_gradients(loss, retain_graph=index < len(losses) - 1)

scores = session.results[0].collections[importance.name].measurements[
    importance.measurement_key(5)
][0]
```

`compute_gradients(loss, forward=-1, retain_graph=False)` differentiates an explicit
scalar objective. The session infers gradient requirements from the collectors;
there is no inspector `gradients` parameter. `ValueCollector(gradients=True)` or
`TensorCapture(..., gradients=True)` requests raw derivatives.

The model forward must run with gradients enabled and selected activations must
require gradients. Derivatives use `torch.autograd.grad`, leaving parameter
`.grad` fields untouched. A derivative shared by multiple collectors is computed
once and dispatched to each collector's independent analysis.

The default reduction `l2` takes each token's feature-vector norm. `sum_squared`
sums squared feature gradients; `accumulate=True` adds reduced scores over
objectives independently for each forward, layer, and module invocation.
Overlap defaults to `top_fraction=0.1`, selects `ceil(fraction * image_tokens)`
tokens, and divides intersection size by that count.

## Results and scopes

Each `ForwardCapture` contains:

- `tensors[(layer, semantic_target)]`: a list in module-call order, containing a
  tensor/path or None when all consumers reduced the capture immediately.
- `gradients[(layer, semantic_target)]`: derivative tensors/paths appended in
  objective order, then module-call order.
- `collections[collector.name].measurements[(layer, measurement)]`: collector
  results in call/objective order; built-in reductions contain per-sample values.
  Accumulating importance maintains one result slot per module call.

If raw and reduced collectors share a target, both the raw tensor and derived
measurements are available. Result namespaces prevent different masks or
reductions from overwriting each other.

Use `scoped=True` to attach hooks once and capture only explicit measured
forwards. `session.forward()` takes no analysis arguments and must contain
exactly one model forward. Nested scopes are rejected. Normal scope exit validates
that every requested tensor was captured; private graph references are released
on scope exit, so gradients must be computed inside the scope.

```python
with inspector.capture(targets=[ValueCollector], scoped=True) as session:
    generated = generate_answer(model, inputs)  # capture is disabled
    with session.forward():
        outputs = model(**scoring_inputs)
    values = session.results[-1].tensors
```

For sample-specific masks, construct configured collectors for each sample and
open a capture session around that sample. When reusing an existing scoped session,
you may update collector masks between scopes; each model forward snapshots them.
Collector names and tensor/gradient dependencies must remain fixed after session
construction. Call `session.clear()` between samples to release result references.

## Storage and cleanup

`capture()` accepts explicit lifecycle/storage settings: `layers`, `storage`,
`directory`, `dtype`, `scoped`, `sink`, `retain_tensors`, and `release_attention`.
No shared analysis-options dictionary or named `collection` selector is used.

Storage modes are `cpu` (detached copies), `live` (original tensors), and `disk`
(detached files; requires `directory`). Gradient sessions default to live storage;
others default to CPU. `dtype` converts only detached CPU/disk snapshots. Private
live references are retained for derivatives even when public storage is CPU/disk.
Public live results keep their own graph references until the caller releases them.

A synchronous `sink(forward_index, layer, target, call_index, stored_value)` receives
each completed tensor capture. If any consumer requests the raw tensor, the sink
gets its stored tensor/path; otherwise it receives a dictionary mapping collector
names to stored measurement dictionaries. Reduced results always remain available
in their collector namespaces. `retain_tensors=False` avoids retaining raw result
values in the session; the sink must consume or write them promptly. Caller-owned
disk files are never deleted by cleanup.

Attention capture requires eager attention and `output_attentions=True` on the
model. `release_attention=True` removes returned weights after registered capture
consumers have processed them. LLaVA preserves the remaining output tuple entries;
external hooks registered later will observe released weights. This reduces output
retention, not the eager attention allocation itself.

Context exit removes hooks on success or error. Use one inspector per model;
concurrent sessions and nested scopes are unsupported. Compiled/fused paths and
checkpoint recomputation require additional model-specific support.

## Extending models and collectors

New model inspectors subclass `ModelInspector` and implement `targets()`,
`tensor_collector()`, and `gradient_collector()`. They can override
`collection_group(targets)` to specialize definition resolution and
`release_attention_hooks()` to implement output release. Generic high-level
collectors operate whenever the model exposes their semantic dependencies.

New public collector definitions subclass `Collection`. Implement
`required_targets()` and `measure_tensor(target, tensor)` for tensor statistics;
implement `wants_gradient()` and `process_gradient()` and set `needs_grad=True`
for derivative analysis. Keep masks/options in the collector constructor.
Use `validate()` for configuration checks and override `snapshot()` when a custom
configuration needs special copying. Runtime accumulation belongs in the supplied
`CollectionCapture`, not the reusable definition. Pass the class or instance
directly to `capture(targets=[...])`; no registry or session changes are needed.

For a configurable single measurement, set `default_label` on the class,
accept `label=None` in custom constructors and forward it to the base
constructor, then return `(self.label, values)` from `measure_tensor()`.
Gradient writers should use `self.measurement_key(layer)`, and composite readers
should use the producing child's `measurement_key(layer)`. Override `labels`
to declare multiple outputs and validate that their names cannot collide within
one collector namespace. Raw tensor targets keep their model-defined semantic
names and do not declare measurement outputs.

Numerical functions in `collections/attention.py` and `collections/measurements.py`
remain stateless. High-level workflows coordinate these functions; model-specific
primitive collectors own hook extraction.

## Tests

Run in an environment with PyTorch installed:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Shared tests live in `tests/inspection/shared/`; LLaVA tests live in
`tests/inspection/llava/`. Each future inspector gets its own test package.
