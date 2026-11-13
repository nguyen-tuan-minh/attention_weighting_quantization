# Repository conventions

These conventions apply to library code, scripts, and tests. Script-specific
configuration and workflow rules live in [scripts/README.md](scripts/README.md);
investigation workflows live in
[scripts/investigate/README.md](scripts/investigate/README.md).

## Readability and state

- **Document function contracts.** Add a docstring to each function and method,
  including internal helpers and hook callbacks. Describe its purpose, inputs,
  return value, assumptions, and state it reads or changes where relevant.
  Include tensor shapes, axis meanings, devices, and gradient requirements when
  these are needed to use the function correctly. Keep simple contracts brief;
  use `Args`, `Returns`, and `Raises` sections for more complex interfaces.
- **Document hooks.** Explain which module input or output a hook reads, what
  it captures, how it transforms or reduces the data, and whether it changes
  the model output. Document registration, capture gating, and removal.
- **Describe shared state at initialization.** Document dictionary keys, value
  types, tensor shapes, and whether values belong to a module call, forward,
  sample, or full run. For class state, explain ownership, reset points, and
  the lifetime of retained tensors or computation graphs. Update this
  documentation whenever the state schema changes.
- **Make data flow traceable.** Keep preprocessing, transformations, model
  execution, validation, and result storage easy to follow. Use descriptive
  names and comments that explain purpose, assumptions, or non-obvious behavior.
  Avoid comments that merely repeat the statement below them.
- **Clarify side effects and cleanup.** Identify functions that register hooks,
  mutate shared state or model settings, retain computation graphs, write files,
  or display plots. Make cleanup explicit through context managers or
  `try`/`finally`, including cleanup after partial initialization and failures.

## Function comments

Use docstrings for the function's contract and inline comments for local
reasoning. Document behavior that is true of the implementation rather than
intended behavior that has not been implemented. For example:

```python
def capture_output(module, inputs, output):
    """Store this layer's output without changing the module result.

    Reads a live [batch, sequence, hidden] output tensor. Appends a detached
    CPU copy to the current forward's captures; retains no autograd graph.
    Returns None so the forward hook preserves the original module output.
    """
    captures.append(output.detach().to("cpu", copy=True))
```
