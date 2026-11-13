"""Generic module-boundary capture, specialized through target definitions."""
import torch
from ..core.collector import Collector


class TensorCollector(Collector):
    """Observe model-defined tensor boundaries without altering model outputs."""

    def __init__(self, targets):
        """Keep resolved targets and an initially empty list of owned hook handles."""
        self.targets = list(targets)
        self.handles = []

    def start(self, publish):
        """Attach hooks that call publish(target, live_tensor) for each invocation.

        Targets select module positional inputs or outputs and extract tensors.
        Invalid captures raise RuntimeError; partial registration is rolled back.
        """
        try:
            for target in self.targets:
                def receive(module, values, output=None, *, target=target):
                    """Extract and publish a tensor; return None to preserve model values."""
                    tensor = target.extract(values if target.before else output)
                    if not torch.is_tensor(tensor):
                        raise RuntimeError(
                            f"{target.name} at layer {target.layer} is unavailable; "
                            "attention weights require eager attention and output_attentions=True"
                        )
                    publish(target, tensor)
                handle = (target.module.register_forward_pre_hook(receive)
                          if target.before else target.module.register_forward_hook(receive))
                self.handles.append(handle)
        except BaseException:
            self.close()
            raise

    def close(self):
        """Remove all owned hooks; repeated calls are safe."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
