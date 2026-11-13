from ...core.inspector import ModelInspector
from ...core.targets import CaptureTarget
from .collectors import LlavaTensorCollector, LlavaGradientCollector


def first(value):
    """Extract a tuple's first tensor or return an unwrapped tensor unchanged."""
    return value[0] if isinstance(value, tuple) else value


class LlavaInspector(ModelInspector):
    """Inspect the LLaVA decoder at model.model.layers without loading models."""

    def targets(self):
        """Describe available decoder module inputs/outputs without attaching hooks.

        Returns CaptureTargets indexed by decoder layer. Q/K/V are raw projection
        outputs; attention.weights extracts tuple element 1 when available.
        Optional modules are advertised only when present. Raises ValueError if
        the model does not expose the expected LLaVA decoder layout.
        """
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("Expected LLaVA decoder at model.model.layers")
        targets = []
        for index, layer in enumerate(layers):
            targets.extend([
                CaptureTarget("block.input", index, layer, first, before=True),
                CaptureTarget("block.output", index, layer, first),
            ])
            modules = {
                "input_norm": getattr(layer, "input_layernorm", None),
                "post_attention_norm": getattr(layer, "post_attention_layernorm", None),
                "mlp": getattr(layer, "mlp", None),
            }
            attention = getattr(layer, "self_attn", None)
            if attention is not None:
                modules["attention"] = attention
                targets.append(CaptureTarget("attention.weights", index, attention,
                                             lambda output: output[1] if isinstance(output, tuple) and len(output) > 1 else None))
                for name, attr in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj"), ("output_projection", "o_proj")):
                    modules[f"attention.{name}"] = getattr(attention, attr, None)
            for name, module in modules.items():
                if module is None:
                    continue
                targets.append(CaptureTarget(name + ".input", index, module, first, before=True))
                # Projection names refer to raw projection outputs, before RoPE
                # and before splitting or repeating heads.
                output_name = name if name in {"attention.query", "attention.key", "attention.value"} else name + ".output"
                targets.append(CaptureTarget(output_name, index, module, first))
        return targets

    def tensor_collector(self, targets):
        """Return an unstarted LLaVA tensor collector for resolved targets."""
        return LlavaTensorCollector(targets)

    def gradient_collector(self):
        """Return a gradient collector preserving LLaVA capture layouts."""
        return LlavaGradientCollector()
