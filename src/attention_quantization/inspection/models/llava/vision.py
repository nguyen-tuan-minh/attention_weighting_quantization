"""CLIP vision targets for shared inspection."""
import torch

from .inspector import LlavaInspector, first
from ...core.targets import CaptureTarget


class LlavaVisionInspector(LlavaInspector):
    """Inspect the loaded CLIP encoder without changing decoder target indices.

    Block/attention targets use zero-based encoder block indices. ``linear.input``
    uses indices into ``linear_modules()`` for quantization of individual Linear
    modules. These index domains must be selected in separate capture sessions.
    """

    def __init__(self, model):
        self.tower = model.get_vision_tower()
        encoder = self.tower.vision_tower
        if getattr(encoder.config, "model_type", None) != "clip_vision_model":
            raise ValueError("LlavaVisionInspector requires a loaded CLIP vision encoder")
        super().__init__(encoder)

    def linear_modules(self):
        """Return encoder Linear modules in stable named_modules order."""
        return {name: module for name, module in self.model.named_modules()
                if isinstance(module, torch.nn.Linear)}

    def targets(self):
        """Resolve CLIP boundaries using the decoder's semantic target names."""
        targets = []
        for index, layer in enumerate(self.model.vision_model.encoder.layers):
            targets.extend([
                CaptureTarget("block.input", index, layer, first, before=True),
                CaptureTarget("block.output", index, layer, first),
            ])
            modules = {
                "input_norm": layer.layer_norm1,
                "post_attention_norm": layer.layer_norm2,
                "attention": layer.self_attn,
                "mlp": layer.mlp,
                "attention.query": layer.self_attn.q_proj,
                "attention.key": layer.self_attn.k_proj,
                "attention.value": layer.self_attn.v_proj,
                "attention.output_projection": layer.self_attn.out_proj,
            }
            for name, module in modules.items():
                targets.append(CaptureTarget(name + ".input", index, module, first, before=True))
                output_name = name if name in {"attention.query", "attention.key", "attention.value"} else name + ".output"
                targets.append(CaptureTarget(output_name, index, module, first))
            targets.append(CaptureTarget("attention.weights", index, layer.self_attn,
                                         lambda output: output[1] if len(output) > 1 else None))
        for index, module in enumerate(self.linear_modules().values()):
            targets.append(CaptureTarget("linear.input", index, module, first, before=True))
        return targets
