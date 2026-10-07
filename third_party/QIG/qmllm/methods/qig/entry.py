import os
import torch

from qmllm.methods.qig.quantize.pre_quant import run_qig, apply_qig, apply_scale, get_blocks, get_named_linears
from qmllm.methods.qig.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_model_weight_act
from qmllm.methods.qig.quantize.quantizer import get_module_by_name_suffix
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.utils.search import get_op_name


def apply_qig_with_disk_cache(model, qig_results, w_bit, q_config, wa_quant, a_bit):
    """Apply QIG scales and pseudo quantization one cached decoder layer at a time."""
    layers = get_blocks(model.model)
    scale_groups = [[] for _ in layers]
    layer_prefixes = [get_op_name(model.model, layer) + "." for layer in layers]
    for prev_op_name, layer_names, scales in qig_results["scale"]:
        for index, prefix in enumerate(layer_prefixes):
            if prev_op_name.startswith(prefix):
                scale_groups[index].append(
                    (
                        prev_op_name[len(prefix):],
                        [name[len(prefix):] for name in layer_names],
                        scales,
                    )
                )
                break
        else:
            raise ValueError(f"Could not map QIG scale operation to a decoder layer: {prev_op_name}")

    for index, _ in enumerate(layers):
        layer = model.load_layer_to_device(index, "cuda")
        if scale_groups[index]:
            apply_scale(layer, scale_groups[index])

        named_linears = get_named_linears(layer)
        if not wa_quant:
            for name, linear in named_linears.items():
                linear.weight.data = pseudo_quantize_tensor(
                    linear.weight.data,
                    n_bits=w_bit,
                    **q_config,
                )
        else:
            for name, linear in named_linears.items():
                new_linear = WALinear.from_float(
                    linear,
                    weight_quant="per_channel",
                    act_quant="per_token",
                    w_bit=w_bit,
                    a_bit=a_bit,
                )
                parent = get_module_by_name_suffix(layer, ".".join(name.split(".")[:-1]))
                setattr(parent, name.split(".")[-1], new_linear)
                del new_linear
        model.save_layer_to_disk(index)
        print(f"[QIG] Scaled and pseudo-quantized layer {index + 1}/{len(layers)} on disk.", flush=True)


def qig_entry(model, prompt_inputs, prompt_kwargs, run_qig_process: bool, pseudo_quant: bool, scale_path: str=None, zero_point: str=True, q_group_size: int=128, w_bit: int=4, a_bit: int=16, wa_quant: bool=False, reweight: bool=False, distort: bool=False, loss_mode: str="mae"):
    '''
    model: here the model is the LLM, you have to extract the LLM first! 
    prompt_tokens: the prompt tokens
    prompt_mask: the prompt mask, mask the answer language tokens
    run_qig_process: whether to run the QIG process
    '''
    q_config = {
        "zero_point": zero_point,  # by default True
        "q_group_size": q_group_size,  # whether to use group quantization
    }

    assert scale_path is not None

    scale_exist = os.path.exists(scale_path)
    # reparameterization
    if run_qig_process and not scale_exist:
        model.to_cpu()
        qig_results = run_qig(
            model,
            prompt_inputs,
            prompt_kwargs,
            w_bit=w_bit,
            a_bit=a_bit,
            q_config=q_config,
            auto_scale=True,
            loss_mode=loss_mode,
            wa_quant=wa_quant,
            reweight=reweight,
            distort=distort,
        )
        
        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)
        
        torch.save(qig_results, scale_path)
        print("QIG results saved at", scale_path)

    if pseudo_quant:
        qig_results = torch.load(scale_path, map_location="cpu")
        model.to_cpu()
        if getattr(model, "disk_offload_dir", None):
            apply_qig_with_disk_cache(model, qig_results, w_bit, q_config, wa_quant, a_bit)
        else:
            apply_qig(model.model, qig_results)

            if not wa_quant:
                # weight quantization
                pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
            else:
                # weight activation quantization
                pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    model.to_cuda()
    return model
