# Attention Weighting Quantization

## Project overview

The project studies how quantizing attention weights affects LLaVA 1.5 7B. The specific method, quantization settings, and evaluation questions will be documented here as they are finalized.

## Model and data

- **Model:** LLaVA 1.5 7B. See the [official LLaVA repository](https://github.com/haotian-liu/LLaVA) for model setup and licensing details.
- **Dataset:** ShareGPT4V, with COCO as the default source. The dataset preparation script filters records by source and can optionally select a reproducible subset. See [ShareGPT4V data documentation](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md) and the [dataset page](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V).

Please follow the upstream dataset and model terms when downloading or using these resources. The data and pretrained model weights are not included in this repository.

### Downloading image data and preparing ShareGPT4V records

Create the project virtual environment and install its dependencies:

```bash
bash scripts/set_up.sh
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
```

The requirements pin PyTorch to its CUDA 12.1 build to match the older NVIDIA driver reported on the target machine. That driver must still expose an NVIDIA GPU to the environment; the setup script cannot update a system driver. Confirm GPU access with `nvidia-smi` and `python scripts/check_device.py` after setup.

If the setup script is already running inside the repository's `.venv`, it exits without changing anything. Check which compute devices PyTorch can use with:

```bash
python scripts/check_device.py
```

Dataset locations are set in [`configs/dataset.yaml`](configs/dataset.yaml). Relative paths there resolve from the repository root. The ShareGPT4V module downloads image archives and extracts them under the configured `raw_dir`; COCO `train2017` is the default:

```bash
python -m attention_quantization.data.sharegpt4v
python -m attention_quantization.data.sharegpt4v --dataset gqa
```

The data module defaults to COCO. Calling the loader downloads the archive only when its images are not already present, loads the ShareGPT4V records, and connects their image paths to the local files. It recognizes a full existing COCO `train2017` directory even if it was extracted manually or by an older script, then leaves a marker so future runs can reuse it. It removes each downloaded ZIP after extraction. The same archive cleanup applies to all supported image sources, including both Visual Genome archives. An interrupted `.part` download is removed before retrying.

```python
from attention_quantization.data import load_sharegpt4v_dataset

dataset = load_sharegpt4v_dataset()
```

Image values use Hugging Face's lazy `Image` feature: image files stay on disk and are decoded when accessed (for example, `dataset[0]["image"]`). The dataset files are cached under the configured `cache_dir`. Select another supported source with `source="gqa"`, `source="textvqa"`, or `source="visual-genome"`; set `download_images=False` to skip image downloading when the files are already available. To save a local COCO dataset, optionally sampled, run:

```bash
python scripts/prepare_dataset.py
```

The script saves the full COCO subset under the configured `processed_dir/sharegpt4v_coco`. Other downloadable sources can be selected with `--source gqa`, `--source textvqa`, or `--source visual-genome`; `--source all` keeps all ShareGPT4V records. Sampling is optional and happens in this script: `python scripts/prepare_dataset.py --samples 128 --seed 42`. Pass a different config file with `--config`. ShareGPT4V image sources can have separate or restricted download steps; see its [data instructions](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md).

### Run LLaVA calibration and inspect attention

[`configs/dataset.yaml`](configs/dataset.yaml) sets `calibration_samples` to `2` and `calibration_seed` to `null` by default. The model ID, local download directory, dtype, device mapping, and attention implementation are in [`configs/model.yaml`](configs/model.yaml).

The forward-only script prepares and saves the calibration dataset first. Its downloader extracts COCO and removes the ZIP before it downloads or loads LLaVA. The script then forwards the selected samples:

```bash
python scripts/run_calibration.py
```

The attention investigation script has two modes. The default `online` mode computes a separate IGA map for every layer, using non-padding text queries after the image tokens. It opens one Matplotlib figure per sample with the processor image and all per-layer overlays together, using a logarithmic color scale. It does not save attention maps. The `save` mode writes each captured language attention matrix under `attention_output_dir` for later analysis. It prepares the dataset and removes the COCO ZIP before downloading or loading LLaVA. Its five sections are: prepare calibration, load model, register hooks, forward, and analyse.

```bash
python scripts/investigate_attention.py                 # online analysis
python scripts/investigate_attention.py --mode save     # save attention maps
python scripts/investigate_attention.py --timing        # print step durations
python scripts/investigate_attention.py --heatmap-only  # standalone maps, no image overlay
```

Both scripts forward each image with its ShareGPT4V user prompt and assistant caption. Images are decoded one at a time. Override the configured sample count or seed with `--samples` and `--seed`.

### Reusing components with other models or datasets

Task-independent helpers live in `src/attention_quantization/`: `config.py` reads YAML and resolves repository paths, `models/huggingface.py` loads a Transformers model and processor, and `data/conversation.py` reads user/assistant turns from a sample. Dataset-specific loading remains in `data/sharegpt4v.py`.

The scripts keep calibration selection and forwarding local to the workflow. Attention hooks, IGA calculation, and plots also remain in `investigate_attention.py`; these are analysis-specific rather than shared infrastructure. To try a different model checkpoint, update `model_id`, `model_dir`, and `model_class` in `configs/model.yaml`. The default `AutoModelForVision2Seq` covers compatible Transformers vision-to-sequence models; `LlavaForConditionalGeneration` is also supported for checkpoints that require the explicit class. A different dataset can provide its own loader while reusing the common conversation helper when its records use the same turn format.

### Run QIG quantization

On the QIG branch, this script clones QIG and the two companion repositories named by QIG (LLaVA-NeXT and its LMMS-Eval fork) into `.third_party/QIG`, creates a separate Python 3.11 environment, prepares ShareGPT4V COCO calibration records using this project's dataset module, and runs a selected QIG method against LLaVA 1.5. It uses images under the configured `raw_dir`; if a selected calibration image is missing, it fetches only that JPEG and never downloads the full COCO ZIP during quantization:

```bash
bash scripts/quantize_qig.sh \
  --output-dir models/quantized/llava-1.5-7b-qig-w4g128 \
  --samples 128 \
  --seed 42 \
  --method qig \
  --w-bit 4 \
  --w-group 128
```

The output directory must not already exist unless `--overwrite` is passed. Supported `--method` values are `qig`, `mbq`, `awq`, `smoothquant`, `rtn`, and `gptq`, matching the methods exposed by QIG's quantization wrapper. Add `--reweight` or `--distort` for methods that support those options (`qig` and `mbq`). Use `--a-bit`, `--alpha`, and `--percdamp` to set activation precision, SmoothQuant scaling, and GPTQ damping. RTN does not need calibration data; the other methods sample ShareGPT4V COCO. The script uses micro batches of one by default to limit GPU memory use; change this with `--micro-batch-size` if needed.

The QIG adapter is isolated in `src/attention_quantization/quantization/qig/`. `requirements-qig.txt` lists the imports reached by this LLaVA workflow and pins the same PyTorch 2.5.1 CUDA 12.1 build used by the project's Linux setup. It includes `protobuf` required by the LLaVA source. The setup script deliberately does not install QIG's broad `requirements.txt`; it installs the three source checkouts editable with dependency resolution disabled, then installs the selected dependency set. This keeps QIG's historical PyTorch 2.8 pin from replacing the project's CUDA-compatible build. QIG source and dependencies stay in `.third_party/QIG` and `.venv-qig`, separate from `.venv`.

There is one upstream caveat: QIG's README model list does not advertise classic `llava`, but the checked-in QIG package contains an `llava_v15` processor and its companion LMMS-Eval fork registers a `llava` model adapter. This integration connects those two source paths directly for LLaVA 1.5. The source and dependency versions are recorded in each artifact; the combination still needs a run on the target CUDA machine to confirm runtime compatibility.

QIG's pseudo quantization path saves scale parameters rather than exporting packed low bit weights. This script saves the resulting model checkpoint, method scales when applicable, the sampled calibration JSONL, and run metadata in the chosen directory. For pseudo-quantized weights, model tensors remain floating point, so the research checkpoint does not provide packed INT4 storage savings. The source implementation and its model adapter are documented in the [QIG repository](https://github.com/ucas-xiang/QIG).

Load the saved checkpoint with the QIG environment:

```python
from attention_quantization.models import load_qig_quantized_model

wrapper = load_qig_quantized_model("models/quantized/llava-1.5-7b-qig-w4g128")
model = wrapper._model
tokenizer = wrapper._tokenizer
```

The loader uses QIG's LMMS-Eval `llava` adapter, so call it from `.venv-qig` after running the setup/quantization script.

## Documentation

Planning notes and supporting project documentation are in [`docs/`](docs/). The [project structure reference](docs/project_structure.md) describes the planned organization of the download, quantization, and evaluation scripts and packages.

## License

See [LICENSE](LICENSE) for this repository’s license. Third-party model weights and dataset assets may have separate terms.
