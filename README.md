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
python -m attention_quantization.data.sharegpt4v --dataset coco
python -m attention_quantization.data.sharegpt4v --dataset coco --max-images 50000 --seed 42
```

The data module defaults to COCO. `configs/dataset.yaml` sets `max_images: 1024` and `download_seed: null`, so downloading extracts a random subset of up to 1,024 images by default; a null seed chooses a new subset on each fresh extraction. Set a numeric `download_seed` for a repeatable subset. `--max-images` and `--seed` override these settings for one run. Setting `max_images: null` in the config extracts every image. The archive itself is still downloaded in full, then deleted after extraction. Existing image files are kept, so lowering the limit does not remove files already on disk. The loader also accepts `max_images=...` and `seed=...`, and removes each archive after extraction. An interrupted `.part` download is removed before retrying. QIG filters its calibration candidates to images present locally, so it can use a capped COCO extraction without attempting to fetch missing images.

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

[`configs/dataset.yaml`](configs/dataset.yaml) sets `calibration_samples` to `2` and `calibration_seed` to `null` by default. [`configs/model.yaml`](configs/model.yaml) selects the original `liuhaotian/llava-v1.5-7b` checkpoint and stores it under `models/llava-v1.5-7b`.

On a new machine, create the shared Python 3.11 project environment with `bash scripts/set_up.sh`. Use `.venv/bin/python` for calibration and attention commands. If `.venv` already exists with another Python version, move or remove that environment before setup; virtual environments cannot be merged in place.

To download only the model from the Hub, run:

```bash
.venv/bin/hf download liuhaotian/llava-v1.5-7b --local-dir models/llava-v1.5-7b
```

The forward-only script prepares and saves the calibration dataset first. Its downloader extracts COCO and removes the ZIP before it downloads or loads LLaVA. The script then forwards the selected samples:

```bash
.venv/bin/python scripts/run_calibration.py
```

The attention investigation script has two modes. The default `online` mode computes a separate IGA map for every layer, using non-padding text queries after the image tokens. It opens one Matplotlib figure per sample with the processor image and all per-layer overlays together, using a logarithmic color scale. It does not save attention maps. The `save` mode writes each captured language attention matrix under `attention_output_dir` for later analysis. It prepares the dataset and removes the COCO ZIP before downloading or loading LLaVA. Its five sections are: prepare calibration, load model, register hooks, forward, and analyse.

```bash
.venv/bin/python scripts/investigate_attention.py                 # online analysis
.venv/bin/python scripts/investigate_attention.py --mode save     # save attention maps
.venv/bin/python scripts/investigate_attention.py --timing        # print step durations
.venv/bin/python scripts/investigate_attention.py --heatmap-only  # standalone maps, no image overlay
```

Both scripts first filter calibration records to images that exist locally, then sample from that available subset; this works with the configured 1,024-image extraction and skips missing COCO files. They forward each image with its ShareGPT4V user prompt and assistant caption, decoding images one at a time. Override the configured sample count or seed with `--samples` and `--seed`.

### Reusing components with other models or datasets

Task-independent helpers live in `src/attention_quantization/`: `config.py` reads YAML and resolves repository paths, `models/loader.py` selects the configured model loader, and `data/conversation.py` reads user/assistant turns from a sample. Dataset-specific loading remains in `data/sharegpt4v.py`.

The scripts keep calibration selection and forwarding local to the workflow. Attention hooks, IGA calculation, and plots also remain in `investigate_attention.py`; these are analysis-specific rather than shared infrastructure. This QIG workflow currently targets the original `liuhaotian/llava-v1.5-7b` checkpoint format. A different dataset can provide its own loader while reusing the common conversation helper when its records use the same turn format.

### Run QIG quantization

The QIG source is kept under `third_party/QIG` so its code can be maintained with this project. The LLaVA-NeXT and LMMS-Eval companion checkouts are ignored by Git and cloned there by the setup script. The script uses the shared Python 3.11 `.venv`, prepares ShareGPT4V COCO calibration records using this project's dataset module, and runs a selected method against the original `liuhaotian/llava-v1.5-7b` model. When `--base-model` is omitted, it downloads the model from `configs/model.yaml` into `models/llava-v1.5-7b` if that directory does not already contain a config and weight file. It filters calibration candidates to image files already present under the configured `raw_dir`; it does not download missing images or the full COCO ZIP during quantization:

```bash
bash scripts/quantize_qig.sh \
  --output-dir models/quantized/llava-1.5-7b-qig-w4g128 \
  --samples 128 \
  --seed 42 \
  --method qig \
  --w-bit 4 \
  --w-group 128
```

The output directory must not already exist unless `--overwrite` is passed. Supported `--method` values are `qig`, `mbq`, `awq`, `smoothquant`, `rtn`, and `gptq`, matching the methods exposed by QIG's quantization wrapper. Add `--reweight` or `--distort` for methods that support those options (`qig` and `mbq`). Use `--a-bit`, `--alpha`, and `--percdamp` to set activation precision, SmoothQuant scaling, and GPTQ damping. RTN does not need calibration data; the other methods sample ShareGPT4V COCO. The script uses micro batches of one by default to limit GPU memory use; change this with `--micro-batch-size` if needed. By default, `--model-placement cuda` keeps the model weights on the GPU during layerwise quantization to avoid copying the full model into host RAM. This needs enough VRAM for the model and quantization workspace; use `--model-placement cpu` to select QIG's original CPU offload behavior if VRAM is limited. For QIG, `--model-placement disk` stores decoder layers in a temporary disk cache and loads them one at a time. The cache is removed automatically after export or ordinary failure. It needs free disk space roughly equal to the decoder weights; choose its parent directory with `--offload-dir`. A cache left by a forced process kill is removed on the next disk-mode run using the same parent directory. Disk mode currently does not support `--reweight` and restores the full model to CUDA for export. Timestamped stage and resource logs, plus per-layer activation/scale timings, are printed during the run.

The QIG integration is in `src/attention_quantization/quantization/qig/`. The shared `requirements.txt` pins PyTorch 2.5.1 with CUDA 12.1 and includes the dependencies used by QIG, LLaVA, and the project scripts. Setup installs the QIG, LLaVA-NeXT, and LMMS-Eval packages editable with dependency resolution disabled, so QIG's broader requirements cannot replace the selected PyTorch build. All Python packages use `.venv`.

There is one upstream caveat: QIG's README model list does not advertise classic `llava`, but the checked-in QIG package contains an `llava_v15` processor and its companion LMMS-Eval fork registers a `llava` model adapter. This integration connects those two source paths directly for LLaVA 1.5. The source and dependency versions are recorded in each artifact; the combination still needs a run on the target CUDA machine to confirm runtime compatibility.

QIG's pseudo quantization path saves scale parameters rather than exporting packed low bit weights. This script saves the resulting model checkpoint, method scales when applicable, the sampled calibration JSONL, and run metadata in the chosen directory. For pseudo-quantized weights, model tensors remain floating point, so the research checkpoint does not provide packed INT4 storage savings. The source implementation and its model adapter are documented in the [QIG repository](https://github.com/ucas-xiang/QIG).

Load the saved checkpoint with the shared project environment:

```python
from attention_quantization.models import load_qig_quantized_model

wrapper = load_qig_quantized_model("models/quantized/llava-1.5-7b-qig-w4g128")
model = wrapper._model
tokenizer = wrapper._tokenizer
```

The loader uses QIG's LMMS-Eval `llava` adapter, so call it from `.venv` after running setup.

## Documentation

Planning notes and supporting project documentation are in [`docs/`](docs/). The [project structure reference](docs/project_structure.md) describes the planned organization of the download, quantization, and evaluation scripts and packages.

## License

See [LICENSE](LICENSE) for this repository’s license. Third-party model weights and dataset assets may have separate terms.
