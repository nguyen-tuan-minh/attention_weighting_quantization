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

If the setup script is already running inside the repository's `.venv`, it exits without changing anything. Check which compute devices PyTorch can use with:

```bash
python scripts/check_device.py
```

Dataset locations are set in [`configs/dataset.yaml`](configs/dataset.yaml). Relative paths there resolve from the repository root. The ShareGPT4V module downloads image archives and extracts them under the configured `raw_dir`; COCO `train2017` is the default:

```bash
python -m attention_quantization.data.sharegpt4v
python -m attention_quantization.data.sharegpt4v --dataset gqa
```

The data module defaults to COCO. Calling the loader downloads the COCO archive if needed, loads the ShareGPT4V records, and connects their image paths to the local files:

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

The original forward-only script has three sections: load model, prepare calibration, and forward:

```bash
python scripts/run_calibration.py
```

The attention investigation script has two modes. The default `online` mode computes a separate IGA map for every layer, using non-padding text queries after the image tokens. It opens one Matplotlib figure per sample with the processor image and all per-layer overlays together. It does not save attention maps. The `save` mode writes each captured language attention matrix under `attention_output_dir` for later analysis. The script is divided into five sections: load model, prepare calibration, register hooks, forward, and analyse.

```bash
python scripts/investigate_attention.py                 # online analysis
python scripts/investigate_attention.py --mode save     # save attention maps
```

Both scripts forward each image with its ShareGPT4V user prompt and assistant caption. Images are decoded one at a time. Override the configured sample count or seed with `--samples` and `--seed`.

## Documentation

Planning notes and supporting project documentation are in [`docs/`](docs/). The [project structure reference](docs/project_structure.md) describes the planned organization of the download, quantization, and evaluation scripts and packages.

## License

See [LICENSE](LICENSE) for this repository’s license. Third-party model weights and dataset assets may have separate terms.
