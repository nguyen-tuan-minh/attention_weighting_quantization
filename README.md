# Attention Weighting Quantization

## Project overview

The project studies how quantizing attention weights affects LLaVA 1.5 7B. The specific method, quantization settings, and evaluation questions will be documented here as they are finalized.

## Model and data

- **Model:** LLaVA 1.5 7B. See the [official LLaVA repository](https://github.com/haotian-liu/LLaVA) for model setup and licensing details.
- **Dataset:** COCO images paired with captions from the ShareGPT4V project. ShareGPT4V releases multiple data files and subsets; this project’s precise COCO split/file and preprocessing are to be specified before running experiments. See [ShareGPT4V data documentation](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md) and the [dataset page](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V).

Please follow the upstream dataset and model terms when downloading or using these resources. The data and pretrained model weights are not included in this repository.

### Downloading captions and image data

The download script stores the ShareGPT4V caption annotations, image archives, and extracted images under `data/raw/`. By default it downloads the ShareGPT4V GPT-4V caption JSON and COCO `train2017`. Other supported image sources can be selected individually:

```bash
python scripts/download_dataset.py                  # ShareGPT4V captions + COCO train2017
python scripts/download_dataset.py --dataset gqa
python scripts/download_dataset.py --dataset textvqa
python scripts/download_dataset.py --dataset visual-genome
```

The caption file is saved to `data/raw/sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json`. The downloader only retrieves source data. To create and inspect a deterministic COCO calibration sample from the assistant captions, run:

```bash
python scripts/inspect_calibration_data.py --samples 128 --seed 42
```

The data interface returns RGB images, the associated user prompt, and the assistant caption. The caption is preserved as the calibration target. Fixed-prompt generation evaluation is a separate later step. ShareGPT4V also lists sources with separate or restricted download steps; see its [data instructions](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md).

## Documentation

Planning notes and supporting project documentation are in [`docs/`](docs/). The [proposed project structure](docs/project_structure.md) describes the planned organization of the download, quantization, and evaluation modules.

## License

See [LICENSE](LICENSE) for this repository’s license. Third-party model weights and dataset assets may have separate terms.
