# Attention Weighting Quantization

## Project overview

The project studies how quantizing attention weights affects LLaVA 1.5 7B. The specific method, quantization settings, and evaluation questions will be documented here as they are finalized.

## Model and data

- **Model:** LLaVA 1.5 7B. See the [official LLaVA repository](https://github.com/haotian-liu/LLaVA) for model setup and licensing details.
- **Dataset:** COCO images paired with captions from the ShareGPT4V project. ShareGPT4V releases multiple data files and subsets; this project’s precise COCO split/file and preprocessing are to be specified before running experiments. See [ShareGPT4V data documentation](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md) and the [dataset page](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V).

Please follow the upstream dataset and model terms when downloading or using these resources. The data and pretrained model weights are not included in this repository.

## License

See [LICENSE](LICENSE) for this repository’s license. Third-party model weights and dataset assets may have separate terms.
