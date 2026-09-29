# Proposed Project Structure

This is a planning reference for organizing the Attention Weighting Quantization project. It separates command-line entry points from reusable implementation and keeps downloaded assets out of version control.

```text
attention_weighting_quantization/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── model.yaml
│   ├── dataset.yaml
│   └── experiments/
├── scripts/
│   ├── download_model.py
│   ├── prepare_dataset.py
│   ├── quantize.py
│   └── investigate.py
├── src/
│   └── attention_quantization/
│       ├── __init__.py
│       ├── model/
│       │   └── download.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── sharegpt4v.py
│       ├── quantization/
│       │   ├── methods.py
│       │   └── attention.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── compare.py
│       └── utils/
│           ├── paths.py
│           └── logging.py
├── notebooks/
│   └── exploration/
├── tests/
└── data/                 # local only; keep downloaded data out of Git
```

## Responsibilities

- `src/attention_quantization/data/sharegpt4v.py` can be run as a module to download one image source (COCO by default), and provides the ShareGPT4V loader and image download functions.
- `scripts/prepare_dataset.py` loads ShareGPT4V, filters records by source (COCO by default), optionally samples records, and saves the result under the configured processed data directory.
- `src/attention_quantization/model/` handles model-related setup and downloading.
- `src/attention_quantization/data/sharegpt4v.py` exposes the ShareGPT4V loader and related image downloader. By default, the loader downloads COCO, filters to COCO records, and returns images as a lazy feature that decodes them on access. `scripts/prepare_dataset.py` optionally samples and saves the selected data.
- `src/attention_quantization/quantization/` contains the quantization methods and attention-specific implementation.
- `src/attention_quantization/evaluation/` compares original and quantized model behavior and computes evaluation metrics.
- `scripts/investigate.py` is an entry point for evaluation and analysis. If attention behavior analysis grows into its own area, it can later become `src/attention_quantization/analysis/attention/`.
- `configs/dataset.yaml` stores dataset locations as paths relative to the repository root. `configs/` also stores model and experiment settings.
- `notebooks/exploration/` is for exploratory work; reusable code should move into `src/`.
- `data/` is a local data location. Do not commit downloaded datasets, model weights, or generated checkpoints to Git.

This layout is a proposal, not a commitment to create all these directories immediately. The exact dataset file, quantization method, metrics, and run environment remain to be specified.
