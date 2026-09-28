# Proposed Project Structure

This is a planning reference for organizing the Attention Weighting Quantization project. It separates command-line entry points from reusable implementation and keeps downloaded assets out of version control.

```text
attention_weighting_quantization/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   ├── model.yaml
│   ├── dataset.yaml
│   └── experiments/
├── scripts/
│   ├── download_model.py
│   ├── download_dataset.py
│   ├── inspect_calibration_data.py
│   ├── quantize.py
│   └── investigate.py
├── src/
│   └── attention_quantization/
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

- `scripts/` contains command-line entry points. `download_dataset.py` downloads the ShareGPT4V caption annotation JSON and one selected image source under `data/raw/`; COCO is the default. `inspect_calibration_data.py` samples and previews calibration records.
- `src/attention_quantization/model/` handles model-related setup and downloading.
- `src/attention_quantization/data/` loads ShareGPT4V records, validates image references, samples reproducibly, and exposes each image with its user prompt and assistant caption.
- `src/attention_quantization/quantization/` contains the quantization methods and attention-specific implementation.
- `src/attention_quantization/evaluation/` compares original and quantized model behavior and computes evaluation metrics.
- `scripts/investigate.py` is an entry point for evaluation and analysis. If attention behavior analysis grows into its own area, it can later become `src/attention_quantization/analysis/attention/`.
- `configs/` stores model, dataset, and experiment settings.
- `notebooks/exploration/` is for exploratory work; reusable code should move into `src/`.
- `data/` is a local data location. Do not commit downloaded datasets, model weights, or generated checkpoints to Git.

This layout is a proposal, not a commitment to create all these directories immediately. The exact dataset file, quantization method, metrics, and run environment remain to be specified.
