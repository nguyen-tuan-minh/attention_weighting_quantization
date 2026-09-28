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
│   ├── quantize.py
│   └── investigate.py
├── src/
│   └── attention_quantization/
│       ├── model/
│       │   └── download.py
│       ├── data/
│       │   ├── download.py
│       │   └── coco_sharegpt4v.py
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

- `scripts/` contains command-line entry points. These call reusable code from `src/`.
- `src/attention_quantization/model/` handles model-related setup and downloading.
- `src/attention_quantization/data/` handles dataset download, organization, and loading for the COCO images and ShareGPT4V captions.
- `src/attention_quantization/quantization/` contains the quantization methods and attention-specific implementation.
- `src/attention_quantization/evaluation/` compares original and quantized model behavior and computes evaluation metrics.
- `scripts/investigate.py` is an entry point for evaluation and analysis. If attention behavior analysis grows into its own area, it can later become `src/attention_quantization/analysis/attention/`.
- `configs/` stores model, dataset, and experiment settings.
- `notebooks/exploration/` is for exploratory work; reusable code should move into `src/`.
- `data/` is a local data location. Do not commit downloaded datasets, model weights, or generated checkpoints to Git.

This layout is a proposal, not a commitment to create all these directories immediately. The exact dataset file, quantization method, metrics, and run environment remain to be specified.
