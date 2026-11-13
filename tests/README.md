# Test organization

Keep inspector-specific tests in a dedicated package for each model family:

```text
tests/
└── inspection/
    ├── shared/       # Storage, reductions, primitive/composite contracts
    ├── llava/        # LLaVA targets, sessions, and aggregate measurements
    └── <model>/      # Tests and fixtures for each future inspector
```

Use small deterministic CPU fixtures for unit tests. Model-specific fixtures
belong in their inspector's folder; keep shared tests independent of concrete
model layouts. Include `__init__.py` in each test subdirectory so unittest
discovery recurses into it. Tests must follow the function documentation and
state conventions in [CONVENTIONS.md](../CONVENTIONS.md).

From the repository root, in a Python environment with PyTorch installed:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m unittest discover -s tests/inspection/llava -v
PYTHONPATH=src python -m unittest discover -s tests/inspection/shared -v
```

The unit suite requires no downloaded models, datasets, GPU, or network access.
Real checkpoint and hardware-specific checks should be kept separate from these
unit tests.
