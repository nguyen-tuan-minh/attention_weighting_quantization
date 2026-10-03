"""Load the QIG runtime without making it a dependency of the main package."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QIGRuntime:
    """QIG entry points used by the project's quantization workflow."""

    get_model: Any
    get_process_model: Any
    qwrapper: Any
    get_multimodal_calib_dataset: Any


def load_qig_runtime(source_dir: str | Path) -> QIGRuntime:
    """Import QIG and its LLaVA helpers from an existing source checkout.

    QIG-specific imports are delayed so dataset and model tools remain
    importable before the selected source checkouts are installed.
    """
    source = Path(source_dir).expanduser().resolve()
    if not (source / "main_quant.py").is_file():
        raise FileNotFoundError(f"QIG source was not found: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    try:
        from lmms_eval.models import get_model
        from qmllm.calibration.coco_vl import get_multimodal_calib_dataset
        from qmllm.models import get_process_model
        from qmllm.quantization.quant_wrapper import qwrapper
    except ImportError as error:
        raise ImportError(
            "QIG runtime dependencies are unavailable. Run scripts/set_up.sh "
            "to install the project-selected dependencies in .venv."
        ) from error

    return QIGRuntime(
        get_model=get_model,
        get_process_model=get_process_model,
        qwrapper=qwrapper,
        get_multimodal_calib_dataset=get_multimodal_calib_dataset,
    )
