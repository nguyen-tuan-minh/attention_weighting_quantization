"""Storage policies for completed captures."""
from pathlib import Path
import torch


class Storage:
    """Store tensors while leaving returned tensors/files owned by the caller."""

    def __init__(self, mode="cpu", directory=None, dtype=None):
        """Validate live/CPU/disk policy; disk requires a caller-owned directory."""
        if mode not in {"cpu", "live", "disk"}:
            raise ValueError("storage must be cpu, live, or disk")
        if mode == "disk" and directory is None:
            raise ValueError("disk storage requires an explicit directory")
        self.mode = mode
        if mode == "live" and dtype is not None:
            raise ValueError("dtype conversion requires detached CPU or disk storage")
        self.dtype = dtype
        self.directory = Path(directory) if directory is not None else None

    def save(self, tensor):
        """Return the live tensor, detached CPU copy, or newly written .pt path.

        Preserves shape and uses the requested dtype for detached storage, or
        the source dtype when none was requested. Disk mode creates directories as needed and
        never overwrites or removes existing captures. Filesystem errors propagate.
        """
        if self.mode == "live":
            return tensor
        tensor = tensor.detach().to(device="cpu", dtype=self.dtype or tensor.dtype, copy=True)
        if self.mode == "cpu":
            return tensor
        self.directory.mkdir(parents=True, exist_ok=True)
        # Exclusive file creation prevents overwriting existing captures.
        import uuid
        path = self.directory / f"{uuid.uuid4().hex}.pt"
        with path.open("xb") as stream:
            torch.save(tensor, stream)
        return path
