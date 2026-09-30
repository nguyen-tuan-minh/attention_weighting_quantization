"""Lazy bridge to the external QIG source implementation."""

from .runtime import QIGRuntime, load_qig_runtime

__all__ = ["QIGRuntime", "load_qig_runtime"]
