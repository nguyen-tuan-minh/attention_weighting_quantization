"""Lifecycle contract shared by primitive and composite collectors."""
from abc import ABC, abstractmethod


class Collector(ABC):
    @abstractmethod
    def start(self, publish):
        """Attach collection resources and publish extracted measurements."""

    @abstractmethod
    def close(self):
        """Release resources; safe to call repeatedly."""
