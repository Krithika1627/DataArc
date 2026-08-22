from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Shared interface for all DataArc pipeline agents."""

    @abstractmethod
    def run(self, state: dict) -> dict:
        ...
