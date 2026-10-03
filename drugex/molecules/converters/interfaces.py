from abc import ABC, abstractmethod


class ConversionException(Exception):
    """Raised when a molecule conversion operation fails."""
    pass


class MolConverter(ABC):
    """Abstract callable interface for converting molecules between representations."""

    @abstractmethod
    def __call__(self, representation):
        pass