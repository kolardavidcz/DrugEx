from abc import ABC, abstractmethod

from drugex.logs import logger
from drugex.molecules.converters.interfaces import ConversionException

class Molecule(ABC):
    """Abstract base class representing a standardized chemical entity with annotations and unique identifiers."""

    @abstractmethod
    def annotate(self, key, value):
        pass

    @abstractmethod
    def getAnnotation(self, key):
        pass

    @abstractmethod
    def getMetadata(self):
        pass

    def __eq__(self, other):
        return self.getUniqueID() == other.getUniqueID()

    def __hash__(self):
        return hash(self.getUniqueID())

    def __str__(self):
        return f"{self.__class__} (ID: {self.getUniqueID()})"

    @abstractmethod
    def getUniqueID(self):
        pass

    @abstractmethod
    def asRDKit(self):
        pass

class ItemException(Exception):
    pass

class MolSupplier(ABC):
    """Abstract generator interface supplying molecules item-by-item with optional conversion and exception filtering."""

    def __iter__(self):
        return self

    def __next__(self):
        next_item = None
        while not next_item:
            try:
                next_item = self.next()
                next_item = self.convert(next_item)
            except ItemException as exp:
                logger.warning(f"Failed to generate next item in {repr(self)}\n\t Cause: {repr(exp)}")
                next_item = None
            except ConversionException as exp:
                logger.warning(f"Failed to convert item {next_item} to the new representation in {repr(self)}\n\t Cause: {repr(exp)}")
                next_item = None

        return next_item

    @abstractmethod
    def next(self):
        """Fetch the next molecule representation item or raise StopIteration when exhausted."""
        pass

    def convert(self, representation):
        """Convert the supplied raw representation to the target molecule format (identity by default)."""
        return representation

    def toList(self):
        return [x for x in self]


class BaseMolSupplier(MolSupplier, ABC):
    """Molecule supplier wrapping a converter to emit Molecule instances with optional deduplication."""

    def __init__(
            self,
            converter,
            hide_duplicates=False
    ):
        """Initialize base supplier with target molecule converter and deduplication flag."""
        super().__init__()
        self.converter = converter
        self.hide_duplicates = hide_duplicates
        self._prev_ids = set()

    def __next__(self):
        mol = super().__next__()
        if not mol:
            return next(self)

        if self.hide_duplicates:
            id_ = mol.getUniqueID()
            if id_ in self._prev_ids:
                logger.info(f"Molecule with ID '{id_}' was skipped because it was already encountered.")
                return next(self)
            else:
                self._prev_ids.add(id_)

        return mol

    def convert(self, representation):
        """Apply converter to raw representation, raising ConversionException if output is empty."""
        ret = self.converter(representation)
        if not ret:
            raise ConversionException(f"Converter returned an empty molecule instance for representation: {representation}")
        return ret