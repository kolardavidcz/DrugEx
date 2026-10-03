from drugex.molecules.converters.default import SmilesToDrEx
from drugex.molecules.interfaces import BaseMolSupplier


class StandardizedSupplier(BaseMolSupplier):
    """Supplier yielding standardized molecules from an input iterator via a standardizer converter."""

    def __init__(self, mols, standardizer):
        """Initialize standardized supplier with molecule collection and standardization converter."""
        super().__init__(converter=standardizer)
        self.mols = mols if hasattr(mols, "__next__") else iter(mols)

    def next(self):
        """Return the next raw molecule item for standardization processing."""
        return next(self.mols)

class ListSupplier(BaseMolSupplier):
    """Supplier iterating over an in-memory molecule sequence, converting each item to DrExMol."""

    def __init__(self, mols, converter=SmilesToDrEx(), hide_duplicates=False):
        """Initialize list supplier with molecule collection, converter, and deduplication flag."""
        super().__init__(converter=converter, hide_duplicates=hide_duplicates)
        self.mols = iter(mols)


    def next(self):
        """Return the next raw molecule item from the list iterator."""
        return next(self.mols)