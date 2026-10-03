import multiprocessing
from abc import ABC, abstractmethod


class ParallelException(Exception):
    pass


class ResultCollector(ABC):

    @abstractmethod
    def __call__(self, result):
        pass

class ListCollector(ResultCollector, ABC):

    @abstractmethod
    def getList(self):
        pass


class ParallelProcessor(ABC):
    """Simple interface to define parameters for parallel processing of data."""

    def __init__(self, n_proc=None, chunk_size=1000, chunks=None):
        """Initialize worker process count and chunk size partitions for multiprocessing."""
        self.nProc = n_proc if n_proc else multiprocessing.cpu_count()
        self.chunks = chunks if chunks else self.nProc
        self.chunkSize = chunk_size

    def getChunkSize(self, data):
        """Determine data chunk partition size based on configured chunk size or process count."""
        if self.chunkSize:
            return self.chunkSize
        else:
            return len(data) // self.chunks

    @abstractmethod
    def apply(self, data, collector):
        """Execute parallel data processing and aggregate results via collector."""
        pass