import concurrent
from tqdm.auto import tqdm

from drugex.parallel.interfaces import ParallelException, ParallelProcessor

class ParallelSupplierEvaluator(ParallelProcessor):
    """Parallel evaluation manager executing MolSupplier instances across chunked data partitions."""

    def __init__(self, supplier_class, n_proc=None, chunk_size=1000, chunks=None, args=None, kwargs=None):
        """Initialize parallel supplier evaluator with supplier class template and multiprocessing configuration."""
        super().__init__(n_proc, chunk_size, chunks)
        self.args = [] if not args else args
        self.kwargs = dict() if not kwargs else kwargs
        self.supplier = supplier_class
        self.result = []
        self.errors = []

    def initSupplier(self, supplier_class, chunk):
        """Instantiate MolSupplier class for a specific data chunk."""
        return supplier_class(chunk, *self.args, **self.kwargs)

    def run(self, chunk, error):
        """Execute MolSupplier extraction on a data chunk with exception routing."""
        try:
            sup = self.initSupplier(self.supplier, chunk)
            return sup.toList(), sup
        except Exception as exp:
            error(exp)

    def error(self, data):
        """Record error data and raise ParallelException during worker execution."""
        self.errors.append(data)
        raise ParallelException(data)

    def apply(self, data, collector, error=None, desc_string=None):
        """Distribute chunked supplier evaluation across ProcessPoolExecutor workers and collect results."""

        error = error if error else self.error
        chunk_size = self.getChunkSize(data)
        data = [data[i: i+chunk_size] for i in range(0, len(data), chunk_size)]
        batch_size = self.nProc
        results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nProc) as executor:
            batches = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
            for batch in tqdm(batches, desc=f"{self.supplier.__name__ if not desc_string else desc_string} (batch processing)"):
                for result in executor.map(self.run, batch, len(batch) * [error]):
                    results.append(collector(result))

        return results