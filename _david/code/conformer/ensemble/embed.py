"""ETKDGv3 embedding with explicit per-isomer budgets and full accounting."""

from __future__ import annotations

from dataclasses import dataclass
import time

from rdkit import Chem

from _david.code.chemistry import embed_conformers
from _david.code.conformer.core.cache import ArtifactCache
from _david.code.conformer.core.contracts import EmbedSpec
from _david.code.conformer.core.errors import EmbeddingError
from _david.code.conformer.prepare.standardize import PreparedMolecule


@dataclass(frozen=True)
class EmbedResult:
    """Conformer pool produced by one :class:`EmbedSpec`."""

    records: tuple[Chem.Mol, ...]
    spec: EmbedSpec
    isomers_attempted: int
    isomers_embedded: int
    conformers_attempted: int
    failures: tuple[str, ...]
    seconds: float
    cache_key: str | None = None

    @property
    def n_records(self) -> int:
        """Number of retained conformer records."""
        return len(self.records)

    @property
    def ok(self) -> bool:
        """True when at least one finite conformer survived."""
        return bool(self.records)


def _label(record: Chem.Mol, isomer_smiles: str, isomer_index: int, conformer_index: int) -> Chem.Mol:
    """Attach stable provenance properties to one conformer record."""
    record.SetProp("EnumeratedIsomerSMILES", isomer_smiles)
    record.SetIntProp("EnumeratedIsomerIndex", int(isomer_index))
    record.SetIntProp("ConformerIndex", int(conformer_index))
    record.SetProp("_Name", f"iso{isomer_index}_conf{conformer_index}")
    return record


def _embed_uncached(prepared: PreparedMolecule, spec: EmbedSpec) -> EmbedResult:
    """Run ETKDGv3 for each retained stereoisomer and collect finite records."""
    started = time.perf_counter()
    records: list[Chem.Mol] = []
    failures: list[str] = []
    isomers_embedded = 0
    conformers_attempted = 0

    for isomer_index, isomer in enumerate(prepared.isomers):
        isomer_smiles = (
            isomer.GetProp("EnumeratedIsomerSMILES")
            if isomer.HasProp("EnumeratedIsomerSMILES")
            else Chem.MolToSmiles(isomer, isomericSmiles=True)
        )
        mol_h = Chem.AddHs(Chem.Mol(isomer))
        conformers_attempted += spec.attempts
        try:
            conf_ids = embed_conformers(
                mol_h,
                num_conformers=spec.attempts,
                random_seed=spec.seed,
                embedding_rmsd=spec.rmsd,
                timeout=int(spec.timeout),
                num_threads=spec.threads,
            )
        except (RuntimeError, ValueError) as exc:
            failures.append(f"isomer {isomer_index}: {exc}")
            continue
        if not conf_ids:
            failures.append(f"isomer {isomer_index}: ETKDGv3 returned zero conformers")
            continue
        isomers_embedded += 1
        for conformer_index, conf_id in enumerate(conf_ids):
            record = Chem.Mol(mol_h, confId=conf_id)
            records.append(_label(record, isomer_smiles, isomer_index, conformer_index))

    elapsed = time.perf_counter() - started
    if not records:
        detail = "; ".join(failures) or "no stereoisomers"
        raise EmbeddingError(f"Embedding produced no finite conformers ({detail})")
    return EmbedResult(
        records=tuple(records),
        spec=spec,
        isomers_attempted=len(prepared.isomers),
        isomers_embedded=isomers_embedded,
        conformers_attempted=conformers_attempted,
        failures=tuple(failures),
        seconds=elapsed,
    )


def embed_ensemble(
    prepared: PreparedMolecule,
    spec: EmbedSpec | None = None,
    *,
    cache: ArtifactCache | None = None,
) -> EmbedResult:
    """Embed a conformer pool, reusing a cached pool when the spec matches.

    The cache key includes the prepared identity, the full embedding spec, and
    the package algorithm version.  Cached records are re-read from a single SDF
    and still carry every provenance property set at write time.
    """
    spec = spec or EmbedSpec()
    payload = {
        "molecule": prepared.identity,
        "isomers": list(prepared.isomer_smiles),
        "embed": spec,
    }
    key = cache.key("embed", payload) if cache is not None else None
    if cache is not None and key is not None:
        cached = cache.get_sdf("embed", key)
        if cached is not None:
            meta = cache.read_meta("embed", key).get("metadata", {})
            return EmbedResult(
                records=tuple(cached),
                spec=spec,
                isomers_attempted=int(meta.get("isomers_attempted", len(prepared.isomers))),
                isomers_embedded=int(meta.get("isomers_embedded", len(prepared.isomers))),
                conformers_attempted=int(meta.get("conformers_attempted", 0)),
                failures=tuple(str(value) for value in meta.get("failures", [])),
                seconds=float(meta.get("seconds", 0.0)),
                cache_key=key,
            )

    result = _embed_uncached(prepared, spec)
    if cache is not None and key is not None:
        cache.put_sdf(
            "embed",
            key,
            result.records,
            metadata={
                "isomers_attempted": result.isomers_attempted,
                "isomers_embedded": result.isomers_embedded,
                "conformers_attempted": result.conformers_attempted,
                "failures": list(result.failures),
                "seconds": result.seconds,
                "molecule": prepared.spec.name,
            },
        )
        return EmbedResult(**{**result.__dict__, "cache_key": key})
    return result


__all__ = ["EmbedResult", "embed_ensemble"]
