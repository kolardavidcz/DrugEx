## 2026-10-03T17:06:57Z

You are a Teamwork Worker implementing Milestone M1: Data Preprocessing, Molecular Graphs & Parallel Processing (Sections 1 & 6 of DATABASE).

## Your Identity & Working Directory
- TypeName: teamwork_preview_worker
- Working Directory: /home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_worker_m1_1
- Parent Orchestrator Conversation ID: a3436e50-05ee-4060-82fe-47da0f7d1667

## Mandatory First Step
Read /home/kolar/build_projects/drugex_edu/.agents/teamwork/ORIGINAL_REQUEST.md in full before starting any work.

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Context & Survey Findings
Read the authoritative survey reports:
1. Codebase Architecture & Discrepancies:
   `/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_explorer_survey_2/handoff.md`
2. Algorithmic, Mathematical & Tensor Rigor:
   `/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_explorer_survey_3/handoff.md`
3. Global Project Blueprint:
   `/home/kolar/build_projects/drugex_edu/.agents/teamwork/PROJECT.md`
4. Ground Truth Code in:
   `/home/kolar/build_projects/drugex_edu/repo/drugex/data/`
   `/home/kolar/build_projects/drugex_edu/repo/drugex/molecules/`
   `/home/kolar/build_projects/drugex_edu/repo/drugex/parallel/`

## Tasks & File Ownership
You exclusively own the Section 1 and Section 6 symbol entries in `/home/kolar/build_projects/drugex_edu/docs/index.html` (the `DATABASE` JavaScript object):
- Symbols in Section 1 (Data Preprocessing & Molecular Representation):
  - Tokenizers & Vocabularies: `drugex.data.corpus.vocabulary.VocSmiles`, `VocGraph`, `VocNonGPT`, `SequenceCorpus`, `Vocabulary`, `SequenceVocabulary`, `interfaces.Corpus`.
  - Datasets & Data Loaders: `drugex.data.datasets.SmilesDataSet`, `SmilesFragDataSet`, `GraphFragDataSet`, `TargetCreator`, `interfaces.DataSet`, `DataToLoader`, `DataSplitter`, `RandomTrainTestSplitter`.
  - Fragmenters & Encoders: `drugex.data.fragments.Fragmenter`, `FragmentCorpusEncoder`, `FragmentPairs`, `FragmentPairEncoder`, `FragmenterWithSelectedFragment`, `dummyMolsFromFragments`.
  - Molecular representations & standardizers: `drugex.molecules.mol.DrExMol`, `DefaultStandardizer`, `CleanSMILES`, `SmilesToDrEx`, `DrExToSMILES`, `RDKitToDrEx`, `ListSupplier`, `StandardizedSupplier`, `MolSupplier`, `MolConverter`.
- Symbols in Section 6 (Parallel Processing & High-Throughput Execution):
  - `drugex.parallel.evaluator.ParallelSupplierEvaluator`, `collectors.ListExtend`, `ListAppend`, `interfaces.ParallelProcessor`, `ResultCollector`, `ListCollector`.

For each symbol:
1. Audit and ground against `repo/drugex` Python code: exact synopsis signature, constructor parameters, return types, member methods (regular before magic dunder), and instance variables. Use precise runtime type annotations (`torch.Tensor`, `pd.DataFrame`, `np.ndarray`, `Chem.Mol`, `DrExMol`, etc.).
2. Elevate descriptions, notes, and parameters to publication-grade academic rigor (bachelor thesis level in ligand-based de novo drug design):
   - Explain computational and algorithmic mechanics (tokenization algorithms, vocabulary encoding matrices, tensor indexing, padding masks, fragment combination enumeration, multiprocessing chunking).
   - Document exact tensor/matrix dimensions (e.g. `(batch_size, seq_len)`, `(N, M + 4)`, `(batch_size, num_nodes, num_features)`), boundary conditions, valid numeric intervals, default values, and failure modes (invalid SMILES, unparseable graphs, empty inputs).
3. Ensure every symbol's `example.code` and `example.output` represents idiomatic, authentic, executable DrugEx 3.4+ code.
4. Maintain 100% graph integrity in `seeAlso` arrays (zero broken cross-links).
5. Strictly adhere to the Zero-Emoji Rule.
6. Verify compliance by running `node tools/check_docs.mjs` (from `docs/`) and ensuring all assertions pass with 0 errors.
7. Stage and commit changes locally with a descriptive commit message (`git add docs/index.html && git commit -m "docs: elevate Section 1 and 6 data, molecule, and parallel symbols to publication grade (M1)"`).

## User Rules & Constraints
- Direct file editing tools only (no ad-hoc modification scripts).
- Zero broken tests/lints.
- Zero emoji characters.

## Output Requirements
- Maintain progress.md in your working directory with regular timestamped heartbeats.
- Write your comprehensive completion report to `/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_worker_m1_1/handoff.md`.
- Send a completion message via send_message to your parent (a3436e50-05ee-4060-82fe-47da0f7d1667).
