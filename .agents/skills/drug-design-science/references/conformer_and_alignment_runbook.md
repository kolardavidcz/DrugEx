# Conformers and alignment: inspect before choosing

Use `consult-literature` before selecting force fields, dielectric treatments, conformer counts, clustering thresholds, or scoring changes. Inspect `_david/code/conformer/` for standalone studies, `_david/code/environment.py` for RL reference preparation/scorer construction, and `drugex/training/scorers/conformer_generators.py` for the actual candidate generator. These paths need not implement the same protocol.

## Dielectric parameters are hypotheses to validate

RDKit [MMFFMolProperties documentation](https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html) defines dielectric model 1 as constant and model 2 as distance-dependent. The default dielectric constant is 1.0. Inspect the installed version and force-field construction, rather than inferring physical behavior from labels such as "Pocket" or "Water".

Changing a dielectric parameter changes an electrostatic model. By itself it does not implement explicit solvent or establish a validated continuum-solvation free energy. Treat epsilon 6.5 and 78.4 in the existing variant factory as experimental settings, not a universal protein-pocket consensus or a guarantee of physiological conformations. A vacuum calculation can be a useful baseline; its suitability for a particular inference needs evidence. The earlier absolute prohibition and the fixed K_ref=3/K_gen=10 recommendations were unsupported.

Compare protocols with consistent embedding budgets, seeds, stereochemistry/protonation, pruning, optimization convergence, and selection criteria. If energy windows differ between variants, changing both screening and energy filtering confounds a causal conclusion about screening alone. Track failures and actual embedded/retained counts. Do not silently replace a failed force field with dummy energies.

## Validate the decision in its intended task

EGCG self-recovery measures recovery of the chosen EGCG ensemble under a particular scorer. It does not by itself establish bioactive-conformation coverage, solvent accuracy, generality to generated candidates, or a production conformer budget. Use independent seeds, representative diverse molecules, matching scorer settings, and repeated timings when proposing a comparison. Where appropriate, use experimental ligand conformations as an additional reference while accounting for target/structure context.

Report requested and actual K, coverage, score distributions, invalid/failure fractions, diversity, and latency. A study's total duration is not per-candidate latency. Distinguish raw TanimotoCombo (normally the sum of shape and color terms, nominally 0–2 for the stated scorer) from modified reward/desirability scales; inspect the implementation's score variant and transformations.

## Verify configuration wiring

Trace parameters into the live constructor. As inspected on 2026-10-27, `create_rocs_scorer()` accepts `timeout` but does not pass it to `RDKitConformerGenerator`, whose constructor defaults to 20 seconds. Therefore `timeout_seconds=0` in config does not establish unlimited generation. Recheck current code before changing it; the user's active benchmark work can alter this path.

The earlier claims of +152% recovery, >98% self-recovery, and 3x throughput are not adopted recommendations. Require their raw measurements, denominator, configuration, and revision before reusing them. No experiment or RL setting is changed by this runbook.
