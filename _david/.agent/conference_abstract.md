# Ligand-based de novo drug design

**David Kolář (B3)**
Školitel: Ing. Martin Šícho, Ph.D.

Sometimes a biologically meaningful molecule is discovered that cannot be developed
further because its properties make it unsuitable for real application. Medicinal
chemistry traditionally responds with small structural modifications, but fine-tuning
of this kind is not enough when the required change is not incremental — when we need a
molecule with a substantially different structure that nevertheless reproduces the
properties we care about. Such a search is naturally *ligand-based*: everything we know
is embodied in a single reference molecule, and no series of analogues or activity
labels is available to learn from.

The aim of this work is to build and calibrate exactly that method inside DrugEx, the
institute's own generative framework. A pretrained SMILES-based recurrent neural network
is fine-tuned by reinforcement learning against a multi-objective reward assembled
directly from the properties of the reference ligand rather than from a dataset: 3D
shape and pharmacophore similarity, synthetic accessibility, and optionally drug-likeness.
Because these objectives conflict, the result is not judged by a single score but by how
the whole property distribution shifts relative to the pretrained baseline, with emphasis
on the upper percentiles that contain the candidates actually worth synthesising. We
demonstrate the workflow on epigallocatechin gallate, a polyphenol whose properties are
hard to reproduce from a different scaffold, and report a systematic search for the
hyperparameters that control this trade-off, showing where the generated molecules
improve on the reference properties and where they do not.

---

## Shorter variant (~150 words)

**Ligand-based de novo drug design** — David Kolář (B3), Školitel: Ing. Martin Šícho, Ph.D.

A biologically interesting molecule is often useless in practice because its properties
are unsuitable, and small structural modifications cannot always fix it. We therefore
look for a different structure that keeps the properties we care about — a purely
ligand-based problem, where a single reference molecule is the only input and no
training dataset exists.

This work builds such a method in DrugEx, the institute's generative framework. A
pretrained SMILES-based recurrent neural network is fine-tuned by reinforcement learning
against a multi-objective reward derived from the reference ligand: 3D shape and
pharmacophore similarity, synthetic accessibility, and optionally drug-likeness. Since
these objectives conflict, performance is measured as a shift of the property
distribution against the pretrained baseline, focusing on the upper percentiles of
generated molecules. We calibrate the controlling hyperparameters and report where the
approach succeeds and where it fails.

## Title alternatives

- Ligand-based de novo drug design
- Property-driven de novo design from a single reference ligand
- Optimising molecular properties without a training dataset
