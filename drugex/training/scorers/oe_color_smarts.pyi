"""
OpenEye ImplicitMillsDean color force field, ported to self-contained SMARTS.

PROVENANCE (verbatim source of every SMARTS below):
    ccr2_gen/oeye/current_apps/apps/openeye/data/rocs/3.9.0.1/rocs/ImplicitMillsDean.cff

That file declares 6 color INTERACTION types -- donor, acceptor, cation, anion,
rings, hydrophobe -- each ``attractive gaussian weight=1.0 radius=1.0`` (.cff lines
185-190). Each type's PATTERN lines (.cff lines 147-183) reference ``$macro`` names
bound in the DEFINE block (.cff lines 40-131). OpenEye's SMARTS dialect allows a
``$macro`` to stand for a previously-defined sub-SMARTS; RDKit/CDPKit do not. This
module reproduces the 6 types by expanding every ``$macro`` inline into a
self-contained recursive SMARTS ``$(...)`` so the patterns compile unchanged in both
RDKit (``Chem.MolFromSmarts``) and CDPKit (``Chem.parseSMARTS``).

Why port it: the open-source color channels do NOT match OpenEye out of the box.
RDKit's rdShapeAlign default color is a 5-pattern fallback with NO hydrophobe; RDKit's
BaseFeatures.fdef treats only AROMATIC atoms as "rings" (so saturated rings are missed)
and its LumpedHydrophobe under-fires on alkyl chains; CDPKit's DefaultPharmacophoreGenerator
uses its own perception. The CCR2 references are hydrophobe- and ring-heavy, so these gaps
bias the color score. Using OE's own atom typing closes the gap.

OPT-IN: this module is inert until a scorer is constructed with ``color_ff="oe_mills_dean"``.

pH / protonation semantics (verbatim from the .cff, NOT invented here):
    The ImplicitMillsDean variant is *implicit*: the acid/base macros it would use for
    explicit protonation states (``ACprimaryAmine``, ``DacidOH``, ...) are commented out
    in the DEFINE block, and ``CATamine``/``anion`` are written to match the NEUTRAL
    (un-ionized) forms. Hence ``cation`` fires on a neutral amine and ``anion`` fires on a
    neutral carboxylic/sulfonic/phosphonic acid -- and, because the heavy-atom skeleton is
    unchanged, on their (de)protonated forms too. This makes the color typing invariant to
    the input protonation state (verified in tests_oe_color_ff.ProtonationInvariance).

.cff macros NOT expanded here (because the .cff itself disables them -- they appear only in
commented-out DEFINE lines and never reach a live PATTERN): ACprimaryAmine, ACtertiaryAmine,
ACsecondaryAmine (L61/66/74), DacidOH (L95). They are intentionally omitted; no live pattern
references them. Everything that a live PATTERN transitively needs IS expanded.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

import re
from typing import Any

_DEFINE: dict[str, str]
_RAW_PATTERNS: dict[str, list[str]]
_SIX_TYPES: Any = ...
_MACRO_REF: Any = ...
OE_MILLS_DEAN_IMPLICIT: dict[str, list[str]]
OE_MILLS_DEAN_PATTERNS: dict[str, list[Chem.Mol]]
OE_TO_CDPKIT_TYPE: dict[str, tuple[int, int]]
_CDPKIT_POS_REF_FLAG: Any = ...

def _expand(smarts: str, _stack: Tuple[str, ...] = ...) -> str:
    """
    Recursively replace every ``$macro`` in ``smarts`` with its inline recursive SMARTS.

    OE writes ``$name`` for "an atom/group matching macro ``name``"; the RDKit/CDPKit spelling
    of that inside a SMARTS is a recursive query ``$( <expanded body> )``. We therefore wrap
    each macro body in ``$( ... )`` on substitution. A bracketed-atom macro body like
    ``[X1H0,...]`` becomes ``$([X1H0,...])``; a multi-atom body like ``php`` becomes
    ``$([...])`` too -- RDKit accepts both. Cycles raise (the .cff DEFINE order is acyclic).
    """

    ...


def _build_expanded() -> Dict[str, list[str]]:
    """Expand every PATTERN body and verify it compiles under RDKit. Raise on any failure."""

    ...


def _cdpkit_type_map() -> Dict[str, tuple[int, int]]:
    """
    Map each OE color name -> (CDPKit FeatureType, FeatureGeometry).

    Color features are positional spheres in ROCS-style overlay, so geometry is SPHERE for
    all six. Returns plain ints (the CDPKit enum values) so the map is importable even when
    CDPKit is absent; the values equal CDPL.Pharm.FeatureType.* / FeatureGeometry.SPHERE.
    """

    ...


def build_cdpkit_oe_pharm_generator() -> Any:
    """
    Build a CDPKit DefaultPharmacophoreGenerator configured for OE ImplicitMillsDean.

    All of CDPKit's own default feature perceptions are disabled, then for each OE color
    type a PatternBasedFeatureGenerator is installed whose include-patterns are the type's
    OE SMARTS.

    Returns
    -------
    Pharm.DefaultPharmacophoreGenerator
        Configured CDPKit pharmacophore generator perceiving the 6 OE color types.

    Raises
    ------
    ImportError
        If CDPKit is not installed.
    """

    ...

