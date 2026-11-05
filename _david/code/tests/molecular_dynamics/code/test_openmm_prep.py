"""Diagnostic test for OpenMM platform and GAFF small-molecule parameterization."""
from __future__ import annotations

import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import openmm
from openmm import app, unit
from openmmforcefields.generators import GAFFTemplateGenerator

print("OpenMM Version:", openmm.__version__)
print("Platforms available:")
for i in range(openmm.Platform.getNumPlatforms()):
    p = openmm.Platform.getPlatform(i)
    print(f"  {i}: {p.getName()}")

from openmm import System, VerletIntegrator, Context, Platform

import parmed as pmd
from openbabel import pybel

smiles = "COc1ccc(C2Oc3ccccc3CC2OC(=O)c2ccccc2)cc1"
ob_mol = pybel.readstring("smi", smiles)
ob_mol.addh()
ob_mol.make3D()
ob_mol.write("mol2", "/tmp/test_lig.mol2", overwrite=True)

import openmmforcefields
from pathlib import Path

ff_xml = Path(openmmforcefields.__file__).parent / "ffxml" / "amber" / "gaff" / "gaff-2.11.xml"
if not ff_xml.exists():
    # search glob
    matches = list(Path(openmmforcefields.__file__).parent.glob("**/gaff-2.11.xml"))
    if matches:
        ff_xml = matches[0]

# Let's map OpenBabel atom types to GAFF types:
# OpenBabel: C3 -> c3, Car -> ca, O3 -> os, O2 -> o, etc.
# GAFF atom types: lowercase.
def get_gaff_type(ob_type: str) -> str:
    mapping = {
        "C3": "c3", "C2": "c2", "Car": "ca", "C1": "c1",
        "O3": "os", "O2": "o", "S3": "ss", "S2": "s2",
        "N3": "n3", "N2": "n2", "Npl": "na", "Nam": "n",
        "H": "ha", "HO": "ho", "HN": "hn"
    }
    return mapping.get(ob_type, ob_type.lower())

xml_lines = ['<ForceField>', ' <Residues>', '  <Residue name="LIG">']
for a in ob_mol.atoms:
    gtype = get_gaff_type(a.type)
    xml_lines.append(f'   <Atom name="A{a.idx}" type="{gtype}" charge="{a.partialcharge:.4f}"/>')

for b in pybel.ob.OBMolBondIter(ob_mol.OBMol):
    xml_lines.append(f'   <Bond atomName1="A{b.GetBeginAtomIdx()}" atomName2="A{b.GetEndAtomIdx()}"/>')

xml_lines.extend(['  </Residue>', ' </Residues>', '</ForceField>'])
lig_xml = "\n".join(xml_lines)
Path("/tmp/test_lig.xml").write_text(lig_xml)
ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml", str(ff_xml), "/tmp/test_lig.xml")

from openmm.app import Element

# Build clean protein Modeller
prot_pdb = app.PDBFile("/home/kolarv/DrugEx/_david/validation/receptors/3OOB_pin1/receptor_clean.pdb")
modeller = app.Modeller(prot_pdb.topology, prot_pdb.positions)

# Build pure ligand topology and positions
lig_top = app.Topology()
lig_chain = lig_top.addChain("Z")
lig_res = lig_top.addResidue("LIG", lig_chain)
omm_atoms = []
for a in ob_mol.atoms:
    el = Element.getBySymbol(a.type[0])
    omm_atoms.append(lig_top.addAtom(f"A{a.idx}", el, lig_res))

for b in pybel.ob.OBMolBondIter(ob_mol.OBMol):
    lig_top.addBond(omm_atoms[b.GetBeginAtomIdx() - 1], omm_atoms[b.GetEndAtomIdx() - 1])

lig_coords = [openmm.Vec3(*a.coords) * unit.angstrom for a in ob_mol.atoms]

print("Adding ligand topology to protein Modeller...")
modeller.add(lig_top, lig_coords)
print("Total complex atoms before solvent:", len(modeller.positions))

print("Adding hydrogens at pH 7.4...")
modeller.addHydrogens(ff, pH=7.4)
print("Adding solvent box...")
modeller.addSolvent(ff, model="tip3p", padding=1.0*unit.nanometer, ionicStrength=0.15*unit.molar)
print(f"Solvated system atoms: {len(modeller.positions)}")

system = ff.createSystem(
    modeller.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0*unit.nanometer,
    constraints=app.HBonds
)
integrator = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
platform = openmm.Platform.getPlatformByName("OpenCL")
simulation = app.Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

print("Minimizing energy on NVIDIA RTX PRO 4000 Blackwell...")
simulation.minimizeEnergy(maxIterations=100)
print("Energy minimization complete!")

print("Running 100 test MD steps...")
simulation.step(100)
print("MD test execution on Blackwell GPU successful!")
