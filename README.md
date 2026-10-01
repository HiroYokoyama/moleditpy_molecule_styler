# Molecule Styler Plugin

[![CI](https://github.com/HiroYokoyama/moleditpy_molecule_styler/actions/workflows/test.yml/badge.svg)](https://github.com/HiroYokoyama/moleditpy_molecule_styler/actions/workflows/test.yml)
[![GitHub tag](https://img.shields.io/github/v/tag/HiroYokoyama/moleditpy_molecule_styler?label=version)](https://github.com/HiroYokoyama/moleditpy_molecule_styler/tags)
[![GitHub Downloads](https://img.shields.io/github/downloads/HiroYokoyama/moleditpy_molecule_styler/total)](https://github.com/HiroYokoyama/moleditpy_molecule_styler/releases)
[![License: GPL-3.0](https://img.shields.io/github/license/HiroYokoyama/moleditpy_molecule_styler)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
[![MoleditPy](https://img.shields.io/badge/MoleditPy->=4.0.0-3577F7)](https://github.com/HiroYokoyama/python_molecular_editor)

## Overview
Molecule Styler adds a dock panel on the right of the MoleditPy window that lists
every disconnected molecule in the 3D scene. Each molecule — and each atom inside
it — gets its own display style, so you can show a ligand as sticks, a solvent as
wireframe and a metal centre as CPK in the same view.

## Key Features
* **Per-molecule styles:** CPK, Ball & Stick, Stick, Wireframe or Hidden.
* **Per-atom styles:** expand a molecule to style single atoms; the molecule row
  shows `(mixed)` when its atoms differ.
* **Apply to all / Reset:** set every atom at once, or return to the default.
* **Uses your settings:** radii, resolutions, bond colour, lighting and the
  colour overrides set by other plugins are honoured.
* **Saved with the project** and cleared on File > New.

## Usage
1. Load molecules in the 3D viewer.
2. Open **View > Molecule Styler Panel**.
3. Pick a style in the combo box of a molecule or atom. The viewer switches to the
   **Per-Molecule Style** automatically; select another 3D style to leave it
   (your choices are kept).

Mixed styles: a bond is hidden if either atom is hidden, and between two CPK atoms
it is hidden inside the spheres; otherwise it takes the thinner of the two styles'
bond radii.

## Requirements
`PyQt6`, `numpy`, `pyvista` (all provided by the MoleditPy environment).

## Development and Testing
```bash
python -m pytest tests/ -v
```
The suite is headless (Qt stubbed) and also runs the static API check against the
main app when it is checked out next to this repository.
