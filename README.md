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
* **Per-molecule styles:** CPK, Custom CPK, Ball & Stick, Stick, Wireframe or Hidden.
* **Custom CPK:** sphere size as a percentage of the van der Waals radius (10-300 %,
  100 % = CPK), set per molecule or per atom in the *Size %* column. Where shrunken
  spheres leave a gap, the bond is drawn as a thin stick.
* **Per-atom styles:** expand a molecule to style single atoms; the molecule row
  shows `(mixed)` when its atoms differ.
* **Opens itself:** selecting Molecule Styler in the 3D Style menu (or opening a
  project that uses it) opens the panel if it is closed, and never opens a second one.
* **Unselect** clears the selection and the highlight.
* **Highlight:** selecting a row in the panel marks its atoms in the 3D view with a
  yellow translucent shell, like the 3D-edit selection. Clicking an empty part of the panel (below the rows or on the
  background) clears the selection, and so does the **Unselect** button.
* **Stable atom ids:** atoms are listed by the host's unique, 0-based atom id
  (`C  (id 0)`), and styles are saved against that id, so they stay on the same
  atoms when the molecule is re-embedded or reloaded.
* **Apply to all / Reset:** set every atom at once, or return to the default.
* **Uses your settings:** radii, resolutions, bond colour, lighting and the
  colour overrides set by other plugins are honoured.
* **Saved with the project:** styles are stored in the project file and restored on
  load, and the 3D view switches to Molecule Styler automatically when a project
  with styles (or with this style active) is opened. File > New clears them.

## Usage
1. Load molecules in the 3D viewer.
2. Open **View > Molecule Styler Panel**.
3. Pick a style in the combo box of a molecule or atom. The viewer switches to the
   **Molecule Styler** automatically; select another 3D style to leave it
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
