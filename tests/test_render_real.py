"""Render with real pyvista/rdkit against *strict* fakes (no MagicMock fallback).

A MagicMock host hides wrong attribute paths (v0.1.0 read edit_3d_manager from
the view manager, where the real host does not have it), so every attribute the
renderer touches here must be declared explicitly. Skipped when the heavy
dependencies are absent (CI installs pytest only plus numpy/PyQt6).
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pv = pytest.importorskip("pyvista")
Chem = pytest.importorskip("rdkit.Chem")
np = pytest.importorskip("numpy")
from rdkit.Chem import AllChem  # noqa: E402

_PATH = Path(__file__).resolve().parents[1] / "molecule_styler.py"
_spec = importlib.util.spec_from_file_location("molecule_styler_real", _PATH)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)


class View3D:
    """Only the attributes the real View3DManager has and the renderer uses."""

    def __init__(self, plotter):
        self.plotter = plotter
        self._drawing_3d = False
        self._3d_color_map = {}
        self.axes_actor = None
        self._plugin_color_overrides = {}
        self._plugin_bond_color_overrides = {}
        self.atom_info_display_mode = None
        self.atom_actor = None
        self.current_mol = None
        self.calls = []

    def apply_3d_settings(self, redraw=True):
        self.calls.append(("apply", redraw))

    def update_atom_id_menu_text(self):
        pass

    def update_atom_id_menu_state(self):
        pass


def make_mw(plotter):
    cleared = []
    v3d = View3D(plotter)
    mw = SimpleNamespace(
        view_3d_manager=v3d,
        init_manager=SimpleNamespace(settings={}),
        edit_3d_manager=SimpleNamespace(
            clear_measurement_selection=lambda: cleared.append(1)
        ),
        get_settings=lambda: {},
    )
    return mw, v3d, cleared


def two_molecules():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO.O"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    conf = mol.GetConformer()
    frags = Chem.GetMolFrags(mol)
    for i in frags[1]:
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + 6, p.y, p.z))
    return mol, frags


def test_render_styled_draws_each_molecule_with_its_own_style():
    plotter = pv.Plotter(off_screen=True)
    mw, v3d, cleared = make_mw(plotter)
    mol, frags = two_molecules()
    st = ms._state
    st.clear()
    st.set_atoms(frags[0], "stick")
    st.set_atoms(frags[1], "cpk")
    try:
        ms.render_styled(mw, mol)
        assert cleared == [1]
        assert v3d._drawing_3d is False
        assert ("apply", False) in v3d.calls
        assert v3d.atom_actor is not None
        assert len(v3d.atom_positions_3d) == mol.GetNumAtoms()
        assert len(plotter.actors) >= 2  # spheres and bond tubes
    finally:
        st.clear()
        plotter.close()


def test_render_styled_empty_molecule_clears_scene():
    plotter = pv.Plotter(off_screen=True)
    mw, v3d, _ = make_mw(plotter)
    try:
        ms.render_styled(mw, None)
        assert v3d.atom_actor is None and v3d.current_mol is None
    finally:
        plotter.close()
