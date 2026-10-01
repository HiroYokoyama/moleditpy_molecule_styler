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


def test_render_styled_uses_unique_atom_ids_and_draws_highlight():
    plotter = pv.Plotter(off_screen=True)
    mw, v3d, _ = make_mw(plotter)
    mol, frags = two_molecules()
    for atom in mol.GetAtoms():  # the host stamps a unique, 0-based id on each atom
        atom.SetIntProp("_original_atom_id", 100 + atom.GetIdx())
    st = ms._state
    st.clear()
    panel = SimpleNamespace(selected=[0, 1], notify_molecule=lambda m: None)
    saved_panel = ms._panel
    ms._panel = panel
    try:
        st.bind(*ms.atom_keys(mol))
        st.set_atoms(frags[1], "cpk")
        ms.render_styled(mw, mol)
        assert st.mode == "id"
        assert set(st.by_id) == {100 + i for i in frags[1]}
        assert any(name == ms.HIGHLIGHT_NAME for name in plotter.actors)
        panel.selected = []
        ms.render_styled(mw, mol)
        assert not any(name == ms.HIGHLIGHT_NAME for name in plotter.actors)
    finally:
        ms._panel = saved_panel
        st.clear()
        st.bind([], "index")
        plotter.close()


def test_render_custom_cpk_with_percent_and_highlight():
    plotter = pv.Plotter(off_screen=True)
    mw, v3d, _ = make_mw(plotter)
    mol, frags = two_molecules()
    st = ms._state
    st.clear()
    st.bind([i for i in range(mol.GetNumAtoms())], "index")
    st.set_atoms(frags[0], "custom_cpk")
    st.set_percent(frags[0], 40)
    saved_panel = ms._panel
    ms._panel = SimpleNamespace(selected=[frags[0][0]], notify_molecule=lambda m: None)
    try:
        ms.render_styled(mw, mol)
        assert v3d.atom_actor is not None
        assert ms.HIGHLIGHT_NAME in plotter.actors
    finally:
        ms._panel = saved_panel
        st.clear()
        plotter.close()
