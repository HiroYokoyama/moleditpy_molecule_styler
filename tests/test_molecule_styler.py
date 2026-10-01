"""Headless tests for molecule_styler.py (PyQt6 stubbed, no rdkit/pyvista)."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _install_stubs():
    if "PyQt6" in sys.modules and hasattr(sys.modules["PyQt6"], "__file__"):
        return
    pyqt6 = types.ModuleType("PyQt6")
    qt_core = types.ModuleType("PyQt6.QtCore")
    qt_core.Qt = MagicMock()
    qt_core.QTimer = MagicMock()
    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    for name in [
        "QComboBox", "QDockWidget", "QHBoxLayout", "QLabel", "QPushButton",
        "QTreeWidget", "QTreeWidgetItem", "QVBoxLayout", "QWidget",
    ]:
        setattr(qt_widgets, name, MagicMock())
    for name, mod in [
        ("PyQt6", pyqt6), ("PyQt6.QtCore", qt_core), ("PyQt6.QtWidgets", qt_widgets),
    ]:
        sys.modules.setdefault(name, mod)


_install_stubs()
_PATH = Path(__file__).resolve().parents[1] / "molecule_styler.py"
_spec = importlib.util.spec_from_file_location("molecule_styler", _PATH)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)


# --- helpers -----------------------------------------------------------------

CFG = {
    "vdw": lambda s: 1.7,
    "display_radius": lambda s: 0.4,
    "cpk_atom_scale": 1.0, "cpk_resolution": 32,
    "ball_stick_atom_scale": 1.0, "ball_stick_resolution": 16,
    "ball_stick_bond_radius": 0.1, "ball_stick_bond_color": (0.5, 0.5, 0.5),
    "stick_bond_radius": 0.15, "stick_resolution": 16,
    "wireframe_bond_radius": 0.02, "wireframe_resolution": 6,
}
POS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (5.0, 0.0, 0.0)]
RED, BLUE = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)


def scene(styles, bonds=((0, 1, 1.0, 0),), colors=(RED, BLUE, RED), **kw):
    return ms.build_scene(
        ["C", "O", "H"], POS, list(bonds), lambda i: styles[i], list(colors), CFG, **kw
    )


# --- pure logic --------------------------------------------------------------

def test_find_fragments_splits_disconnected_molecules():
    assert ms.find_fragments(5, [(0, 1), (3, 4)]) == [[0, 1], [2], [3, 4]]


def test_find_fragments_no_atoms_and_no_bonds():
    assert ms.find_fragments(0, []) == []
    assert ms.find_fragments(2, []) == [[0], [1]]


def test_find_fragments_merges_chain_regardless_of_bond_order():
    assert ms.find_fragments(4, [(3, 2), (1, 2), (0, 1)]) == [[0, 1, 2, 3]]


def test_formula_hill_order():
    assert ms.formula_of(["O", "H", "H"]) == "H2O"
    assert ms.formula_of(["H", "C", "O", "C", "H", "N"]) == "C2H2NO"
    assert ms.formula_of([]) == ""


def test_state_default_set_and_common():
    st = ms.StyleState()
    assert st.style_of(0) == ms.DEFAULT_STYLE
    st.set_atoms([0, 1], "cpk")
    assert st.common_style([0, 1]) == "cpk"
    assert st.common_style([0, 2]) is None
    st.set_atoms([0], ms.DEFAULT_STYLE)  # back to default removes the entry
    assert 0 not in st.styles
    st.set_atoms([2], "bogus")
    assert 2 not in st.styles


def test_state_index_mode_roundtrip():
    st = ms.StyleState()
    st.set_atoms([0, 5], "stick")
    st.set_atoms([1], "hidden")
    assert st.to_dict() == {"indices": {"0": "stick", "1": "hidden", "5": "stick"}}
    st2 = ms.StyleState()
    st2.load_dict(st.to_dict())
    assert st2.styles == {0: "stick", 1: "hidden", 5: "stick"}


def test_state_load_dict_filters_garbage():
    st = ms.StyleState()
    st.load_dict({"indices": {"0": "stick", "x": "cpk", "2": "nope", "-1": "cpk", "3": "ball_and_stick"},
                  "atom_ids": {"7": "wireframe", "8": "bogus"}})
    assert st.by_index == {0: "stick"} and st.by_id == {7: "wireframe"}
    st.load_dict(None)
    assert not st.has_styles()


def test_state_load_dict_ignores_unknown_formats():
    st = ms.StyleState()
    st.load_dict({"0": "stick", "1": "cpk"})  # the flat map of v0.2.0 is no longer read
    assert not st.has_styles()


def test_state_id_mode_follows_atoms_when_indices_change():
    st = ms.StyleState()
    st.bind([10, 11, 12], "id")  # atom ids, 0-based unique
    st.set_atoms([1], "cpk")
    assert st.by_id == {11: "cpk"}
    st.bind([12, 11, 10], "id")  # same atoms, re-embedded in another order
    assert st.style_of(1) == "cpk"
    assert st.style_of(0) == ms.DEFAULT_STYLE
    assert st.common_style([1]) == "cpk"


def test_state_id_roundtrip():
    st = ms.StyleState()
    st.bind([5, 6, 7], "id")
    st.set_atoms([0, 2], "stick")
    assert st.to_dict() == {"atom_ids": {"5": "stick", "7": "stick"}}
    st2 = ms.StyleState()
    st2.load_dict(st.to_dict())
    st2.bind([7, 6, 5], "id")
    assert st2.style_of(0) == "stick" and st2.style_of(2) == "stick"
    assert st2.style_of(1) == ms.DEFAULT_STYLE


def test_state_index_and_id_dicts_do_not_mix():
    st = ms.StyleState()
    st.bind([3, 4], "id")
    st.set_atoms([0], "cpk")
    st.bind([0, 1], "index")  # a molecule without ids
    assert st.style_of(0) == ms.DEFAULT_STYLE and st.by_id == {3: "cpk"}


class IdAtom:
    def __init__(self, sym, uid=None):
        self.sym, self.uid = sym, uid

    def GetSymbol(self):
        return self.sym

    def HasProp(self, name):
        return name == ms.ATOM_ID_PROP and self.uid is not None

    def GetIntProp(self, name):
        return self.uid


class IdMol:
    def __init__(self, atoms, bonds=()):
        self.atoms, self.bonds = atoms, list(bonds)

    def GetNumAtoms(self):
        return len(self.atoms)

    def GetAtoms(self):
        return list(self.atoms)

    def GetAtomWithIdx(self, i):
        return self.atoms[i]

    def GetBonds(self):
        return [SimpleNamespace(GetBeginAtomIdx=lambda a=a: a, GetEndAtomIdx=lambda b=b: b)
                for a, b in self.bonds]


def test_atom_keys_uses_unique_ids_when_all_present():
    mol = IdMol([IdAtom("C", 0), IdAtom("O", 4), IdAtom("H", 2)])
    assert ms.atom_keys(mol) == ([0, 4, 2], "id")


def test_atom_keys_falls_back_to_index():
    assert ms.atom_keys(IdMol([IdAtom("C", 0), IdAtom("O")])) == ([0, 1], "index")
    assert ms.atom_keys(IdMol([IdAtom("C", 3), IdAtom("O", 3)])) == ([0, 1], "index")


# --- scene building ----------------------------------------------------------

def test_spheres_per_style():
    spheres, _ = scene(["cpk", "ball_and_stick", "stick"])
    by_idx = {s[0]: s for s in spheres}
    assert by_idx[0][2] == pytest.approx(1.7) and by_idx[0][3] == 32
    assert by_idx[1][2] == pytest.approx(0.4) and by_idx[1][3] == 16
    assert by_idx[2][2] == pytest.approx(0.15)


def test_wireframe_and_hidden_have_no_spheres():
    spheres, _ = scene(["wireframe", "hidden", "hidden"])
    assert spheres == []


def test_hidden_atom_hides_its_bonds():
    _, segs = scene(["hidden", "stick", "stick"])
    assert segs == []


def test_cpk_cpk_bond_not_drawn_but_cpk_stick_is():
    _, segs = scene(["cpk", "cpk", "cpk"])
    assert segs == []
    _, segs = scene(["cpk", "stick", "cpk"])
    assert segs and segs[0][2] == pytest.approx(0.15)


def test_mixed_bond_uses_thinner_radius_and_split_colours():
    _, segs = scene(["stick", "wireframe", "stick"])
    assert len(segs) == 2  # two halves, one per atom colour
    assert all(s[2] == pytest.approx(0.02) for s in segs)
    assert segs[0][3] == RED and segs[1][3] == BLUE


def test_ball_and_stick_pair_uses_uniform_grey():
    _, segs = scene(["ball_and_stick", "ball_and_stick", "stick"])
    assert len(segs) == 1 and segs[0][3] == segs[0][4] == (0.5, 0.5, 0.5)


def test_bond_colour_override_wins():
    _, segs = scene(["stick", "stick", "stick"], bond_colors={0: (0.1, 0.2, 0.3)})
    assert len(segs) == 1 and segs[0][3] == (0.1, 0.2, 0.3)


def test_multiple_bond_line_counts():
    for order, expected in [(1.0, 1), (1.5, 1), (2.0, 2), (3.0, 3)]:
        _, segs = scene(["ball_and_stick"] * 3, bonds=[(0, 1, order, 0)])
        assert len(segs) == expected, order


def test_double_bond_lines_are_offset_and_parallel():
    _, segs = scene(["stick"] * 3, bonds=[(0, 1, 2.0, 0)], colors=(RED, RED, RED))
    (a0, a1, *_), (b0, b1, *_) = segs
    assert a0[1] == pytest.approx(-b0[1]) or a0[2] == pytest.approx(-b0[2])
    assert a0 != b0
    assert [x - y for x, y in zip(a1, a0)] == pytest.approx([x - y for x, y in zip(b1, b0)])


# --- host helpers ------------------------------------------------------------

def test_qcolor_rgb():
    assert ms._qcolor_rgb("#FF0000") == (1.0, 0.0, 0.0)
    assert ms._qcolor_rgb("garbage") is None
    assert ms._qcolor_rgb(None) is None


def test_make_cfg_defaults_and_vdw_fallback():
    cfg = ms._make_cfg({}, {"C": 0.7}, None)
    assert cfg["display_radius"]("C") == 0.7
    assert cfg["display_radius"]("Xx") == ms._FALLBACK_DISPLAY_RADIUS
    assert cfg["vdw"]("C") == ms._FALLBACK_VDW  # pt is None -> fallback
    assert cfg["ball_stick_bond_color"] == pytest.approx((127 / 255.0,) * 3)
    pt = SimpleNamespace(GetRvdw=lambda z: 1.7, GetAtomicNumber=lambda s: 6)
    assert ms._make_cfg({}, {}, pt)["vdw"]("C") == 1.7


def test_host_constants_reads_module_of_manager_class():
    mod = types.ModuleType("fake_view3d")
    mod.CPK_COLORS_PV = {"C": [0.1, 0.1, 0.1]}
    mod.VDW_DISPLAY_RADII = {"C": 0.9}
    sys.modules["fake_view3d"] = mod
    try:
        cls = type("V", (), {"__module__": "fake_view3d"})
        colors, radii, pt = ms._host_constants(cls())
        assert colors == {"C": [0.1, 0.1, 0.1]} and radii == {"C": 0.9} and pt is None
    finally:
        del sys.modules["fake_view3d"]


# --- panel logic (methods bound to fakes) ------------------------------------

class FakeCombo:
    def __init__(self, index=0):
        self.index = index
        self.blocked = []

    def setCurrentIndex(self, i):
        self.index = i

    def blockSignals(self, flag):
        self.blocked.append(flag)

    def currentIndex(self):
        return self.index


def make_panel(n=3):
    st = ms.StyleState()
    ctx = MagicMock()
    panel = SimpleNamespace(
        state=st, context=ctx, mw=ctx.get_main_window(), symbols=["C"] * n,
        fragments=[[0, 1], [2]], _combos=[], all_combo=FakeCombo(),
        redraw=MagicMock(),
    )
    panel._set_combo = lambda c, idx: ms.StylerPanel._set_combo(panel, c, idx)
    panel._sync_combos = lambda: ms.StylerPanel._sync_combos(panel)
    return panel


def test_set_combo_shows_style_or_mixed():
    p = make_panel()
    c = FakeCombo()
    p.state.set_atoms([0, 1], "stick")
    ms.StylerPanel._set_combo(p, c, [0, 1])
    assert c.index == ms.STYLES.index("stick")
    ms.StylerPanel._set_combo(p, c, [0, 2])
    assert c.index == len(ms.STYLES)  # mixed
    assert c.blocked == [True, False, True, False]


def test_on_choice_applies_style_and_redraws():
    p = make_panel()
    c = FakeCombo()
    p._combos = [(c, [0, 1])]
    ms.StylerPanel._on_choice(p, c, [0, 1], ms.STYLES.index("cpk"))
    assert p.state.common_style([0, 1]) == "cpk"
    assert c.index == ms.STYLES.index("cpk")
    p.redraw.assert_called_once()


def test_on_choice_mixed_entry_is_inert():
    p = make_panel()
    c = FakeCombo()
    ms.StylerPanel._on_choice(p, c, [0, 1], len(ms.STYLES))
    assert p.state.styles == {}
    p.redraw.assert_not_called()


def test_apply_all_and_reset():
    p = make_panel()
    p.all_combo.index = ms.STYLES.index("wireframe")
    ms.StylerPanel._apply_all(p)
    assert p.state.common_style([0, 1, 2]) == "wireframe"
    ms.StylerPanel._reset(p)
    assert p.state.styles == {}
    assert p.redraw.call_count == 2


def test_redraw_switches_style_or_redraws_in_place():
    ctx = MagicMock()
    v3d = MagicMock()
    ctx.get_main_window.return_value = SimpleNamespace(view_3d_manager=v3d)
    ctx.current_mol = object()
    p = SimpleNamespace(
        mw=ctx.get_main_window(), context=ctx, _current_mol=lambda: ctx.current_mol
    )
    v3d.current_3d_style = "cpk"
    ms.StylerPanel.redraw(p)
    v3d.set_3d_style.assert_called_once_with(ms.STYLE_NAME)
    v3d.current_3d_style = ms.STYLE_NAME
    ms.StylerPanel.redraw(p)
    v3d.draw_molecule_3d.assert_called_once_with(ctx.current_mol)
    ctx.mark_project_modified.assert_called()


def test_redraw_without_molecule_is_noop():
    ctx = MagicMock()
    v3d = MagicMock()
    p = SimpleNamespace(
        mw=SimpleNamespace(view_3d_manager=v3d), context=ctx, _current_mol=lambda: None
    )
    ms.StylerPanel.redraw(p)
    v3d.set_3d_style.assert_not_called()
    v3d.draw_molecule_3d.assert_not_called()


def test_rebuild_if_changed_only_when_signature_differs():
    class Atom:
        def __init__(self, s): self.s = s
        def GetSymbol(self): return self.s
        def HasProp(self, name): return False

    class Bond:
        def __init__(self, a, b): self.a, self.b = a, b
        def GetBeginAtomIdx(self): return self.a
        def GetEndAtomIdx(self): return self.b

    class Mol:
        def __init__(self, syms, bonds): self.syms, self.bonds = syms, bonds
        def GetNumAtoms(self): return len(self.syms)
        def GetAtoms(self): return [Atom(s) for s in self.syms]
        def GetBonds(self): return [Bond(a, b) for a, b in self.bonds]

    st = ms.StyleState()
    st.set_atoms([5], "cpk")  # index 5 only exists in the 6-atom molecule below
    p = SimpleNamespace(state=st, _signature=None, fragments=[], symbols=[], _populate=MagicMock())
    mol = Mol(["C", "O", "H"], [(0, 1)])
    ms.StylerPanel._rebuild_if_changed(p, mol)
    assert p.fragments == [[0, 1], [2]] and p.symbols == ["C", "O", "H"]
    assert p._populate.call_count == 1
    assert st.styles == {5: "cpk"}  # kept: a transient smaller mol must not erase styles
    ms.StylerPanel._rebuild_if_changed(p, Mol(["C", "O", "H"], [(0, 1)]))
    assert p._populate.call_count == 1  # identical structure: no rebuild
    ms.StylerPanel._rebuild_if_changed(p, None)
    assert p._populate.call_count == 2 and p.fragments == []
    assert st.styles == {5: "cpk"}
    ms.StylerPanel._rebuild_if_changed(p, Mol(["C", "O", "H", "H", "H", "H"], []))  # molecule comes back
    assert st.style_of(5) == "cpk"


# --- plugin entry points -----------------------------------------------------

def test_initialize_registers_everything():
    ctx = MagicMock()
    ms.initialize(ctx)
    ctx.register_3d_style.assert_called_once_with(ms.STYLE_NAME, ms.render_styled)
    assert ctx.add_menu_action.call_args[0][0] == "View/Molecule Styler Panel"
    ctx.register_save_handler.assert_called_once()
    ctx.register_load_handler.assert_called_once()
    ctx.register_document_reset_handler.assert_called_once()


def _ctx(style):
    mw, actions = _mw_with_menu(["Ball & Stick", ms.STYLE_NAME])
    mw.view_3d_manager = SimpleNamespace(current_3d_style=style)
    ctx = MagicMock()
    ctx.get_main_window.return_value = mw
    return ctx, mw, actions


def test_save_load_reset_roundtrip(monkeypatch):
    ctx, mw, actions = _ctx("ball_and_stick")
    monkeypatch.setattr(ms, "_context", ctx)
    ms._state.clear()
    assert ms._save() == {}  # nothing styled and style not active
    ms._state.set_atoms([0, 2], "stick")
    saved = ms._save()
    assert saved == {
        ms.SAVE_KEY: {"indices": {"0": "stick", "2": "stick"}},
        ms.ACTIVE_KEY: False,
    }
    ms._reset_document()
    assert ms._state.styles == {}
    ms._load(saved)
    assert ms._state.styles == {0: "stick", 2: "stick"}
    ms._load({ms.SAVE_KEY: {"0": "stick"}})  # v0.2.0 flat map: not supported any more
    assert not ms._state.has_styles()
    ms._load(None)
    assert ms._state.styles == {}


def test_save_records_active_style_even_without_per_atom_styles(monkeypatch):
    ctx, _, _ = _ctx(ms.STYLE_NAME)
    monkeypatch.setattr(ms, "_context", ctx)
    ms._state.clear()
    assert ms._save() == {ms.SAVE_KEY: {}, ms.ACTIVE_KEY: True}


def test_load_with_styles_switches_style_and_ticks_menu(monkeypatch):
    ctx, mw, actions = _ctx("cpk")
    monkeypatch.setattr(ms, "_context", ctx)
    ms._load({ms.SAVE_KEY: {"indices": {"1": "stick"}}, ms.ACTIVE_KEY: False})
    assert mw.view_3d_manager.current_3d_style == ms.STYLE_NAME
    assert actions[1].checked
    ms._state.clear()


def test_load_with_active_flag_only_switches_style(monkeypatch):
    ctx, mw, _ = _ctx("cpk")
    monkeypatch.setattr(ms, "_context", ctx)
    ms._load({ms.SAVE_KEY: {}, ms.ACTIVE_KEY: True})
    assert mw.view_3d_manager.current_3d_style == ms.STYLE_NAME


def test_load_of_unrelated_data_leaves_style_alone(monkeypatch):
    ctx, mw, _ = _ctx("cpk")
    monkeypatch.setattr(ms, "_context", ctx)
    ms._load({})
    ms._load(None)
    assert mw.view_3d_manager.current_3d_style == "cpk"


def test_load_does_not_redraw_previous_molecule(monkeypatch):
    ctx, mw, _ = _ctx("cpk")
    mw.view_3d_manager.set_3d_style = MagicMock()
    mw.view_3d_manager.draw_molecule_3d = MagicMock()
    monkeypatch.setattr(ms, "_context", ctx)
    ms._load({ms.SAVE_KEY: {"indices": {"0": "stick"}}})
    mw.view_3d_manager.set_3d_style.assert_not_called()
    mw.view_3d_manager.draw_molecule_3d.assert_not_called()
    ms._state.clear()


def test_select_style_without_context_or_view_is_safe():
    ms._select_style(None)
    ctx = MagicMock()
    ctx.get_main_window.return_value = SimpleNamespace()
    ms._select_style(ctx)
    assert ms._is_active(None) is False


def test_toggle_panel_creates_once_then_toggles(monkeypatch):
    created = []

    class FakePanel:
        def __init__(self, ctx, state):
            created.append(self)
            self.dock = object()
            self.toggled = 0
            self.activated = 0

        def activate_style(self):
            self.activated += 1

        def toggle(self):
            self.toggled += 1

    monkeypatch.setattr(ms, "StylerPanel", FakePanel)
    monkeypatch.setattr(ms, "_panel", None)
    ctx = MagicMock()
    ms._toggle_panel(ctx)
    ms._toggle_panel(ctx)
    assert len(created) == 1 and created[0].toggled == 1
    assert created[0].activated == 1  # switched to the style when first opened
    ctx.register_window.assert_called_once()
    monkeypatch.setattr(ms, "_panel", None)


def test_render_styled_skips_when_reentrant():
    v3d = SimpleNamespace(_drawing_3d=True)
    mw = SimpleNamespace(view_3d_manager=v3d)
    ms.render_styled(mw, object())  # must return without touching anything else
    assert v3d._drawing_3d is True


def test_activate_style_switches_only_when_needed():
    v3d = MagicMock()
    p = SimpleNamespace(mw=SimpleNamespace(view_3d_manager=v3d))
    v3d.current_3d_style = "cpk"
    ms.StylerPanel.activate_style(p)
    v3d.set_3d_style.assert_called_once_with(ms.STYLE_NAME)
    v3d.current_3d_style = ms.STYLE_NAME
    ms.StylerPanel.activate_style(p)
    v3d.set_3d_style.assert_called_once()


def test_toggle_shows_then_hides_and_switches_style_on_show():
    dock = MagicMock()
    dock.isVisible.return_value = False
    p = SimpleNamespace(dock=dock, activate_style=MagicMock())
    ms.StylerPanel.toggle(p)
    dock.setVisible.assert_called_with(True)
    p.activate_style.assert_called_once()
    dock.isVisible.return_value = True
    ms.StylerPanel.toggle(p)
    dock.setVisible.assert_called_with(False)
    p.activate_style.assert_called_once()


class FakeAction:
    def __init__(self, text):
        self._text, self.checked = text, False

    def text(self):
        return self._text

    def setChecked(self, flag):
        self.checked = flag


def _mw_with_menu(texts):
    actions = [FakeAction(t) for t in texts]
    menu = SimpleNamespace(actions=lambda: actions)
    button = SimpleNamespace(menu=lambda: menu)
    return SimpleNamespace(init_manager=SimpleNamespace(style_button=button)), actions


def test_sync_style_menu_ticks_plugin_entry():
    mw, actions = _mw_with_menu(["Ball & Stick", "Stick", ms.STYLE_NAME])
    ms._sync_style_menu(mw)
    assert [a.checked for a in actions] == [False, False, True]


def test_sync_style_menu_tolerates_missing_or_empty_menu():
    ms._sync_style_menu(SimpleNamespace())  # no init_manager
    ms._sync_style_menu(SimpleNamespace(init_manager=SimpleNamespace()))  # no button
    mw, actions = _mw_with_menu(["Ball & Stick"])  # entry absent
    ms._sync_style_menu(mw)
    assert actions[0].checked is False
    mw.init_manager.style_button = SimpleNamespace(menu=lambda: None)
    ms._sync_style_menu(mw)


def test_activate_style_and_redraw_tick_the_menu():
    mw, actions = _mw_with_menu(["Ball & Stick", ms.STYLE_NAME])
    v3d = MagicMock()
    v3d.current_3d_style = "ball_and_stick"
    mw.view_3d_manager = v3d
    ms.StylerPanel.activate_style(SimpleNamespace(mw=mw))
    assert actions[1].checked
    actions[1].checked = False
    ctx = MagicMock()
    p = SimpleNamespace(mw=mw, context=ctx, _current_mol=lambda: object())
    ms.StylerPanel.redraw(p)
    assert actions[1].checked


def test_save_uses_unique_ids_not_indices(monkeypatch):
    ctx, _, _ = _ctx("ball_and_stick")
    monkeypatch.setattr(ms, "_context", ctx)
    ms._state.clear()
    ms._state.bind([7, 8, 9], "id")
    ms._state.set_atoms([0], "cpk")
    assert ms._save()[ms.SAVE_KEY] == {"atom_ids": {"7": "cpk"}}
    ms._state.clear()
    ms._state.bind([], "index")


# --- row selection -> yellow highlight ---------------------------------------

class FakeItem:
    def __init__(self, indices):
        self.indices = indices

    def data(self, column, role):
        return self.indices


def test_selection_collects_atoms_of_selected_rows_and_redraws():
    p = SimpleNamespace(
        tree=SimpleNamespace(selectedItems=lambda: [FakeItem([3, 1]), FakeItem([1, 2]), FakeItem(None)]),
        selected=[],
        draw_selection=MagicMock(),
    )
    ms.StylerPanel._on_selection(p)
    assert p.selected == [1, 2, 3]
    p.draw_selection.assert_called_once()


def test_draw_selection_calls_highlight_and_renders(monkeypatch):
    calls = []
    monkeypatch.setattr(ms, "draw_highlight", lambda v3d, mol, idx: calls.append(list(idx)))
    plotter = MagicMock()
    mw = SimpleNamespace(view_3d_manager=SimpleNamespace(plotter=plotter))
    p = SimpleNamespace(mw=mw, selected=[0, 2], _current_mol=lambda: object())
    ms.StylerPanel.draw_selection(p)
    assert calls == [[0, 2]]
    plotter.render.assert_called_once()


def test_draw_selection_is_noop_without_molecule_or_plotter(monkeypatch):
    monkeypatch.setattr(ms, "draw_highlight", lambda *a: pytest.fail("must not draw"))
    mw = SimpleNamespace(view_3d_manager=SimpleNamespace(plotter=MagicMock()))
    ms.StylerPanel.draw_selection(SimpleNamespace(mw=mw, selected=[0], _current_mol=lambda: None))
    mw2 = SimpleNamespace(view_3d_manager=SimpleNamespace(plotter=None))
    ms.StylerPanel.draw_selection(SimpleNamespace(mw=mw2, selected=[0], _current_mol=lambda: object()))
    ms.StylerPanel.draw_selection(SimpleNamespace(mw=SimpleNamespace(), selected=[0], _current_mol=lambda: object()))


def test_draw_selection_swallows_render_errors(monkeypatch):
    def boom(*a):
        raise RuntimeError("vtk gone")

    monkeypatch.setattr(ms, "draw_highlight", boom)
    mw = SimpleNamespace(view_3d_manager=SimpleNamespace(plotter=MagicMock()))
    ms.StylerPanel.draw_selection(SimpleNamespace(mw=mw, selected=[0], _current_mol=lambda: object()))


# --- a new molecule starts at the default style --------------------------------

def test_new_molecule_with_overlapping_ids_does_not_inherit_styles():
    st = ms.StyleState()
    st.bind([0, 1, 2], "id", ["C", "C", "O"])
    st.set_atoms([0, 1], "cpk")
    st.bind([0, 1, 2], "id", ["N", "N", "H"])  # a different molecule, ids restart at 0
    assert not st.has_styles()
    assert st.style_of(0) == ms.DEFAULT_STYLE == "ball_and_stick"


def test_edited_molecule_keeps_styles_when_atoms_are_added_or_transiently_missing():
    st = ms.StyleState()
    st.bind([0, 1, 2, 3], "id", ["C", "C", "O", "H"])
    st.set_atoms([0], "stick")
    st.bind([0, 1, 2, 3, 4], "id", ["C", "C", "O", "H", "H"])  # atom added
    assert st.style_of(0) == "stick"
    st.bind([0, 1, 2], "id", ["C", "C", "O"])  # transient subset during a redraw
    assert st.style_of(0) == "stick"
    st.bind([0, 1, 2, 3], "id", ["C", "C", "O", "H"])
    assert st.style_of(0) == "stick"


def test_new_molecule_detection_in_index_mode_and_across_modes():
    st = ms.StyleState()
    st.bind([0, 1], "index", ["C", "O"])
    st.set_atoms([0], "cpk")
    st.bind([0, 1], "index", ["Cl", "Br"])
    assert not st.has_styles()
    st.set_atoms([0], "cpk")
    st.bind([5, 6], "id", ["Cl", "Br"])  # mode change is not compared
    assert st.by_index == {0: "cpk"}


def test_project_load_adopts_its_molecule_instead_of_clearing_styles():
    st = ms.StyleState()
    st.bind([0, 1], "id", ["C", "C"])  # molecule that was open before the load
    st.load_dict({"atom_ids": {"3": "stick"}})
    st.bind([3, 4], "id", ["O", "H"])  # the project's own molecule
    assert st.by_id == {3: "stick"}


def test_bind_without_symbols_never_clears():
    st = ms.StyleState()
    st.bind([0, 1], "id", ["C", "C"])
    st.set_atoms([0], "cpk")
    st.bind([0, 1], "id")
    assert st.style_of(0) == "cpk"


def test_unselect_clears_selection_and_highlight():
    tree = MagicMock()
    p = SimpleNamespace(tree=tree, selected=[1, 2], draw_selection=MagicMock())
    ms.StylerPanel._unselect(p)
    tree.clearSelection.assert_called_once()
    assert p.selected == []
    p.draw_selection.assert_called_once()


def test_comparison_uses_unique_ids_not_indices():
    st = ms.StyleState()
    st.bind([10, 11, 12], "id", ["C", "C", "O"])
    st.set_atoms([0], "cpk")  # atom with unique id 10
    st.bind([12, 11, 10], "id", ["O", "C", "C"])  # same atoms, shuffled indices
    assert st.style_of(2) == "cpk"
