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


def test_state_prune_and_roundtrip():
    st = ms.StyleState()
    st.set_atoms([0, 5], "stick")
    st.set_atoms([1], "hidden")
    data = st.to_dict()
    assert data == {"0": "stick", "1": "hidden", "5": "stick"}
    st.prune(3)
    assert st.styles == {0: "stick", 1: "hidden"}
    st2 = ms.StyleState()
    st2.load_dict({"0": "stick", "x": "cpk", "2": "nope", "-1": "cpk", "3": "ball_and_stick"})
    assert st2.styles == {0: "stick"}
    st2.load_dict(None)
    assert st2.styles == {}


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
    st.set_atoms([5], "cpk")
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
    ms.StylerPanel._rebuild_if_changed(p, Mol(["C"] * 6, []))  # molecule comes back
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


def test_save_load_reset_roundtrip():
    ms._state.clear()
    assert ms._save() == {}
    ms._state.set_atoms([0, 2], "stick")
    saved = ms._save()
    assert saved == {ms.SAVE_KEY: {"0": "stick", "2": "stick"}}
    ms._reset_document()
    assert ms._state.styles == {}
    ms._load(saved)
    assert ms._state.styles == {0: "stick", 2: "stick"}
    ms._load(None)
    assert ms._state.styles == {}


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
