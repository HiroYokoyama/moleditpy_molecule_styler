"""
Molecule Styler Plugin for MoleditPy.
Adds a dock panel on the right of the main window that lists every
disconnected molecule in the 3D scene and lets you pick a display style
(CPK, Ball & Stick, Stick, Wireframe, Hidden) for each molecule, or for
each individual atom inside it.

Source code, README, and full license (GNU GPL):
    https://github.com/HiroYokoyama/moleditpy_molecule_styler
Copyright (c) HiroYokoyama. Licensed under the GNU General Public License;
see the LICENSE file in the repository above for the full terms.
"""

# pylint: disable=too-many-instance-attributes,no-name-in-module,too-many-locals

import logging
import math
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import numpy as np
except ImportError:  # CI installs pytest only; rendering needs numpy at runtime
    np = None

logger = logging.getLogger(__name__)

# --- Plugin Metadata ---
PLUGIN_NAME = "Molecule Styler"
PLUGIN_VERSION = "0.3.1"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Right-hand panel listing each disconnected molecule, with per-molecule and "
    "per-atom 3D display styles (CPK, Ball & Stick, Stick, Wireframe, Hidden)."
)
PLUGIN_CATEGORY = "Visualization"
PLUGIN_TAGS = ["Visualization"]
PLUGIN_DEPENDENCIES = ["pyvista", "PyQt6", "numpy"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=4.0.0, <5.0.0"
PLUGIN_SUPPORTED_PYTHON_VERSION = ">=3.9, <3.15"
PLUGIN_SUPPORTED_OS = ["Windows", "macOS", "Linux", "WSL"]

STYLE_NAME = "Molecule Styler"
SAVE_KEY = "styles"
ACTIVE_KEY = "active"
DEFAULT_STYLE = "ball_and_stick"
MIXED_LABEL = "(mixed)"
POLL_MS = 600
WIDTH_SCALE = 1.2  # panel is 20% wider than Qt's default size hint

STYLES = ["cpk", "ball_and_stick", "stick", "wireframe", "hidden"]
STYLE_LABELS = {
    "cpk": "CPK",
    "ball_and_stick": "Ball & Stick",
    "stick": "Stick",
    "wireframe": "Wireframe",
    "hidden": "Hidden",
}

# Fallbacks used only when the host's constants cannot be found.
_FALLBACK_COLOR = (0.5, 0.5, 0.5)
_FALLBACK_DISPLAY_RADIUS = 0.4
_FALLBACK_VDW = 1.5

_BOND_RADIUS_KEY = {
    "ball_and_stick": ("ball_stick_bond_radius", 0.1),
    "stick": ("stick_bond_radius", 0.15),
    "wireframe": ("wireframe_bond_radius", 0.02),
}


# ---------------------------------------------------------------------------
# Pure logic (no Qt / rdkit / pyvista) -- unit tested headlessly
# ---------------------------------------------------------------------------


def find_fragments(num_atoms, bonds):
    """Group atom indices into connected molecules.

    Args:
        num_atoms: number of atoms.
        bonds: iterable of (begin_idx, end_idx) pairs.
    Returns:
        List of sorted index lists, ordered by their lowest atom index.
    """
    parent = list(range(num_atoms))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in bonds:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    for i in range(num_atoms):
        groups.setdefault(root(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


def formula_of(symbols):
    """Hill-order molecular formula from a list of element symbols."""
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else [])
        order += sorted(k for k in counts if k not in ("C", "H"))
    else:
        order = sorted(counts)
    return "".join(f"{k}{counts[k] if counts[k] > 1 else ''}" for k in order)


ATOM_ID_PROP = "_original_atom_id"  # the host's unique, 0-based editor atom id


def atom_keys(mol):
    """Per-atom style keys: (keys, mode).

    The host stamps every atom with a unique editor id (``_original_atom_id``,
    counted from 0) that survives re-embedding and file round-trips, whereas the
    RDKit index does not. Use it when every atom has one; otherwise fall back to
    the index.
    """
    atoms = list(mol.GetAtoms())
    ids = []
    for atom in atoms:
        if not atom.HasProp(ATOM_ID_PROP):
            return list(range(len(atoms))), "index"
        ids.append(atom.GetIntProp(ATOM_ID_PROP))
    if len(set(ids)) != len(ids):
        return list(range(len(atoms))), "index"
    return ids, "id"


class StyleState:
    """Atom -> style map; atoms not listed use DEFAULT_STYLE.

    Styles are stored under the atom's unique id when the molecule has one
    (``by_id``) and under the RDKit index otherwise (``by_index``). Callers
    always speak in RDKit indices; ``bind`` tells the state how to translate.
    """

    def __init__(self):
        self.by_id = {}
        self.by_index = {}
        self.mode = "index"
        self.keys = []

    def bind(self, keys, mode):
        """Attach the current molecule's key list and key mode."""
        self.keys = list(keys)
        self.mode = mode

    def _active(self):
        return self.by_id if self.mode == "id" else self.by_index

    def key(self, idx):
        """Storage key for the atom at RDKit index `idx`."""
        if self.mode == "id" and idx < len(self.keys):
            return self.keys[idx]
        return idx

    def style_of(self, idx):
        """Style of one atom."""
        return self._active().get(self.key(idx), DEFAULT_STYLE)

    def set_atoms(self, indices, style):
        """Assign `style` to every atom in `indices`."""
        if style not in STYLES:
            return
        store = self._active()
        for i in indices:
            k = self.key(i)
            if style == DEFAULT_STYLE:
                store.pop(k, None)
            else:
                store[k] = style

    def common_style(self, indices):
        """The shared style of `indices`, or None when they differ."""
        found = {self.style_of(i) for i in indices}
        return found.pop() if len(found) == 1 else None

    @property
    def styles(self):
        """The live style map for the current key mode."""
        return self._active()

    def has_styles(self):
        """True if any atom has a non-default style stored."""
        return bool(self.by_id or self.by_index)

    def clear(self):
        """Back to the default style everywhere."""
        self.by_id.clear()
        self.by_index.clear()

    def to_dict(self):
        """JSON-serialisable form for the project file."""
        out = {}
        if self.by_id:
            out["atom_ids"] = {str(k): v for k, v in sorted(self.by_id.items())}
        if self.by_index:
            out["indices"] = {str(k): v for k, v in sorted(self.by_index.items())}
        return out

    @staticmethod
    def _parse(raw):
        result = {}
        if isinstance(raw, dict):
            for key, style in raw.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                if idx >= 0 and style in STYLES and style != DEFAULT_STYLE:
                    result[idx] = style
        return result

    def load_dict(self, data):
        """Restore from `to_dict` output; anything else is ignored."""
        self.clear()
        if not isinstance(data, dict):
            return
        self.by_id = self._parse(data.get("atom_ids"))
        self.by_index = self._parse(data.get("indices"))


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return (v[0] / n, v[1] / n, v[2] / n) if n > 0 else (0.0, 0.0, 1.0)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _perpendicular(direction):
    d = _unit(direction)
    arb = (0.0, 0.0, 1.0) if abs(d[2]) < 0.9 else (0.0, 1.0, 0.0)
    return _unit(_cross(d, arb))


def build_scene(symbols, positions, bonds, styles, colors, cfg, bond_colors=None):
    """Turn per-atom styles into spheres and bond segments.

    Args:
        symbols: element symbol per atom.
        positions: (x, y, z) per atom.
        bonds: list of (begin, end, order, bond_idx); order is a float.
        styles: callable atom_idx -> style name.
        colors: list of (r, g, b) floats in 0..1 per atom.
        cfg: dict with scale/radius/resolution settings and the callables
            ``vdw(symbol)`` and ``display_radius(symbol)``.
        bond_colors: optional bond_idx -> (r, g, b) override.
    Returns:
        (spheres, segments) -- spheres are (atom_idx, pos, radius, resolution),
        segments are (p0, p1, radius, rgb0, rgb1).
    """
    bond_colors = bond_colors or {}
    spheres = []
    for i, sym in enumerate(symbols):
        st = styles(i)
        if st == "cpk":
            r = cfg["vdw"](sym) * cfg["cpk_atom_scale"]
            res = cfg["cpk_resolution"]
        elif st == "ball_and_stick":
            r = cfg["display_radius"](sym) * cfg["ball_stick_atom_scale"]
            res = cfg["ball_stick_resolution"]
        elif st == "stick":
            r = cfg["stick_bond_radius"]
            res = cfg["stick_resolution"]
        else:  # wireframe / hidden: no sphere
            continue
        spheres.append((i, positions[i], r, res))

    segments = []
    for begin, end, order, bidx in bonds:
        sa, sb = styles(begin), styles(end)
        if "hidden" in (sa, sb) or (sa == "cpk" and sb == "cpk"):
            continue
        radii = [
            cfg[_BOND_RADIUS_KEY[s][0]]
            for s in (sa, sb)
            if s in _BOND_RADIUS_KEY
        ]
        radius = min(radii)
        p0, p1 = positions[begin], positions[end]
        if bidx in bond_colors:
            c0 = c1 = bond_colors[bidx]
        elif sa == "ball_and_stick" and sb == "ball_and_stick":
            c0 = c1 = cfg["ball_stick_bond_color"]
        else:
            c0, c1 = colors[begin], colors[end]

        n_lines = max(1, int(order))
        if n_lines == 1:
            offsets, factor = [(0.0, 0.0, 0.0)], 1.0
        else:
            side = _perpendicular((p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]))
            if n_lines == 2:
                step, factor = radius * 1.6, 0.6
                offsets = [tuple(c * step for c in side), tuple(-c * step for c in side)]
            else:
                step, factor = radius * 2.2, 0.45
                offsets = [
                    (0.0, 0.0, 0.0),
                    tuple(c * step for c in side),
                    tuple(-c * step for c in side),
                ]
        for off in offsets:
            a = (p0[0] + off[0], p0[1] + off[1], p0[2] + off[2])
            b = (p1[0] + off[0], p1[1] + off[1], p1[2] + off[2])
            if c0 == c1:
                segments.append((a, b, radius * factor, c0, c1))
            else:
                mid = tuple((x + y) / 2.0 for x, y in zip(a, b))
                segments.append((a, mid, radius * factor, c0, c0))
                segments.append((mid, b, radius * factor, c1, c1))
    return spheres, segments


# ---------------------------------------------------------------------------
# Host access helpers
# ---------------------------------------------------------------------------


def _host_constants(v3d):
    """CPK colours / display radii / periodic table from the host's own module."""
    mod = sys.modules.get(type(v3d).__module__)
    return (
        getattr(mod, "CPK_COLORS_PV", {}) or {},
        getattr(mod, "VDW_DISPLAY_RADII", {}) or {},
        getattr(mod, "pt", None),
    )


def _qcolor_rgb(hex_color):
    """'#RRGGBB' -> (r, g, b) floats, or None if unparsable."""
    try:
        h = str(hex_color).lstrip("#")
        return tuple(int(h[k : k + 2], 16) / 255.0 for k in (0, 2, 4))
    except (TypeError, ValueError):
        return None


def _make_cfg(settings, display_radii, pt):
    def get(key, default):
        return settings.get(key, default)

    def vdw(sym):
        try:
            r = pt.GetRvdw(pt.GetAtomicNumber(sym))
            return r if r > 0.1 else _FALLBACK_VDW
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return _FALLBACK_VDW

    bond_rgb = _qcolor_rgb(get("ball_stick_bond_color", "#7F7F7F")) or (0.5, 0.5, 0.5)
    return {
        "vdw": vdw,
        "display_radius": lambda s: display_radii.get(s, _FALLBACK_DISPLAY_RADIUS),
        "cpk_atom_scale": get("cpk_atom_scale", 1.0),
        "cpk_resolution": get("cpk_resolution", 32),
        "ball_stick_atom_scale": get("ball_stick_atom_scale", 1.0),
        "ball_stick_resolution": get("ball_stick_resolution", 16),
        "ball_stick_bond_radius": get("ball_stick_bond_radius", 0.1),
        "ball_stick_bond_color": bond_rgb,
        "stick_bond_radius": get("stick_bond_radius", 0.15),
        "stick_resolution": get("stick_resolution", 16),
        "wireframe_bond_radius": get("wireframe_bond_radius", 0.02),
        "wireframe_resolution": get("wireframe_resolution", 6),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_state = StyleState()
_panel = None
_context = None


def render_styled(mw, mol):
    """Style callback: draw `mol` with the per-atom styles in `_state`."""
    v3d = mw.view_3d_manager
    if getattr(v3d, "_drawing_3d", False):
        return
    v3d._drawing_3d = True  # same re-entrancy guard the host uses
    try:
        _render_body(mw, v3d, mol)
        v3d.apply_3d_settings(redraw=False)
    finally:
        v3d._drawing_3d = False
    if _panel is not None:
        _panel.notify_molecule(mol)


def _render_body(mw, v3d, mol):
    import pyvista as pv  # host dependency; imported lazily for headless tests

    settings = mw.init_manager.settings
    plotter = v3d.plotter

    edit_3d = getattr(mw, "edit_3d_manager", None)  # lives on MainWindow, not View3DManager
    clear_sel = getattr(edit_3d, "clear_measurement_selection", None)
    if clear_sel is not None:
        clear_sel()
    v3d._3d_color_map.clear()
    camera_state = plotter.camera_position
    old_axes = getattr(v3d, "axes_actor", None)
    if old_axes is not None:
        try:
            plotter.remove_actor(old_axes)
        except (AttributeError, RuntimeError, TypeError):
            logger.debug("axes removal failed", exc_info=True)
        v3d.axes_actor = None
    plotter.clear()
    plotter.set_background(mw.get_settings().get("background_color", "#919191"))

    if mol is None or mol.GetNumAtoms() == 0:
        v3d.atom_actor = None
        v3d.current_mol = None
        plotter.render()
        return

    lighting = mw.get_settings().get("lighting_enabled", True)
    if lighting:
        plotter.add_light(
            pv.Light(
                position=(1, 1, 2),
                light_type="cameralight",
                intensity=settings.get("light_intensity", 1.2),
            )
        )

    cpk_colors, display_radii, pt = _host_constants(v3d)
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    positions = [tuple(conf.GetAtomPosition(i)) for i in range(n)]
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    colors = [tuple(cpk_colors.get(s, _FALLBACK_COLOR)) for s in symbols]
    for idx, hex_color in getattr(v3d, "_plugin_color_overrides", {}).items():
        rgb = _qcolor_rgb(hex_color)
        if rgb is not None and 0 <= idx < n:
            colors[idx] = rgb
    bond_colors = {
        k: _qcolor_rgb(c)
        for k, c in getattr(v3d, "_plugin_bond_color_overrides", {}).items()
        if _qcolor_rgb(c) is not None
    }
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble(), b.GetIdx())
        for b in mol.GetBonds()
    ]

    keys, mode = atom_keys(mol)
    _state.bind(keys, mode)
    cfg = _make_cfg(settings, display_radii, pt)
    spheres, segments = build_scene(
        symbols, positions, bonds, _state.style_of, colors, cfg, bond_colors
    )

    v3d.atom_positions_3d = np.array(positions)
    src = pv.PolyData(v3d.atom_positions_3d)
    src["colors"] = np.array(colors)
    src["radii"] = np.full(n, 0.3)
    v3d.glyph_source = src

    mesh_props = {
        "smooth_shading": True,
        "specular": settings.get("specular", 0.2),
        "specular_power": settings.get("specular_power", 20),
        "lighting": lighting,
    }

    v3d.atom_actor = None
    by_res = {}
    for idx, pos, radius, res in spheres:
        by_res.setdefault(res, []).append((idx, pos, radius))
    for res, items in by_res.items():
        pts = pv.PolyData(np.array([p for _, p, _ in items]))
        pts["colors"] = np.array([colors[i] for i, _, _ in items])
        pts["radii"] = np.array([r for _, _, r in items])
        glyphs = pts.glyph(
            scale="radii",
            geom=pv.Sphere(radius=1.0, theta_resolution=res, phi_resolution=res),
            orient=False,
        )
        actor = plotter.add_mesh(glyphs, scalars="colors", rgb=True, **mesh_props)
        v3d.atom_actor = v3d.atom_actor or actor
    for idx, _, _, _ in spheres:
        v3d._3d_color_map[f"atom_{idx}"] = [int(c * 255) for c in colors[idx]]

    if segments:
        pts, radii, rgbs, cells = [], [], [], []
        for p0, p1, radius, c0, c1 in segments:
            base = len(pts)
            pts += [p0, p1]
            radii += [radius, radius]
            rgbs += [c0, c1]
            cells += [2, base, base + 1]
        line_mesh = pv.PolyData(np.array(pts), lines=np.array(cells))
        line_mesh["radii"] = np.array(radii)
        line_mesh["colors"] = (np.array(rgbs) * 255).astype(np.uint8)
        tube = line_mesh.tube(
            scalars="radii", absolute=True, n_sides=cfg["stick_resolution"]
        )
        plotter.add_mesh(tube, scalars="colors", rgb=True, **mesh_props)

    for helper, args in (
        ("_add_3d_aromatic_rings", (mol, DEFAULT_STYLE, mesh_props)),
        ("_add_3d_labels", (mol, mol)),
    ):
        fn = getattr(v3d, helper, None)
        if fn is not None:
            try:
                fn(*args)
            except Exception:  # pylint: disable=broad-except
                logger.debug("host helper %s failed", helper, exc_info=True)

    if _panel is not None and _panel.selected:
        draw_highlight(v3d, mol, _panel.selected)

    plotter.camera_position = camera_state
    vcam = getattr(getattr(plotter, "renderer", None), "GetActiveCamera", None)
    if vcam is not None:
        cam = vcam()
        if cam:
            cam.SetParallelProjection(
                mw.get_settings().get("projection_mode", "Perspective")
                == "Orthographic"
            )
    plotter.render()

    if getattr(v3d, "atom_info_display_mode", None) is not None:
        v3d.show_all_atom_info()
    v3d.update_atom_id_menu_text()
    v3d.update_atom_id_menu_state()


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


HIGHLIGHT_NAME = "molecule_styler_highlight"


def draw_highlight(v3d, mol, indices):
    """Yellow translucent shells around `indices`, like the host's 3D-edit selection."""
    import pyvista as pv  # host dependency; imported lazily for headless tests

    plotter = v3d.plotter
    try:
        plotter.remove_actor(HIGHLIGHT_NAME)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        logger.debug("highlight removal failed", exc_info=True)
    indices = [i for i in indices if mol is not None and 0 <= i < mol.GetNumAtoms()]
    positions = getattr(v3d, "atom_positions_3d", None)
    if not indices or positions is None or len(positions) < mol.GetNumAtoms():
        return
    _, display_radii, pt = _host_constants(v3d)
    cfg = _make_cfg({}, display_radii, pt)
    radii = []
    for i in indices:
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        if _state.style_of(i) == "cpk":
            radii.append(cfg["vdw"](sym) * 1.15)
        else:
            radii.append(cfg["display_radius"](sym) * 1.3)
    src = pv.PolyData(np.array([positions[i] for i in indices]))
    src["radii"] = np.array(radii)
    glyphs = src.glyph(
        scale="radii",
        geom=pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=16),
        orient=False,
    )
    plotter.add_mesh(
        glyphs, color="yellow", opacity=0.3, name=HIGHLIGHT_NAME, pickable=False
    )


def _sync_style_menu(mw):
    """Tick this style in the toolbar's "3D Style" menu.

    The host's menu actions form an exclusive group but only tick themselves
    when clicked; switching style from code leaves the old entry (usually
    Ball & Stick) ticked, which looks as if the style never changed.
    """
    init = getattr(mw, "init_manager", None)
    button = getattr(init, "style_button", None)
    menu = button.menu() if button is not None else None
    if menu is None:
        return
    for action in menu.actions():
        if action.text() == STYLE_NAME:
            action.setChecked(True)
            return


class StylerPanel:
    """Right-hand dock listing molecules and atoms with a style combo each."""

    def __init__(self, context, state):
        self.context = context
        self.mw = context.get_main_window()
        self.state = state
        self.fragments = []
        self.symbols = []
        self._signature = None
        self._combos = []  # (combo, indices)
        self.selected = []  # RDKit indices highlighted in 3D

        self.dock = QDockWidget("Molecule Styler", self.mw)
        self.dock.setObjectName("MoleculeStylerDock")
        body = QWidget()
        layout = QVBoxLayout(body)

        self.info = QLabel("")
        layout.addWidget(self.info)

        row = QHBoxLayout()
        row.addWidget(QLabel("All:"))
        self.all_combo = QComboBox()
        self.all_combo.addItems([STYLE_LABELS[s] for s in STYLES])
        self.all_combo.setCurrentIndex(STYLES.index(DEFAULT_STYLE))
        row.addWidget(self.all_combo, 1)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_all)
        row.addWidget(apply_btn)
        layout.addLayout(row)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Molecule / Atom", "Style"])
        self.tree.setColumnWidth(0, int(170 * WIDTH_SCALE))
        self.tree.itemExpanded.connect(self._on_expanded)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.tree, 1)

        row2 = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        reset = QPushButton("Reset styles")
        reset.clicked.connect(self._reset)
        row2.addWidget(refresh)
        row2.addWidget(reset)
        layout.addLayout(row2)

        self.dock.setWidget(body)
        self.mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        width = int(self.dock.sizeHint().width() * WIDTH_SCALE)
        self.mw.resizeDocks([self.dock], [width], Qt.Orientation.Horizontal)

        self.timer = QTimer(self.dock)
        self.timer.timeout.connect(self._poll)
        self.timer.start(POLL_MS)
        self.refresh()

    # -- molecule tracking -------------------------------------------------

    def _current_mol(self):
        return self.context.current_mol

    def notify_molecule(self, mol):
        """Called by the style callback after each draw."""
        self._rebuild_if_changed(mol)

    def _poll(self):
        self._rebuild_if_changed(self._current_mol())

    def refresh(self):
        """Force a rebuild of the molecule list."""
        self._signature = None
        self._rebuild_if_changed(self._current_mol())

    def _rebuild_if_changed(self, mol):
        if mol is None:
            sig = (0, ())
            frags, symbols = [], []
        else:
            n = mol.GetNumAtoms()
            frags = find_fragments(
                n, [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
            )
            symbols = [a.GetSymbol() for a in mol.GetAtoms()]
            keys, mode = atom_keys(mol)
            self.state.bind(keys, mode)
            sig = (n, tuple(tuple(f) for f in frags), tuple(symbols), tuple(keys))
        if sig == self._signature:
            return
        self._signature = sig
        self.fragments, self.symbols = frags, symbols
        # Styles are deliberately NOT pruned here: a redraw, project load or
        # re-embed can briefly present a smaller (or empty) molecule, and
        # dropping entries then would erase the user's styling for good.
        # build_scene only looks up atoms that exist, so stale indices are inert.
        self._populate()

    # -- tree --------------------------------------------------------------

    def _make_combo(self, indices):
        combo = QComboBox()
        combo.addItems([STYLE_LABELS[s] for s in STYLES])
        combo.addItem(MIXED_LABEL)
        combo.activated.connect(
            lambda pos, idx=list(indices), c=combo: self._on_choice(c, idx, pos)
        )
        self._combos.append((combo, list(indices)))
        self._set_combo(combo, indices)
        return combo

    def _set_combo(self, combo, indices):
        common = self.state.common_style(indices)
        pos = STYLES.index(common) if common else len(STYLES)
        combo.blockSignals(True)
        combo.setCurrentIndex(pos)
        combo.blockSignals(False)

    def _populate(self):
        self.tree.clear()
        self._combos = []
        self.info.setText(
            f"{len(self.fragments)} molecule(s), {len(self.symbols)} atoms"
            if self.symbols
            else "No molecule loaded"
        )
        for k, frag in enumerate(self.fragments, 1):
            label = f"Molecule {k}  {formula_of([self.symbols[i] for i in frag])}"
            item = QTreeWidgetItem([label, ""])
            item.setData(0, Qt.ItemDataRole.UserRole, list(frag))
            item.addChild(QTreeWidgetItem(["...", ""]))  # lazy placeholder
            self.tree.addTopLevelItem(item)
            self.tree.setItemWidget(item, 1, self._make_combo(frag))

    def _on_expanded(self, item):
        frag = item.data(0, Qt.ItemDataRole.UserRole)
        if not frag or item.childCount() != 1 or item.child(0).text(0) != "...":
            return
        item.takeChild(0)
        for i in frag:
            child = QTreeWidgetItem([f"{self.symbols[i]}  (id {self.state.key(i)})", ""])
            child.setData(0, Qt.ItemDataRole.UserRole, [i])
            item.addChild(child)
            self.tree.setItemWidget(child, 1, self._make_combo([i]))

    # -- actions -----------------------------------------------------------

    def _on_choice(self, combo, indices, pos):
        if pos >= len(STYLES):  # "(mixed)" is display-only
            self._set_combo(combo, indices)
            return
        self.state.set_atoms(indices, STYLES[pos])
        self._sync_combos()
        self.redraw()

    def _apply_all(self):
        pos = self.all_combo.currentIndex()
        if 0 <= pos < len(STYLES):
            self.state.set_atoms(range(len(self.symbols)), STYLES[pos])
            self._sync_combos()
            self.redraw()

    def _reset(self):
        self.state.clear()
        self._sync_combos()
        self.redraw()

    def _sync_combos(self):
        for combo, indices in self._combos:
            self._set_combo(combo, indices)

    def redraw(self):
        """Redraw the scene, switching to this plugin's style if needed."""
        v3d = getattr(self.mw, "view_3d_manager", None)
        mol = self._current_mol()
        if v3d is None or mol is None:
            return
        if getattr(v3d, "current_3d_style", None) == STYLE_NAME:
            v3d.draw_molecule_3d(mol)
        else:
            v3d.set_3d_style(STYLE_NAME)
        _sync_style_menu(self.mw)
        self.context.mark_project_modified()

    def _on_selection(self):
        """Highlight the atoms of the selected rows in the 3D view."""
        picked = set()
        for item in self.tree.selectedItems():
            picked.update(item.data(0, Qt.ItemDataRole.UserRole) or [])
        self.selected = sorted(picked)
        self.draw_selection()

    def draw_selection(self):
        """(Re)draw the yellow highlight for the current selection."""
        v3d = getattr(self.mw, "view_3d_manager", None)
        mol = self._current_mol()
        if v3d is None or mol is None or getattr(v3d, "plotter", None) is None:
            return
        try:
            draw_highlight(v3d, mol, self.selected)
            v3d.plotter.render()
        except Exception:  # pylint: disable=broad-except
            logger.debug("highlight failed", exc_info=True)

    def activate_style(self):
        """Switch the 3D view to this plugin's style (no-op if already active)."""
        v3d = getattr(self.mw, "view_3d_manager", None)
        if v3d is not None and getattr(v3d, "current_3d_style", None) != STYLE_NAME:
            v3d.set_3d_style(STYLE_NAME)
        _sync_style_menu(self.mw)

    def toggle(self):
        """Show or hide the dock; showing it switches to the plugin's style."""
        show = not self.dock.isVisible()
        self.dock.setVisible(show)
        if show:
            self.activate_style()


# ---------------------------------------------------------------------------
# Plugin entry points
# ---------------------------------------------------------------------------


def _toggle_panel(context):
    global _panel  # pylint: disable=global-statement
    if _panel is None:
        _panel = StylerPanel(context, _state)
        context.register_window("styler_panel", _panel.dock)
        _panel.activate_style()
        return
    _panel.toggle()


def _is_active(context):
    """True while the 3D view is using this plugin's style."""
    mw = context.get_main_window() if context is not None else None
    v3d = getattr(mw, "view_3d_manager", None)
    return getattr(v3d, "current_3d_style", None) == STYLE_NAME


def _select_style(context):
    """Make this plugin's style the current one without drawing.

    Project loading restores the molecule (and draws it) right after the load
    handlers run, so only the style name has to be in place beforehand; going
    through set_3d_style would first redraw the previous molecule.
    """
    mw = context.get_main_window() if context is not None else None
    v3d = getattr(mw, "view_3d_manager", None)
    if v3d is None:
        return
    v3d.current_3d_style = STYLE_NAME
    _sync_style_menu(mw)


def _save():
    active = _is_active(_context)
    if not _state.has_styles() and not active:
        return {}
    return {SAVE_KEY: _state.to_dict(), ACTIVE_KEY: active}


def _load(data):
    data = data if isinstance(data, dict) else {}
    _state.load_dict(data.get(SAVE_KEY))
    if data.get(ACTIVE_KEY) or _state.has_styles():
        _select_style(_context)
    if _panel is not None:
        _panel.refresh()


def _reset_document():
    _state.clear()
    if _panel is not None:
        _panel.refresh()


def initialize(context):
    """Register the style, the View-menu toggle and the persistence handlers."""
    global _context  # pylint: disable=global-statement
    _context = context
    context.register_3d_style(STYLE_NAME, render_styled)
    context.add_menu_action(
        "View/Molecule Styler Panel", lambda: _toggle_panel(context)
    )
    context.register_save_handler(_save)
    context.register_load_handler(_load)
    context.register_document_reset_handler(_reset_document)
