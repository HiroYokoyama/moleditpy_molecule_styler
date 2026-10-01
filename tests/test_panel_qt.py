"""Panel behaviour that needs a real Qt (skipped when PyQt6 is absent or stubbed)."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6.QtTest")
if not hasattr(sys.modules.get("PyQt6"), "__file__"):  # a stub from another test file
    pytest.skip("PyQt6 is stubbed in this process", allow_module_level=True)

from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QTreeWidgetItem  # noqa: E402

_PATH = Path(__file__).resolve().parents[1] / "molecule_styler.py"
_spec = importlib.util.spec_from_file_location("molecule_styler_qt", _PATH)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)


def test_click_below_the_rows_calls_back_but_a_row_click_does_not(qapp):
    calls = []
    tree, body = ms._make_clearing_widgets(lambda: calls.append(1))
    for name in ("one", "two"):
        tree.addTopLevelItem(QTreeWidgetItem([name]))
    tree.resize(300, 300)
    tree.show()
    qapp.processEvents()

    row_rect = tree.visualItemRect(tree.topLevelItem(0))
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=row_rect.center())
    assert calls == []
    assert tree.selectedItems()

    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(20, 280))
    assert calls == [1]

    body.show()
    QTest.mouseClick(body, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    assert calls == [1, 1]
    tree.close()
    body.close()


def test_empty_click_unselects_through_the_panel(qapp):
    from PyQt6.QtWidgets import QMainWindow

    mw = QMainWindow()
    mw.view_3d_manager = MagicMock()
    ctx = MagicMock()
    ctx.get_main_window.return_value = mw
    ctx.current_mol = None
    panel = ms.StylerPanel(ctx, ms.StyleState())
    panel.tree.addTopLevelItem(QTreeWidgetItem(["a row"]))
    panel.tree.topLevelItem(0).setSelected(True)
    panel.selected = [0]
    mw.show()
    qapp.processEvents()
    QTest.mouseClick(panel.tree.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(10, panel.tree.viewport().height() - 5))
    assert panel.selected == [] and not panel.tree.selectedItems()
    panel.timer.stop()
    mw.close()


def _load_copy(name):
    """A second, independent copy of the plugin module, like a plugin reload."""
    spec = importlib.util.spec_from_file_location(name, _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _window_and_context():
    from PyQt6.QtWidgets import QMainWindow

    mw = QMainWindow()
    mw.view_3d_manager = MagicMock()
    mw.view_3d_manager.current_3d_style = "cpk"
    ctx = MagicMock()
    ctx.get_main_window.return_value = mw
    ctx.current_mol = None
    return mw, ctx


def _flush_deletes():
    """Run the deleteLater() calls the plugin queued (the app's event loop does this)."""
    from PyQt6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _docks(mw):
    from PyQt6.QtWidgets import QDockWidget

    return mw.findChildren(QDockWidget, ms.DOCK_OBJECT_NAME)


def test_reloaded_module_does_not_leave_two_panels_open(qapp):
    mw, ctx = _window_and_context()
    first, second = _load_copy("styler_copy_a"), _load_copy("styler_copy_b")
    mw.show()
    first.ensure_panel(ctx)
    assert len(_docks(mw)) == 1
    second.ensure_panel(ctx)  # new module instance: its own _panel is still None
    _flush_deletes()
    qapp.processEvents()
    assert len(_docks(mw)) == 1
    visible = [d for d in _docks(mw) if d.isVisible()]
    assert len(visible) == 1 and visible[0] is second._panel.dock
    second._panel.timer.stop()  # the first copy's timer died with its removed dock
    mw.close()


def test_watcher_removes_a_duplicate_dock(qapp):
    mw, ctx = _window_and_context()
    old, new = _load_copy("styler_copy_c"), _load_copy("styler_copy_d")
    mw.show()
    old.ensure_panel(ctx)
    new._panel = None
    new.ensure_panel(ctx)
    old.ensure_panel.__globals__["_panel"].timer.stop()
    # a stray dock appears again (e.g. created by a stale watcher) -> the tick heals it
    old.StylerPanel(ctx, old.StyleState())
    assert len(_docks(mw)) >= 2
    new._watch_style(ctx)
    _flush_deletes()
    assert [d for d in _docks(mw) if d is new._panel.dock]
    assert len(_docks(mw)) == 1
    new._panel.timer.stop()  # the stray's timer died with its dock
    mw.close()


def test_initialize_stops_the_watcher_of_a_previous_module_copy(qapp):
    mw, ctx = _window_and_context()
    old, new = _load_copy("styler_copy_e"), _load_copy("styler_copy_f")
    old.initialize(ctx)
    old_timer = old._watch_timer
    assert old_timer.isActive()
    new.initialize(ctx)  # module reloaded: the old copy's timer must not keep running
    assert not old_timer.isActive() and new._watch_timer.isActive()
    new._watch_timer.stop()
