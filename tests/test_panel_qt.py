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
