import numpy as np
import pytest

pytest.importorskip("pyqtgraph.Qt")

from pyqtgraph.Qt import QtCore, QtGui  # noqa: E402

from idscope.config import load_config  # noqa: E402
from idscope.demo import DEMO_CONFIG  # noqa: E402
from idscope.errors import IdscopeError  # noqa: E402
from idscope.gui.panel import MIME_SIGNAL  # noqa: E402

KINDS = ["trace", "trace", "trace", "profile", "profile", "spacetime", "map"]


@pytest.fixture
def config_path(tmp_path, data_dir):
    text = DEMO_CONFIG.replace("A=demo_A.nc", f"A={data_dir / 'a.nc'}").replace(
        "B=demo_B.nc", f"B={data_dir / 'b.nc'}"
    )
    path = tmp_path / "demo.yaml"
    path.write_text(text)
    return path


@pytest.fixture
def window(qtbot, config_path):
    from idscope.gui import create_application, create_window

    create_application()
    win = create_window(load_config(config_path), config_path, time=3.0)
    qtbot.addWidget(win)
    win.resize(1200, 800)
    win.show()
    return win


def test_demo_layout(window):
    assert [panel.kind for panel in window.panels] == KINDS
    assert all(not panel.errors for panel in window.panels)
    assert [len(panel.curves) for panel in window.panels] == [2, 2, 2, 2, 2, 2, 2]
    assert window.timebar.slider.maximum() == 40
    assert window.panels[6].image_selector.count() == 2


def test_time_moves_profiles_and_maps(window):
    te, psi = window.panels[3], window.panels[6]
    y_before = te.curves[0].item.yData.copy()
    z_before = psi._image_grid[2].copy()
    window.set_time(8.0)
    assert not np.allclose(y_before, te.curves[0].item.yData)
    assert not np.allclose(z_before, psi._image_grid[2])
    assert window.timebar.time == 8.0
    assert window.panels[0]._time_line.value() == 8.0


def test_image_selection_survives_reload(window):
    psi = window.panels[6]
    psi.image_selector.setCurrentIndex(1)
    z_b = psi._image_grid[2].copy()
    window.reload()
    assert psi.image_selector.currentIndex() == 1
    np.testing.assert_allclose(psi._image_grid[2], z_b)


def test_timebar_steps_through_slices(window):
    window.timebar.step(1)
    assert window.time == pytest.approx(3.25)
    window.timebar.step(-2)
    assert window.time == pytest.approx(2.75)
    window.panels[0].timeRequested.emit(5.1)  # e.g. time line dragged
    assert window.time == pytest.approx(5.1)
    window.timebar.step(1)
    assert window.time == pytest.approx(5.25)


def test_apply_sources(window, data_dir):
    window.sources_edit.setText(f"only={data_dir / 'a.nc'}")
    window.apply_sources()
    assert [s.label for s in window.config.sources] == ["only"]
    assert [len(panel.curves) for panel in window.panels] == [1] * 7
    assert window.panels[6].header.isHidden()


def test_sources_relative_to_config(window, config_path, data_dir):
    import shutil

    shutil.copy(data_dir / "b.nc", config_path.parent / "local.nc")
    window.sources_edit.setText("local.nc")  # relative to the configuration file
    window.apply_sources()
    assert window.config.sources[0].uri == str(config_path.parent.resolve() / "local.nc")
    assert window.sources_edit.text() == "local.nc"
    assert len(window.panels[0].curves) == 1


def test_errors_are_shown_in_panel(window):
    panel = window.panels[0]
    panel.add_signal("equilibrium/time_slice/global_quantities/ipp")
    assert len(panel.errors) == 2
    assert len(panel.curves) == 2
    assert not panel.message.isHidden()
    assert "did you mean 'ip'" in panel.message.toolTip()
    panel.add_signal("core_profiles/profiles_1d/electrons/temperature")
    assert "cannot be shown in a trace panel" in panel.message.toolTip()
    assert "not shown" in window.status_label.text()


def test_setup_dialog(window):
    from idscope.gui.setup_dialog import SetupDialog

    panel = window.panels[1]
    dialog = SetupDialog(panel.config, ["A", "B"], window)
    dialog.title.setText("Flux")
    dialog.table.item(0, 2).setText("1000")
    dialog.table.item(0, 6).setText("A")
    assert dialog.apply()
    window.refresh_panel(panel)
    assert panel.config.title == "Flux"
    assert panel.config.signals[0].scale == 1000.0
    assert panel.config.signals[0].sources == ["A"]
    assert len(panel.curves) == 1
    dialog.x_min.setText("abc")
    with pytest.raises(IdscopeError):
        dialog.read_config()
    dialog.x_min.setText("1")
    dialog.x_max.setText("0")
    with pytest.raises(IdscopeError):
        dialog.read_config()


def test_browser(window):
    browser = window.browser
    assert browser.source_combo.count() == 2
    assert browser.ids_combo.findText("core_profiles") >= 0
    browser.ids_combo.setCurrentIndex(browser.ids_combo.findText("summary"))
    assert browser.tree.topLevelItemCount() == 1  # global_quantities
    browser.ids_combo.setCurrentIndex(browser.ids_combo.findText("core_profiles"))
    browser.filter.setText("temperature")
    root = browser.tree.invisibleRootItem()
    visible = [root.child(i).text(0) for i in range(root.childCount())
               if not root.child(i).isHidden()]
    assert visible == ["profiles_1d"]

    window.set_current_panel(window.panels[0])
    browser.addRequested.emit("summary/global_quantities/ip/value", False)
    assert len(window.panels[0].curves) == 4
    n_panels = len(window.panels)
    browser.addRequested.emit("core_profiles/profiles_1d/electrons/density", True)
    assert len(window.panels) == n_panels + 1
    assert window.current_panel.kind == "profile"


def test_drop_signal_on_panel(window):
    panel = window.panels[4]
    mime = QtCore.QMimeData()
    mime.setData(MIME_SIGNAL, QtCore.QByteArray(b"equilibrium/time_slice/profiles_1d/psi"))
    event = QtGui.QDropEvent(
        QtCore.QPointF(20, 20), QtCore.Qt.DropAction.CopyAction, mime,
        QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
    )
    panel.dropEvent(event)
    assert len(panel.config.signals) == 2
    assert len(panel.curves) == 4
    assert window.current_panel is panel


def test_layout_changes(window):
    n = len(window.panels)
    window.change_layout(window.panels[0], "add_below")
    assert len(window.panels) == n + 1
    assert window.panels[1].config.signals == []
    assert window.current_panel is window.panels[1]
    window.change_layout(window.panels[1], "remove")
    assert len(window.panels) == n
    window.change_layout(window.panels[-1], "add_column")
    assert len(window.config.columns) == 4
    assert window.panels[0].curves  # data is shown again after the rebuild


def test_save_and_reopen(window, tmp_path):
    path = tmp_path / "saved.yaml"
    window._save_to(path)
    config = load_config(path)
    assert config.time == 3.0
    assert len(config.panels()) == 7
    window.load_config_file(path)
    assert [panel.kind for panel in window.panels] == KINDS


def test_point_mode_values(window):
    window.set_mode("point")
    window.set_current_panel(window.panels[0])
    window.set_time(5.0)
    assert "Ip (equilibrium) [A] = 1" in window.status_label.text()
    assert "Ip (equilibrium) [B] = 0.8" in window.status_label.text()


def test_export_image(window, tmp_path):
    path = tmp_path / "scope.png"
    window.export_image(path)
    assert path.stat().st_size > 10000
