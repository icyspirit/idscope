import numpy as np
import pytest

pytest.importorskip("pyqtgraph.Qt")

import pyqtgraph as pg  # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui  # noqa: E402

from idscope.config import SignalConfig, load_config  # noqa: E402
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
    # the map panel has psi and the boundary drawn over it, for both sources
    assert [len(panel.curves) for panel in window.panels] == [2, 2, 2, 2, 2, 2, 4]
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
    assert [len(panel.curves) for panel in window.panels] == [1] * 6 + [2]
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
    dialog.contours.setValue(3)
    assert dialog.apply()
    window.refresh_panel(panel)
    assert panel.config.title == "Flux"
    assert panel.config.contours == 3
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


def plot_edges(panel):
    """Left and right edges of the plot area, in window coordinates."""
    rect = panel.plot.getViewBox().sceneBoundingRect()
    view = panel.graphics
    to_window = panel.window().mapFromGlobal
    left = to_window(view.mapToGlobal(view.mapFromScene(rect.topLeft()))).x()
    right = to_window(view.mapToGlobal(view.mapFromScene(rect.topRight()))).x()
    return left, right


def test_shared_x_axes(window):
    traces, (te, q, st), (psi,) = window.column_panels
    master = traces[0].plot.getViewBox()
    for panel in traces[1:] + [st]:
        assert panel.plot.getViewBox().linkedView(0) is master
    assert te.plot.getViewBox().linkedView(0) is None  # rho, not time
    assert q.plot.getViewBox().linkedView(0) is None  # psi, not rho
    shown = [p.plot.getAxis("bottom").style["showValues"] for p in window.panels]
    # stacked time traces only label the lowest one; te (rho) and q (psi) differ
    assert shown == [False, False, True, True, True, True, True]

    window.share_x_action.setChecked(False)
    assert all(p.plot.getViewBox().linkedView(0) is None for p in window.panels)
    assert all(p.plot.getAxis("bottom").style["showValues"] for p in window.panels)


def test_stacked_plots_line_up(window, qtbot):
    qtbot.wait(50)
    for panels in window.column_panels:
        # one of the panels of the second column has a color bar
        assert len({plot_edges(panel) for panel in panels}) == 1


def test_map_overlay_and_contours(window):
    psi = window.panels[6]
    boundary = [c for c in psi.curves if c.item is not None]
    assert [c.name for c in boundary] == ["boundary [A]", "boundary [B]"]
    x_before = boundary[0].item.xData.copy()
    window.set_time(1.0)
    assert not np.allclose(x_before, boundary[0].item.xData)
    assert psi.title_item.textItem.toPlainText() == "Poloidal flux  t = 1 s"
    # contours: 10 -> contour lines only, no filled image
    assert psi._image is None
    assert len(psi._contours) == 10
    assert psi._colorbar.getAxis("left").labelText == "profiles_2d[0]/psi [Wb]"
    psi.config.contours = 0
    window.refresh_panel(psi)
    assert isinstance(psi._image, pg.ImageItem) and psi._contours == []
    assert psi._colorbar.getAxis("left").labelText == "profiles_2d[0]/psi [Wb]"


def test_pan_mode_and_major_grid_by_default(window):
    assert window.mode == "pan"
    assert window.mode_actions["pan"].isChecked()
    for panel in window.panels:
        assert panel.plot.getViewBox().state["mouseMode"] == pg.ViewBox.PanMode
        assert panel.plot.getAxis("left").style["maxTickLevel"] == 0  # no minor grid


def test_slice_time_in_title_or_legend(window):
    te = window.panels[3]
    window.set_time(3.1)
    assert te.title_item.textItem.toPlainText() == "Electron temperature  t = 3 s"
    assert all("@" not in label.text for _, label in te.legend.items)
    # a fixed time slice next to the time-dependent profile: the times differ
    te.add_signal("core_profiles/profiles_1d[4]/electrons/temperature")
    assert te.title_item.textItem.toPlainText() == "Electron temperature"
    # an entry per signal (with its time), and one per source (its line style)
    labels = [label.text for _, label in te.legend.items]
    assert labels[2:] == ["A", "B"]
    assert all("@" in label for label in labels[:2])


def test_wildcards_in_panels(window):
    window.change_layout(window.panels[2], "add_below")
    panel = window.current_panel
    panel.add_signal("magnetics/flux_loop[*]/flux/data")
    assert panel.kind == "trace"
    assert [c.name for c in panel.curves] == [f"FL{k} [{s}]" for s in "AB" for k in (1, 2, 3)]
    # a color per flux loop, a line style per source
    assert [label.text for _, label in panel.legend.items] == ["FL1", "FL2", "FL3", "A", "B"]
    pens = [c.item.opts["pen"] for c in panel.curves]
    assert pens[0].color() == pens[3].color() != pens[1].color()
    assert pens[0].style() != pens[3].style()

    # a value per flux loop: bars, per loop and source
    panel.config.signals = [SignalConfig("magnetics/flux_loop[*]/flux/data[:, -1]")]
    window.refresh_panel(panel)
    assert panel.kind == "profile" and len(panel._bars) == 2
    ticks = panel.plot.getAxis("bottom")._tickLevels
    assert [name for _, name in ticks[0]] == ["FL1", "FL2", "FL3"]
    assert panel.plot.getAxis("bottom").labelText == "flux_loop"
    np.testing.assert_allclose(panel._bars[0].item.opts["height"], [0.1, 0.2, 0.3],
                               atol=0.003)
    assert [label.text for _, label in panel.legend.items] == ["A", "B"]

    # time-dependent values follow the time
    panel.config.signals = [SignalConfig("pf_active/coil[*]/current/data", scale=1e-3)]
    panel.config.kind = "profile"
    window.refresh_panel(panel)
    window.set_time(0.0)
    np.testing.assert_allclose(panel._bars[0].item.opts["height"][:3], [12, 15, 18])
    window.set_time(5.0)
    np.testing.assert_allclose(panel._bars[0].item.opts["height"][:3], [-6, -9, -12])
    assert panel.title_item.textItem.toPlainText() == "t = 5 s"
