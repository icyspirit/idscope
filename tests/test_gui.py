import numpy as np
import pytest

pytest.importorskip("pyqtgraph.Qt")

import pyqtgraph as pg  # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets  # noqa: E402

from idscope.config import SignalConfig, load_config  # noqa: E402
from idscope.data import parse_signal_path  # noqa: E402
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
def window(make_window, config_path):
    win = make_window(load_config(config_path), config_path, time=3.0)
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
    assert browser.tree.topLevelItemCount() == 2  # global_quantities, fusion
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
    panel.add_signal("magnetics/flux_loop[:]/flux/data")
    assert panel.kind == "trace"
    assert [c.name for c in panel.curves] == [f"FL{k} [{s}]" for s in "AB" for k in (1, 2, 3)]
    # a color per flux loop, a line style per source
    assert [label.text for _, label in panel.legend.items] == ["FL1", "FL2", "FL3", "A", "B"]
    pens = [c.item.opts["pen"] for c in panel.curves]
    assert pens[0].color() == pens[3].color() != pens[1].color()
    assert pens[0].style() != pens[3].style()

    # a value per flux loop: bars, per loop and source
    panel.config.signals = [SignalConfig("magnetics/flux_loop[:]/flux/data[-1]")]
    window.refresh_panel(panel)
    assert panel.kind == "profile" and len(panel._bars) == 2
    ticks = panel.plot.getAxis("bottom")._tickLevels
    assert [name for _, name in ticks[0]] == ["FL1", "FL2", "FL3"]
    assert panel.plot.getAxis("bottom").labelText == "flux_loop"
    np.testing.assert_allclose(panel._bars[0].item.opts["height"], [0.1, 0.2, 0.3],
                               atol=0.003)
    assert [label.text for _, label in panel.legend.items] == ["A", "B"]

    # time-dependent values follow the time
    panel.config.signals = [SignalConfig("pf_active/coil[:]/current/data", scale=1e-3)]
    panel.config.kind = "profile"
    window.refresh_panel(panel)
    window.set_time(0.0)
    np.testing.assert_allclose(panel._bars[0].item.opts["height"][:3], [12, 15, 18])
    window.set_time(5.0)
    np.testing.assert_allclose(panel._bars[0].item.opts["height"][:3], [-6, -9, -12])
    assert panel.title_item.textItem.toPlainText() == "t = 5 s"
    # the bars at the selected time are the sample closest to it: data[k]
    window.set_time(1.3)  # closest to the sample at 1.25 s, the sixth
    source = window.pool.get(window.config.sources[0])
    sample = source.signal(parse_signal_path("pf_active/coil[:]/current/data[5]"))
    np.testing.assert_allclose(panel._bars[0].item.opts["height"],
                               [1e-3 * float(m.values) for m in sample.members])


def test_modified_mark(window, config_path, data_dir, tmp_path, monkeypatch):
    from idscope.gui.setup_dialog import SetupDialog
    from idscope.session import session_path

    assert window.windowTitle() == "idscope — idscope demo[*]"  # Qt shows [*] as *
    assert not window.isWindowModified()
    # the sources and the time are not changes of the panels
    window.set_time(5.0)
    window.sources_edit.setText(f"A={data_dir / 'a.nc'}")
    window.apply_sources()
    assert not window.isWindowModified()

    window.change_layout(window.panels[0], "add_below")
    assert window.isWindowModified()
    window._save_to(tmp_path / "saved.yaml")
    assert not window.isWindowModified()
    assert window.windowTitle() == "idscope — idscope demo[*]"

    window.panels[0].add_signal("summary/global_quantities/ip/value")
    assert window.isWindowModified()
    window.load_config_file(config_path)
    assert not window.isWindowModified()

    # the setup dialog, when something changed in it
    def setup(change):
        def exec_(dialog):
            change(dialog)
            dialog.apply()
            return 1
        monkeypatch.setattr(SetupDialog, "exec", exec_)
        window.open_setup(window.panels[1])

    setup(lambda dialog: None)
    assert not window.isWindowModified()
    setup(lambda dialog: dialog.title.setText("Flux"))
    assert window.isWindowModified()
    assert window.panels[1].config.title == "Flux"

    def drag():
        window.columns_splitter.setSizes([200, 500, 500])
        window.columns_splitter.splitterMoved.emit(200, 1)

    for change in (lambda: window.share_x_action.trigger(), drag):
        window.load_config_file(config_path)
        change()
        assert window.isWindowModified()

    window.new_from_template("equilibrium")
    assert not window.isWindowModified()
    assert window.windowTitle() == "idscope — equilibrium[*]"
    window.new_config()
    assert window.windowTitle() == "idscope[*]"
    assert [s.label for s in window.config.sources] == ["A", "B"]  # kept

    window.close()
    assert not session_path().exists()  # only the window of `idscope` keeps a session


def test_recent_files(window, config_path, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda parent, title, text: warnings.append(text))
    saved = tmp_path / "saved.yaml"
    window._save_to(saved)
    assert window.recent_files == [str(saved), str(config_path)]
    assert window.recent_menu.actions()[0].text().startswith("&1 ")

    saved.unlink()
    window.recent_menu.actions()[0].trigger()
    assert "Cannot read" in warnings[0]
    assert window.recent_files == [str(config_path)]  # forgotten
    window.recent_menu.actions()[0].trigger()
    assert window.config_path == config_path
    assert len(window.panels) == len(KINDS)

    clear = window.recent_menu.actions()[-1]
    assert clear.text() == "Clear list"
    clear.trigger()
    assert window.recent_files == [] and not window.recent_menu.isEnabled()


def test_session_and_settings(qtbot, make_window, config_path):
    from idscope.session import load_session, session_path

    first = make_window(load_config(config_path), config_path, persist=True)
    first.resize(760, 560)
    first.show()
    size = first.size()
    first.set_mode("zoom")
    first.browser_dock.hide()
    first.browser.ids_combo.setCurrentIndex(first.browser.ids_combo.findText("summary"))
    first.change_layout(first.panels[0], "remove")
    first.set_time(4.0)
    # the session is written a moment after a change, before the window closes
    qtbot.waitUntil(session_path().is_file, timeout=5000)
    assert load_session().modified
    first.close()

    session = load_session()
    assert session.config_path == config_path
    assert session.modified
    assert session.config.time == 4.0
    assert len(session.config.panels()) == len(KINDS) - 1

    second = make_window(session.config, session.config_path,
                         modified=session.modified, persist=True)
    second.show()
    assert second.isWindowModified()
    assert len(second.panels) == len(KINDS) - 1
    assert second.time == 4.0
    assert second.mode == "zoom" and second.mode_actions["zoom"].isChecked()
    # the size of the first window (not 1300 x 850), within the 800 x 600 screen of
    # the offscreen platform
    assert second.height() == size.height() and second.width() <= size.width()
    assert not second.browser_dock.isVisible()
    assert second.browser.ids_combo.currentText() == "summary"
    assert second.recent_files == [str(config_path)]


def test_sizes_are_kept_until_dragged(window, tmp_path):
    from idscope.templates import load_template

    window.new_from_template("pf_active")
    path = tmp_path / "pf_active.yaml"
    window._save_to(path)
    template = load_template("pf_active")
    assert [c.width for c in load_config(path).columns] == [c.width for c in template.columns]
    assert [p.height for p in load_config(path).panels()] == [
        p.height for p in template.panels()
    ]
    # dragged by the user
    window.columns_splitter.setSizes([300, 700])
    window.columns_splitter.splitterMoved.emit(300, 1)
    window._save_to(path)
    widths = [c.width for c in load_config(path).columns]
    assert widths[1] / widths[0] == pytest.approx(7 / 3, rel=0.02)


def test_time_of_the_configuration(make_window, config_path):
    config = load_config(config_path)
    config.time = 7.0
    win = make_window(config, config_path)
    assert win.time == 7.0
    assert win.panels[0]._time_line.value() == 7.0
    # the time bar shows it too
    assert win.timebar.spin.value() == 7.0
    assert win.timebar.grid[win.timebar.slider.value()] == 7.0


def test_browser_when_filled_quantities_cannot_be_listed(old_access_layer, window):
    import imas

    browser = window.browser
    window.reload()  # list again, now by reading the data
    browser.ids_combo.setCurrentIndex(browser.ids_combo.findText("core_profiles"))
    assert "found by reading some of the data" in browser.status.text()
    assert "Access Layer version 5.7" in browser.status.toolTip()
    browser.filter.setText("temperature")
    root = browser.tree.invisibleRootItem()
    assert [root.child(i).text(0) for i in range(root.childCount())
            if not root.child(i).isHidden()] == ["profiles_1d"]

    imas.DBEntry.list_filled_paths = old_access_layer  # listed again
    window.reload()
    assert "found by reading" not in browser.status.text()
    assert browser.status.toolTip() == ""


def activate(window, qtbot):
    """Make ``window`` the active window, for its keyboard shortcuts."""
    window.activateWindow()
    qtbot.waitUntil(lambda: QtWidgets.QApplication.activeWindow() is window)


def test_delete_and_backspace(window, qtbot):
    Key = QtCore.Qt.Key
    activate(window, qtbot)
    first, second = window.panels[:2]
    window.set_current_panel(first)
    first.graphics.setFocus()
    qtbot.keyClick(first.graphics, Key.Key_Delete)  # the signals first
    assert first.config.signals == [] and len(window.panels) == len(KINDS)
    assert window.isWindowModified()
    qtbot.keyClick(first.graphics, Key.Key_Backspace)  # then the empty panel
    assert len(window.panels) == len(KINDS) - 1
    assert window.current_panel.config is second.config  # the one that took its place
    # the keyboard stays in the panels, for the next key
    assert window._panels_have_focus()
    qtbot.keyClick(QtWidgets.QApplication.focusWidget(), Key.Key_Delete)
    assert second.config.signals == []

    # text fields keep their keys
    window.sources_edit.setFocus()
    text = window.sources_edit.text()
    qtbot.keyClick(window.sources_edit, Key.Key_Backspace)
    assert window.sources_edit.text() == text[:-1]
    assert len(window.panels) == len(KINDS) - 1

    # all undone: the panel is back, with its signal, and is the current one again
    for _ in range(3):
        window.undo()
    assert len(window.panels) == len(KINDS)
    assert window.current_panel.config.title == "Plasma current"
    assert [len(p.config.signals) for p in window.panels[:2]] == [1, 1]
    assert not window.isWindowModified()

    # the last panel stays when it is empty
    window.new_config()
    window.delete_in_current_panel()
    assert len(window.panels) == 1
    assert window.undo_action.text() == "&Undo new"


def test_undo_and_redo(window, data_dir):
    assert not window.undo_action.isEnabled() and not window.redo_action.isEnabled()
    te = window.panels[3]
    window.set_current_panel(te)
    window.change_layout(te, "add_below")
    assert window.undo_action.text() == "&Undo add panel"
    window.current_panel.add_signal("summary/global_quantities/ip/value")
    window.set_time(7.0)

    window.undo()  # the signal
    assert window.current_panel.config.signals == []
    assert window.time == 7.0  # the time is not undone
    window.undo()  # the panel
    assert len(window.panels) == len(KINDS)
    assert window.current_panel.config.title == "Electron temperature"
    assert not window.isWindowModified()  # as in the file again
    window.redo()
    assert len(window.panels) == len(KINDS) + 1 and window.isWindowModified()
    assert window.redo_action.text() == "&Redo add signal"
    window.change_layout(window.current_panel, "remove")
    assert not window.redo_action.isEnabled()  # a new change starts a new branch

    # the sources can be undone, but do not count as a change of the panels
    window.sources_edit.setText(f"only={data_dir / 'a.nc'}")
    window.apply_sources()
    window.undo()
    assert [s.label for s in window.config.sources] == ["A", "B"]
    assert [len(p.curves) for p in window.panels[:2]] == [2, 2]

    # a drag of a splitter is one change, however many steps it moves in
    window.columns_splitter.setSizes([200, 500, 500])
    window.columns_splitter.splitterMoved.emit(200, 1)
    window.columns_splitter.setSizes([300, 400, 500])
    window.columns_splitter.splitterMoved.emit(300, 1)
    assert window.undo_action.text() == "&Undo panel sizes"
    window.undo()
    assert window.undo_action.text() == "&Undo remove panel"  # the sources were undone
    assert [c.width for c in window.config.columns] == [1.0, 1.0, 1.0]


def test_undo_open_new_and_template(window, config_path, tmp_path):
    window.change_layout(window.panels[0], "remove")
    saved = tmp_path / "saved.yaml"
    window._save_to(saved)
    window.new_from_template("pf_active")
    assert window.config_path is None
    window.new_config()
    assert len(window.panels) == 1

    window.undo()
    assert window.config.title == "pf_active" and window.config_path is None
    window.undo()  # back to the saved file
    assert window.config_path == saved and not window.isWindowModified()
    assert window.windowTitle() == "idscope — idscope demo[*]"
    assert window.redo_action.text() == "&Redo template pf_active"
    # before the save: 7 panels, unlike that file
    window.undo()
    assert len(window.panels) == len(KINDS)
    assert window.config_path == saved and window.isWindowModified()
    window.redo()
    assert not window.isWindowModified()


def test_setup_dialog_delete_key(window, qtbot):
    from idscope.gui.setup_dialog import SetupDialog

    Key = QtCore.Qt.Key
    psi = window.panels[6]
    dialog = SetupDialog(psi.config, ["A", "B"], window)
    qtbot.addWidget(dialog)
    dialog.show()
    activate(dialog, qtbot)
    table = dialog.table
    assert table.rowCount() == 2
    table.setFocus()
    table.selectRow(1)
    qtbot.keyClick(table, Key.Key_Delete)
    assert table.rowCount() == 1
    # while a cell is edited, the keys edit its text
    table.editItem(table.item(0, 0))
    editor = QtWidgets.QApplication.focusWidget()
    assert isinstance(editor, QtWidgets.QLineEdit)
    text = editor.text()
    editor.end(False)
    qtbot.keyClick(editor, Key.Key_Backspace)
    assert editor.text() == text[:-1] and table.rowCount() == 1
    dialog.reject()


def test_a_drag_after_a_save_is_undone_on_its_own(window, tmp_path):
    splitter = window.columns_splitter
    splitter.setSizes([200, 500, 500])
    splitter.splitterMoved.emit(200, 1)
    window._save_to(tmp_path / "saved.yaml")
    splitter.setSizes([300, 400, 500])
    splitter.splitterMoved.emit(300, 1)
    assert window.isWindowModified()
    window.undo()
    assert not window.isWindowModified()  # back to the saved sizes


def test_members_have_one_color_in_the_window(window):
    window.change_layout(window.panels[2], "add_below")
    loops = window.current_panel
    loops.add_signal("magnetics/flux_loop[:]/flux/data")
    window.change_layout(loops, "add_below")
    some = window.current_panel
    some.add_signal("magnetics/flux_loop[1:]/flux/data")  # without FL1

    def colors(panel):
        return {c.name: c.item.opts["pen"].color().name() for c in panel.curves}

    all_loops = colors(loops)
    assert colors(some) == {name: all_loops[name] for name in colors(some)}
    # a scope shown afterwards starts again from the first color
    window.new_from_template("pf_active")
    assert window.member_colors[("coil", "CS3U", None)] == 0
