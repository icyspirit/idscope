"""The idscope main window."""

import logging
import time as _time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from idscope import __version__
from idscope.config import (
    ColumnConfig,
    PanelConfig,
    ScopeConfig,
    default_config,
    load_config,
    save_config,
)
from idscope.data.sources import (
    DataSource,
    SourcePool,
    format_source_list,
    parse_source_list,
)
from idscope.errors import IdscopeError
from idscope.gui.browser import BrowserWidget
from idscope.gui.panel import Panel
from idscope.gui.setup_dialog import SetupDialog
from idscope.gui.timebar import TimeBar

logger = logging.getLogger(__name__)

MAX_TRACE_STEPS = 2001
"""Number of slider steps when only traces (no profiles or maps) are shown."""


@contextmanager
def _busy():
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


def _normalized(sizes: List[int]) -> List[float]:
    """Relative sizes with mean 1 (so that equal sizes are all 1.0)."""
    total = sum(sizes)
    if not sizes or total <= 0:
        return [1.0] * len(sizes)
    mean = total / len(sizes)
    return [round(size / mean, 3) for size in sizes]


class MainWindow(QtWidgets.QMainWindow):
    """A scope window: sources, a grid of panels and a time bar."""

    def __init__(self, config: Optional[ScopeConfig] = None, config_path=None):
        super().__init__()
        self.config = config if config is not None else default_config()
        self.config_path: Optional[Path] = Path(config_path) if config_path else None
        self.pool = SourcePool(self.config.dd_version)
        self.panels: List[Panel] = []
        self.current_panel: Optional[Panel] = None
        self.time: Optional[float] = self.config.time
        self.mode = "zoom"
        self.resize(1300, 850)

        self._build_toolbar()
        self.timebar = TimeBar()
        self.timebar.timeChanged.connect(self.set_time)
        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(2, 2, 2, 2)
        central_layout.setSpacing(2)
        self.panel_area = QtWidgets.QWidget()
        self.panel_area_layout = QtWidgets.QVBoxLayout(self.panel_area)
        self.panel_area_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(self.panel_area, stretch=1)
        central_layout.addWidget(self.timebar)
        self.setCentralWidget(central)
        self.columns_splitter: Optional[QtWidgets.QSplitter] = None
        self._build_docks()
        self._build_menus()
        self.status_label = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status_label, stretch=1)
        self.coordinates_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.coordinates_label)
        self._logged_notes = set()

        self._show_sources()
        self.build_panels()
        self.refresh_all()
        self.browser.set_sources(self.sources())
        self._update_title()

    # ------------------------------------------------------------------ build

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        self.mode_group = QtGui.QActionGroup(self)
        self.mode_actions = {}
        for mode, text, tip in (
            ("zoom", "Zoom", "Drag a rectangle to zoom, right-drag to scale, wheel to zoom"),
            ("pan", "Pan", "Drag to move the view, wheel to zoom"),
            ("point", "Point", "Click or drag in a time trace to set the time, "
                               "read values in the status bar"),
        ):
            action = QtGui.QAction(text, self)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(lambda checked, m=mode: self.set_mode(m))
            self.mode_group.addAction(action)
            toolbar.addAction(action)
            self.mode_actions[mode] = action
        self.mode_actions["zoom"].setChecked(True)
        toolbar.addSeparator()
        autoscale = toolbar.addAction("Autoscale all")
        autoscale.setToolTip("Autoscale all panels (a double-click opens the setup)")
        autoscale.triggered.connect(self.autoscale_all)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel(" Sources: "))
        self.sources_edit = QtWidgets.QLineEdit()
        self.sources_edit.setPlaceholderText(
            "e.g. shot1.nc, ref=imas:hdf5?path=/path/to/entry  (comma separated)"
        )
        self.sources_edit.setToolTip(
            "Data sources, separated by commas: IMAS URIs, netCDF files or HDF5 "
            "entry directories.\nA label can be given with label=source. "
            "All panels show every source unless a signal selects some."
        )
        self.sources_edit.returnPressed.connect(self.apply_sources)
        self.sources_edit.setMinimumWidth(300)
        toolbar.addWidget(self.sources_edit)
        apply = toolbar.addAction("Apply")
        apply.setToolTip("Load the data of these sources (Enter)")
        apply.triggered.connect(self.apply_sources)
        reload_action = toolbar.addAction("Reload")
        reload_action.setToolTip("Read the data again, e.g. after it changed on disk (F5)")
        reload_action.setShortcut(QtGui.QKeySequence("F5"))
        reload_action.triggered.connect(self.reload)

    def _build_docks(self) -> None:
        self.browser = BrowserWidget()
        self.browser.addRequested.connect(self.add_signal)
        self.browser_dock = QtWidgets.QDockWidget("IDS browser", self)
        self.browser_dock.setObjectName("browser_dock")
        self.browser_dock.setWidget(self.browser)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.browser_dock)
        self.resizeDocks([self.browser_dock], [330], QtCore.Qt.Orientation.Horizontal)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_dock = QtWidgets.QDockWidget("Log", self)
        self.log_dock.setObjectName("log_dock")
        self.log_dock.setWidget(self.log_view)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()

    def _build_menus(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        for text, shortcut, slot in (
            ("&New", QtGui.QKeySequence.StandardKey.New, self.new_config),
            ("&Open…", QtGui.QKeySequence.StandardKey.Open, self.open_config),
            ("&Save", QtGui.QKeySequence.StandardKey.Save, self.save),
            ("Save &as…", QtGui.QKeySequence.StandardKey.SaveAs, self.save_as),
            (None, None, None),
            ("&Export image…", QtGui.QKeySequence("Ctrl+E"), self.export_image_dialog),
            (None, None, None),
            ("&Quit", QtGui.QKeySequence.StandardKey.Quit, self.close),
        ):
            if text is None:
                file_menu.addSeparator()
                continue
            action = file_menu.addAction(text)
            action.setShortcut(QtGui.QKeySequence(shortcut))
            action.triggered.connect(slot)

        layout_menu = menu.addMenu("&Layout")
        layout_menu.addAction("Add panel below current").triggered.connect(
            lambda: self.change_layout(self.current_panel, "add_below")
        )
        layout_menu.addAction("Add column").triggered.connect(
            lambda: self.change_layout(self.current_panel, "add_column")
        )
        layout_menu.addAction("Remove current panel").triggered.connect(
            lambda: self.change_layout(self.current_panel, "remove")
        )
        layout_menu.addSeparator()
        layout_menu.addAction("Setup current panel…").triggered.connect(
            lambda: self.open_setup(self.current_panel)
        )

        view_menu = menu.addMenu("&View")
        browser_action = self.browser_dock.toggleViewAction()
        browser_action.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        view_menu.addAction(browser_action)
        view_menu.addAction(self.log_dock.toggleViewAction())
        view_menu.addSeparator()
        self.link_action = view_menu.addAction("Link time axes")
        self.link_action.setCheckable(True)
        self.link_action.setChecked(self.config.link_time_axes)
        self.link_action.toggled.connect(self._set_link_time_axes)
        view_menu.addAction("Autoscale all").triggered.connect(self.autoscale_all)
        view_menu.addSeparator()
        for text, shortcut, delta in (("Previous time slice", "Ctrl+Left", -1),
                                      ("Next time slice", "Ctrl+Right", 1)):
            action = view_menu.addAction(text)
            action.setShortcut(QtGui.QKeySequence(shortcut))
            action.triggered.connect(lambda checked=False, d=delta: self.timebar.step(d))

        help_menu = menu.addMenu("&Help")
        help_menu.addAction("About idscope").triggered.connect(self._about)

    # ----------------------------------------------------------------- panels

    def build_panels(self) -> None:
        """Create the panel widgets from ``config.columns``."""
        if self.columns_splitter is not None:
            self.columns_splitter.setParent(None)
            self.columns_splitter.deleteLater()
        self.panels = []
        if not self.config.columns:
            self.config.columns = [ColumnConfig([PanelConfig()])]
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        for column in self.config.columns:
            if not column.panels:
                column.panels.append(PanelConfig())
            column_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            column_splitter.setChildrenCollapsible(False)
            for panel_config in column.panels:
                panel = self._create_panel(panel_config)
                column_splitter.addWidget(panel)
                self.panels.append(panel)
            splitter.addWidget(column_splitter)
            # Qt distributes stretch factors, which keeps relative sizes on resize
            for i, panel_config in enumerate(column.panels):
                column_splitter.setStretchFactor(i, max(int(panel_config.height * 100), 1))
            column_splitter.setSizes(
                [max(int(p.height * 1000), 1) for p in column.panels]
            )
        for i, column in enumerate(self.config.columns):
            splitter.setStretchFactor(i, max(int(column.width * 100), 1))
        splitter.setSizes([max(int(c.width * 1000), 1) for c in self.config.columns])
        self.columns_splitter = splitter
        self.panel_area_layout.addWidget(splitter)
        if self.current_panel not in self.panels:
            self.current_panel = None
        self.set_current_panel(self.current_panel or self.panels[0])
        for panel in self.panels:
            panel.set_mode(self.mode)

    def _create_panel(self, config: PanelConfig) -> Panel:
        panel = Panel(config)
        panel.activated.connect(self.set_current_panel)
        panel.timeRequested.connect(self.set_time)
        panel.setupRequested.connect(self.open_setup)
        panel.layoutRequested.connect(self.change_layout)
        panel.signalsChanged.connect(self.refresh_panel)
        panel.hoverMessage.connect(self._show_hover)
        return panel

    def set_current_panel(self, panel: Optional[Panel]) -> None:
        if self.current_panel is not None and self.current_panel in self.panels:
            self.current_panel.set_current(False)
        self.current_panel = panel
        if panel is not None:
            panel.set_current(True)

    def sync_layout_to_config(self) -> None:
        """Store splitter sizes as relative widths/heights in the configuration."""
        if self.columns_splitter is None:
            return
        widths = _normalized(self.columns_splitter.sizes())
        for i, column in enumerate(self.config.columns):
            column.width = widths[i]
            column_splitter = self.columns_splitter.widget(i)
            heights = _normalized(column_splitter.sizes())
            for panel_config, height in zip(column.panels, heights):
                panel_config.height = height

    def _locate(self, panel: Panel):
        for c, column in enumerate(self.config.columns):
            for p, panel_config in enumerate(column.panels):
                if panel_config is panel.config:
                    return c, p
        raise ValueError("panel not in configuration")

    def change_layout(self, panel: Optional[Panel], action: str) -> None:
        """Add a panel below ``panel``, add a column after it, or remove it."""
        if panel is None:
            panel = self.panels[0] if self.panels else None
        self.sync_layout_to_config()
        c, p = self._locate(panel) if panel is not None else (0, 0)
        columns = self.config.columns
        new_config = None
        if action == "add_below":
            new_config = PanelConfig()
            columns[c].panels.insert(p + 1, new_config)
        elif action == "add_column":
            new_config = PanelConfig()
            columns.insert(c + 1, ColumnConfig([new_config]))
        elif action == "remove":
            del columns[c].panels[p]
            if not columns[c].panels:
                del columns[c]
            if not columns:
                columns.append(ColumnConfig([PanelConfig()]))
        self.build_panels()
        if new_config is not None:
            self.set_current_panel(next(x for x in self.panels if x.config is new_config))
        self.refresh_all()

    # ------------------------------------------------------------------- data

    def sources(self) -> List[DataSource]:
        return [self.pool.get(spec) for spec in self.config.sources]

    def apply_sources(self) -> None:
        """Use the sources typed in the Sources field (like Apply in jScope)."""
        try:
            specs = parse_source_list(self.sources_edit.text(), self._base_dir())
        except IdscopeError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid source", str(exc))
            return
        self.config.sources = specs
        self.pool.keep_only(specs)
        self._show_sources()
        self.refresh_all()
        self.browser.set_sources(self.sources())

    def reload(self) -> None:
        """Read all data again."""
        self.pool.reload()
        self._logged_notes.clear()
        self.refresh_all()
        self.browser.set_sources(self.sources())

    def refresh_all(self) -> None:
        start = _time.perf_counter()
        errors = []
        with _busy():
            sources = self.sources()
            for panel in self.panels:
                errors += self._load_panel(panel, sources)
            self._update_time_grid()
            self._link_time_axes()
        self._report(errors, _time.perf_counter() - start)

    def refresh_panel(self, panel: Panel) -> None:
        start = _time.perf_counter()
        with _busy():
            errors = self._load_panel(panel, self.sources())
            self._update_time_grid()
            self._link_time_axes()
        self._report(errors, _time.perf_counter() - start)

    def _load_panel(self, panel: Panel, sources: List[DataSource]) -> List[str]:
        errors = panel.load(sources, self.time)
        name = panel.title()
        return [f"[{name}] {error}" for error in errors]

    def _report(self, errors: List[str], seconds: float) -> None:
        for source in self.config.sources:
            for note in self.pool.get(source).notes:
                if note not in self._logged_notes:
                    self._logged_notes.add(note)
                    self.log(note)
        for error in errors:
            self.log(error)
        n_curves = sum(len(panel.curves) for panel in self.panels)
        text = f"{n_curves} signals loaded in {seconds:.2f} s"
        if errors:
            text += f"; {len(errors)} not shown (see View → Log)"
        self.status_label.setText(text)

    def log(self, message: str) -> None:
        logger.info(message)
        stamp = _time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"{stamp}  {message}")

    # ------------------------------------------------------------------- time

    def _update_time_grid(self) -> None:
        slices, traces = [], []
        for panel in self.panels:
            panel_slices, panel_traces = panel.time_arrays()
            slices += panel_slices
            traces += panel_traces
        if slices:
            grid = np.unique(np.concatenate(slices))
        elif traces:
            finite = np.concatenate([t[np.isfinite(t)] for t in traces])
            grid = (
                np.linspace(finite.min(), finite.max(), MAX_TRACE_STEPS)
                if finite.size else np.empty(0)
            )
        else:
            grid = np.empty(0)
        grid = grid[np.isfinite(grid)]
        self.timebar.set_grid(grid)
        if self.time is None and len(grid):
            self.set_time(float(grid[len(grid) // 2]))

    def set_time(self, t: Optional[float]) -> None:
        """Move the time cursor: profiles and maps show the slice closest to t."""
        self.time = t
        self.timebar.set_time(t)
        for panel in self.panels:
            panel.set_time(t)
        if self.mode == "point" and self.current_panel is not None and t is not None:
            values = self.current_panel.values_at_time(t)
            self.status_label.setText(" | ".join([f"t = {t:.5g} s"] + values))

    def _show_hover(self, text: str) -> None:
        self.coordinates_label.setText(text)

    # ----------------------------------------------------------------- modes

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.mode_actions[mode].setChecked(True)
        for panel in self.panels:
            panel.set_mode(mode)

    def autoscale_all(self) -> None:
        for panel in self.panels:
            panel.autoscale()

    def _set_link_time_axes(self, linked: bool) -> None:
        self.config.link_time_axes = linked
        self._link_time_axes()

    def _link_time_axes(self) -> None:
        """Link the x axis of all panels that have time on it."""
        timed = [p for p in self.panels if p.kind in ("trace", "spacetime")]
        master = timed[0].plot if timed else None
        for panel in self.panels:
            if panel in timed[1:] and self.config.link_time_axes:
                panel.plot.setXLink(master)
            else:
                panel.plot.setXLink(None)

    # ---------------------------------------------------------------- dialogs

    def open_setup(self, panel: Optional[Panel]) -> None:
        if panel is None:
            return
        self.set_current_panel(panel)
        dialog = SetupDialog(panel.config, [s.label for s in self.config.sources], self)

        def on_applied():
            panel.apply_config()
            self.refresh_panel(panel)

        dialog.applied.connect(on_applied)
        dialog.exec()

    def add_signal(self, path: str, new_panel: bool = False) -> None:
        """Add a signal (from the browser) to the current panel or a new one."""
        target = self.current_panel
        if new_panel or target is None:
            self.change_layout(target, "add_below")
            target = self.current_panel
        target.add_signal(path)

    # ------------------------------------------------------------------ files

    def _update_title(self) -> None:
        name = self.config.title or (self.config_path.name if self.config_path else "")
        self.setWindowTitle(f"idscope — {name}" if name else "idscope")

    def load_config_file(self, path) -> None:
        config = load_config(path)
        self.config = config
        self.config_path = Path(path)
        self.pool.close()
        self.pool = SourcePool(config.dd_version)
        self._logged_notes.clear()
        self.time = config.time
        self.current_panel = None
        self.link_action.setChecked(config.link_time_axes)
        self._show_sources()
        self.build_panels()
        self.refresh_all()
        self.browser.set_sources(self.sources())
        self._update_title()

    def new_config(self) -> None:
        sources = self.config.sources
        self.config = default_config()
        self.config.sources = sources
        self.config_path = None
        self.time = None
        self.current_panel = None
        self._show_sources()
        self.build_panels()
        self.refresh_all()
        self._update_title()

    def open_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open configuration", str(self._dialog_dir()),
            "idscope configuration (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        try:
            self.load_config_file(path)
        except IdscopeError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot open configuration", str(exc))

    def current_config(self) -> ScopeConfig:
        """The configuration with the current layout sizes and time."""
        self.sync_layout_to_config()
        self.config.time = self.time
        return self.config

    def save(self) -> None:
        if self.config_path is None:
            self.save_as()
        else:
            self._save_to(self.config_path)

    def save_as(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save configuration", str(self._dialog_dir() / "scope.yaml"),
            "idscope configuration (*.yaml *.yml)",
        )
        if path:
            if not path.endswith((".yaml", ".yml")):
                path += ".yaml"
            self._save_to(Path(path))

    def _save_to(self, path: Path) -> None:
        try:
            save_config(self.current_config(), path)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot save", str(exc))
            return
        self.config_path = path
        self._show_sources()
        self._update_title()
        self.status_label.setText(f"Saved {path}")

    def export_image(self, path) -> None:
        """Save the panels and the time bar as an image (PNG, JPG, ...)."""
        # Axes resize to fit their tick labels after a first paint: let that settle
        for _ in range(3):
            QtWidgets.QApplication.processEvents()
            self.centralWidget().grab()
        pixmap = self.centralWidget().grab()
        if not pixmap.save(str(path)):
            raise IdscopeError(f"Cannot write image {path}")

    def export_image_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export image", str(self._dialog_dir() / "idscope.png"),
            "Images (*.png *.jpg)",
        )
        if path:
            try:
                self.export_image(path)
            except IdscopeError as exc:
                QtWidgets.QMessageBox.warning(self, "Cannot export", str(exc))

    def _base_dir(self) -> Optional[Path]:
        """Relative source paths refer to the directory of the configuration."""
        return self.config_path.parent.resolve() if self.config_path else None

    def _show_sources(self) -> None:
        self.sources_edit.setText(format_source_list(self.config.sources, self._base_dir()))

    def _dialog_dir(self) -> Path:
        return self.config_path.parent if self.config_path else Path.cwd()

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About idscope",
            f"<b>idscope</b> {__version__}<br>"
            "A lightweight jScope-like viewer for IMAS IDS data, "
            "built on imas-python and pyqtgraph.",
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.pool.close()
        super().closeEvent(event)
