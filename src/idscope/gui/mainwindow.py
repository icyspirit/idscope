"""The idscope main window."""

import copy
import dataclasses
import logging
import os
import time as _time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from idscope import __version__
from idscope.config import (
    ColumnConfig,
    PanelConfig,
    ScopeConfig,
    config_to_dict,
    default_config,
    load_config,
    save_config,
)
from idscope.data.sources import (
    DataSource,
    SourcePool,
    format_source_list,
    parse_source,
    parse_source_list,
    unique_labels,
)
from idscope.errors import IdscopeError
from idscope.gui.browser import BrowserWidget
from idscope.gui.panel import COLORBAR_SPACE, Panel
from idscope.gui.setup_dialog import SetupDialog
from idscope.gui.timebar import TimeBar
from idscope.session import SETTINGS_FILE, config_dir, save_session
from idscope.templates import aliases_of, load_template, template_names

logger = logging.getLogger(__name__)

MAX_TRACE_STEPS = 2001
"""Number of slider steps when only traces (no profiles or maps) are shown."""
MODES = ("pan", "zoom", "point")
MODE_KEYS = {"pan": "P", "zoom": "Z", "point": "C"}
"""Keys of the mouse modes, on the plots."""
MAX_RECENT_FILES = 10
SESSION_DELAY_MS = 1000
"""The session is written this long after the last change, and when closing."""
WINDOW_STATE_VERSION = 1
"""Version of the saved layout of the docks and toolbar: change it when they change."""
MAX_UNDO = 100
"""Number of changes that can be undone."""


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


def _ensure_panels(config: ScopeConfig) -> ScopeConfig:
    """Give every column a panel, and the configuration a column."""
    if not config.columns:
        config.columns = [ColumnConfig([PanelConfig()])]
    for column in config.columns:
        if not column.panels:
            column.panels.append(PanelConfig())
    return config


def _layout(config: ScopeConfig) -> dict:
    """What a configuration file holds of the panels: all but the sources, the time
    and the Data Dictionary version, which go with the data at hand."""
    data = config_to_dict(config)
    for key in ("sources", "time", "dd_version"):
        data.pop(key, None)
    return data


@dataclasses.dataclass
class _Step:
    """A state of the scope that Undo (or Redo) goes back to."""

    config: ScopeConfig
    """A copy of the configuration."""
    config_path: Optional[Path]
    saved: Optional[dict]
    """The layout of the configuration file (see _layout), None if unknown."""
    description: str = ""
    """The change that left this state, e.g. "remove panel"."""
    replaced: bool = False
    """The change replaced the scope (Open, New, a template): Undo also goes back to
    the configuration file of this state."""
    current: Optional[Tuple[int, int]] = None
    """Column and row of the current panel."""


def _open_settings() -> QtCore.QSettings:
    """The settings of the window, next to the last session (see idscope.session)."""
    directory = config_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Cannot create %s: %s", directory, exc)
    return QtCore.QSettings(str(directory / SETTINGS_FILE), QtCore.QSettings.Format.IniFormat)


class MainWindow(QtWidgets.QMainWindow):
    """A scope window: sources, a grid of panels and a time bar.

    The title shows ``*`` when the panels differ from the configuration file; the
    sources and the time do not count. Edit → Undo takes back the changes of the
    panels and of the sources, and the scopes replaced by Open, New or a template.

    Args:
        modified: the panels may differ from ``config_path``, e.g. in the last
            session: compare them with that file.
        persist: restore the size, docks and mode of the window from the settings,
            and keep them, and the scope as the last session, when the window closes
            (see :mod:`idscope.session`). The session is also written a moment after
            each change, in case idscope does not end normally.
    """

    def __init__(self, config: Optional[ScopeConfig] = None, config_path=None,
                 modified: bool = False, persist: bool = False):
        super().__init__()
        self.config = _ensure_panels(config if config is not None else default_config())
        self.config_path: Optional[Path] = Path(config_path) if config_path else None
        # The layout of the configuration file, to compare the panels with
        self._saved: Optional[dict] = (
            self._file_layout(self.config_path) if modified else _layout(self.config)
        )
        self._undo: List[_Step] = []
        self._redo: List[_Step] = []
        self._merging: Optional[str] = None  # the change that the next one may extend
        self.persist = persist
        self.settings: Optional[QtCore.QSettings] = _open_settings() if persist else None
        self.pool = SourcePool(self.config.dd_version)
        self.panels: List[Panel] = []
        self.current_panel: Optional[Panel] = None
        self.member_colors: dict = {}
        """Colors of the members of signals with wildcards (see Panel)."""
        self.time: Optional[float] = self.config.time
        self.mode = self._setting("mode", "pan")
        if self.mode not in MODES:
            self.mode = "pan"
        self.recent_files: List[str] = self._read_recent_files()
        self.recent_sources: List[str] = self._read_list("recent_sources", "sources")
        last_dir = self._setting("last_dir", "")
        self._last_dir: Optional[Path] = Path(last_dir) if last_dir else None
        self._session_timer = QtCore.QTimer(self)
        self._session_timer.setSingleShot(True)
        self._session_timer.setInterval(SESSION_DELAY_MS)
        self._session_timer.timeout.connect(self._write_session)
        self.resize(1300, 850)
        # White around the plots as well (their headers, the time bar, the docks)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("white"))
        self.setPalette(palette)
        self.setAcceptDrops(True)  # files and entries: sources; configurations

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
        self._build_plot_keys()
        central_layout.addWidget(self.panel_area, stretch=1)
        central_layout.addWidget(self.timebar)
        self.setCentralWidget(central)
        self.columns_splitter: Optional[QtWidgets.QSplitter] = None
        self._sizes_dragged = False  # the splitters no longer show the configured sizes
        self._build_docks()
        self._build_menus()
        self.status_label = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status_label, stretch=1)
        self.coordinates_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.coordinates_label)
        self._logged_notes = set()
        self._shortcuts_dialog: Optional[QtWidgets.QDialog] = None

        self._show_sources()
        self.build_panels()
        self.refresh_all()
        self.browser.set_sources(self.sources())
        self._update_title()
        self._update_modified()
        self._stable = self._snapshot()  # the state that the next change starts from
        self._update_undo_actions()
        if self.config_path is not None:
            self._add_recent_file(self.config_path)
        if self.settings is not None:
            self._restore_window()

    # ------------------------------------------------------------------ build

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        self.mode_group = QtGui.QActionGroup(self)
        self.mode_actions = {}
        for mode, text, tip in (
            ("pan", "Pan", "Drag to move the view, right-drag to scale, wheel to zoom"),
            ("zoom", "Zoom", "Drag a rectangle to zoom, right-drag to scale, wheel to zoom"),
            ("point", "Point", "Click or drag in a time trace to set the time, "
                               "read values in the status bar"),
        ):
            action = QtGui.QAction(text, self)
            action.setCheckable(True)
            action.setToolTip(f"{tip} ({MODE_KEYS[mode]})")
            action.triggered.connect(lambda checked, m=mode: self.set_mode(m))
            self.mode_group.addAction(action)
            toolbar.addAction(action)
            self.mode_actions[mode] = action
        self.mode_actions[self.mode].setChecked(True)
        toolbar.addSeparator()
        autoscale = toolbar.addAction("Autoscale all")
        autoscale.setToolTip("Autoscale all panels (Shift+A; A autoscales the current "
                             "panel)")
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
        self.sources_edit.setAcceptDrops(False)  # files dropped on it are sources
        self.sources_edit.installEventFilter(self)  # Down: the recent sources
        toolbar.addWidget(self.sources_edit)
        self.recent_sources_menu = QtWidgets.QMenu(self)
        self.recent_sources_menu.setToolTipsVisible(True)
        self.recent_sources_button = QtWidgets.QToolButton()
        self.recent_sources_button.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.recent_sources_button.setToolTip("Recent sources (Down in the field)")
        self.recent_sources_button.setMenu(self.recent_sources_menu)
        self.recent_sources_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        toolbar.addWidget(self.recent_sources_button)
        self._update_recent_sources_menu()
        apply = toolbar.addAction("Apply")
        apply.setToolTip("Load the data of these sources (Enter)")
        apply.triggered.connect(self.apply_sources)
        reload_action = toolbar.addAction("Reload")
        reload_action.setToolTip("Read the data again, e.g. after it changed on disk (F5)")
        reload_action.setShortcut(QtGui.QKeySequence("F5"))
        reload_action.triggered.connect(self.reload)

    def _build_docks(self) -> None:
        self.browser = BrowserWidget()
        self.browser.preferred_ids = str(self._setting("browser/ids", ""))
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
        # The same keys on every platform (some have no Quit or Save as key)
        for text, shortcut, slot in (
            ("&New", "Ctrl+N", self.new_config),
            ("&Open…", "Ctrl+O", self.open_config),
            ("&Save", "Ctrl+S", self.save),
            ("Save &as…", "Ctrl+Shift+S", self.save_as),
            (None, None, None),
            ("&Export image…", "Ctrl+E", self.export_image_dialog),
            (None, None, None),
            ("&Quit", "Ctrl+Q", self.close),
        ):
            if text is None:
                file_menu.addSeparator()
                continue
            action = file_menu.addAction(text)
            action.setShortcut(QtGui.QKeySequence(shortcut))
            action.triggered.connect(slot)
            if slot == self.new_config:
                templates_menu = file_menu.addMenu("New from &template")
                for name in template_names():
                    aliases = aliases_of(name)
                    text = f"{name} ({', '.join(aliases)})" if aliases else name
                    templates_menu.addAction(text).triggered.connect(
                        lambda checked=False, n=name: self.new_from_template(n)
                    )
            elif slot == self.open_config:
                self.recent_menu = file_menu.addMenu("Open &recent")
                self._update_recent_menu()

        edit_menu = menu.addMenu("&Edit")
        self.undo_action = edit_menu.addAction("&Undo")
        self.undo_action.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
        self.undo_action.triggered.connect(self.undo)
        self.redo_action = edit_menu.addAction("&Redo")
        self.redo_action.setShortcuts([QtGui.QKeySequence("Ctrl+Shift+Z"),
                                       QtGui.QKeySequence("Ctrl+Y")])
        self.redo_action.triggered.connect(self.redo)
        edit_menu.addSeparator()
        self.delete_action = edit_menu.addAction("&Delete signals, then the empty panel")
        self.delete_action.setToolTip(
            "Delete the signals of the current panel; in a panel without signals, "
            "delete the panel"
        )
        # The keys go to text fields when they have the focus
        self.delete_action.setShortcuts(
            [QtGui.QKeySequence("Del"), QtGui.QKeySequence("Backspace")]
        )
        self.delete_action.triggered.connect(self.delete_in_current_panel)

        layout_menu = menu.addMenu("&Layout")
        for text, shortcut, action in (("Add panel below current", "Ctrl+Shift+Down",
                                        "add_below"),
                                       ("Add column", "Ctrl+Shift+Right", "add_column"),
                                       ("Remove current panel", None, "remove")):
            item = layout_menu.addAction(text)
            if shortcut:
                item.setShortcut(QtGui.QKeySequence(shortcut))
            item.triggered.connect(
                lambda checked=False, a=action: self.change_layout(self.current_panel, a)
            )
        layout_menu.addSeparator()
        # "\t": the key is shown, but only works on the plots (see _build_plot_keys)
        layout_menu.addAction("Setup current panel…\tEnter").triggered.connect(
            lambda: self.open_setup(self.current_panel)
        )

        view_menu = menu.addMenu("&View")
        browser_action = self.browser_dock.toggleViewAction()
        browser_action.setShortcut(QtGui.QKeySequence("Ctrl+B"))
        view_menu.addAction(browser_action)
        view_menu.addAction(self.log_dock.toggleViewAction())
        view_menu.addSeparator()
        self.share_x_action = view_menu.addAction("Share x axes")
        self.share_x_action.setToolTip(
            "Time axes are shared across the window, profile coordinates within a "
            "column; stacked panels then show x values only at the bottom"
        )
        self.share_x_action.setCheckable(True)
        self.share_x_action.setChecked(self.config.share_x)
        self.share_x_action.toggled.connect(self._set_share_x)
        self.share_x_action.triggered.connect(
            lambda checked=False: self._changed("shared x axes")
        )
        view_menu.addAction("Autoscale current panel\tA").triggered.connect(
            self.autoscale_current
        )
        view_menu.addAction("Autoscale all\tShift+A").triggered.connect(self.autoscale_all)
        view_menu.addSeparator()
        for text, shortcut, slot in (
            ("Previous time slice", "Ctrl+Left", lambda: self.timebar.step(-1)),
            ("Next time slice", "Ctrl+Right", lambda: self.timebar.step(1)),
            ("First time slice", "Ctrl+Home", lambda: self.timebar.go_to(0)),
            ("Last time slice", "Ctrl+End", lambda: self.timebar.go_to(-1)),
        ):
            action = view_menu.addAction(text)
            action.setShortcut(QtGui.QKeySequence(shortcut))
            action.triggered.connect(slot)
        view_menu.addSeparator()
        for text, shortcut, slot in (("Go to the sources", "Ctrl+L", self.focus_sources),
                                     ("Go to the IDS browser filter", "Ctrl+F",
                                      self.focus_filter),
                                     ("Go to the time", "Ctrl+T", self.focus_time),
                                     ("Back to the panels", "Esc", self.focus_panels)):
            action = view_menu.addAction(text)
            action.setShortcut(QtGui.QKeySequence(shortcut))
            action.triggered.connect(slot)
        view_menu.addSeparator()
        self.full_screen_action = view_menu.addAction("Full screen")
        self.full_screen_action.setCheckable(True)
        self.full_screen_action.setShortcut(QtGui.QKeySequence("F11"))
        self.full_screen_action.toggled.connect(self.set_full_screen)

        help_menu = menu.addMenu("&Help")
        shortcuts_action = help_menu.addAction("Keyboard shortcuts")
        shortcuts_action.setShortcut(QtGui.QKeySequence("F1"))
        shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction("About idscope").triggered.connect(self._about)

    def _build_plot_keys(self) -> None:
        """Single keys for the panels, which work once the keyboard is in them (after
        a click on a panel): text fields, lists and dialogs keep their keys."""
        def setup():
            self.open_setup(self.current_panel)

        # (what the keys do, [(key, slot), ...]); keys with the same slot are
        # alternatives, of which the first is shown (Enter, Return)
        self.plot_keys = [
            ("Select the panel above, below, on the left or on the right",
             [("Up", lambda: self.select_panel(0, -1)),
              ("Down", lambda: self.select_panel(0, 1)),
              ("Left", lambda: self.select_panel(-1, 0)),
              ("Right", lambda: self.select_panel(1, 0))]),
            ("Previous, next time slice", [(",", lambda: self.timebar.step(-1)),
                                           (".", lambda: self.timebar.step(1))]),
            ("Pan, Zoom, Point mode", [(MODE_KEYS[mode], lambda m=mode: self.set_mode(m))
                                       for mode in MODES]),
            ("Autoscale the current panel", [("A", self.autoscale_current)]),
            ("Autoscale all panels", [("Shift+A", self.autoscale_all)]),
            ("Setup of the current panel", [("Enter", setup), ("Return", setup)]),
            ("Previous, next image of the current panel (Show)",
             [("[", lambda: self.step_image(-1)), ("]", lambda: self.step_image(1))]),
        ]
        for _, keys in self.plot_keys:
            for key, slot in keys:
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.panel_area)
                shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(slot)

    # ----------------------------------------------------------------- panels

    def build_panels(self) -> None:
        """Create the panel widgets from ``config.columns``."""
        if self.columns_splitter is not None:
            self.columns_splitter.setParent(None)
            self.columns_splitter.deleteLater()
        self.panels = []
        self.column_panels: List[List[Panel]] = []
        _ensure_panels(self.config)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)
        # Handles (of the columns too) without the grip the style would draw: only
        # white between the panels
        splitter.setStyleSheet("QSplitter::handle { background: transparent; }")
        # emitted when the user drags a handle, not when the window is resized
        splitter.splitterMoved.connect(self._on_splitter_moved)
        for column in self.config.columns:
            column_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            column_splitter.setChildrenCollapsible(False)
            column_splitter.setHandleWidth(2)
            column_splitter.splitterMoved.connect(self._on_splitter_moved)
            panels = []
            for panel_config in column.panels:
                panel = self._create_panel(panel_config)
                column_splitter.addWidget(panel)
                panels.append(panel)
            self.panels += panels
            self.column_panels.append(panels)
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
        self._sizes_dragged = False
        self.panel_area_layout.addWidget(splitter)
        if self.current_panel not in self.panels:
            self.current_panel = None
        self.set_current_panel(self.current_panel or self.panels[0])
        for panel in self.panels:
            panel.set_mode(self.mode)

    def _create_panel(self, config: PanelConfig) -> Panel:
        panel = Panel(config, member_colors=self.member_colors)
        panel.activated.connect(self.set_current_panel)
        panel.timeRequested.connect(self.set_time)
        panel.setupRequested.connect(self.open_setup)
        panel.layoutRequested.connect(self.change_layout)
        panel.signalsChanged.connect(self._on_signals_changed)
        panel.hoverMessage.connect(self._show_hover)
        return panel

    def _on_signals_changed(self, panel: Panel, description: str) -> None:
        self._changed(description)
        self.refresh_panel(panel)

    def set_current_panel(self, panel: Optional[Panel]) -> None:
        if self.current_panel is not None and self.current_panel in self.panels:
            self.current_panel.set_current(False)
        self.current_panel = panel
        if panel is not None:
            panel.set_current(True)

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        self._sizes_dragged = True
        self._changed("panel sizes", merge=True)  # a drag moves in many steps

    def sync_layout_to_config(self) -> None:
        """Store splitter sizes as relative widths/heights in the configuration.

        Only sizes that the user dragged are stored: otherwise the splitters show the
        configured ones, up to rounding to pixels.
        """
        if self.columns_splitter is None or not self._sizes_dragged:
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

    def _location(self) -> Optional[Tuple[int, int]]:
        """Column and row of the current panel."""
        return self._locate(self.current_panel) if self.current_panel else None

    def _panel_at(self, location: Optional[Tuple[int, int]]) -> Optional[Panel]:
        if location is None:
            return None
        c, p = location
        if c < len(self.column_panels) and p < len(self.column_panels[c]):
            return self.column_panels[c][p]
        return None

    def _panels_have_focus(self) -> bool:
        widget = QtWidgets.QApplication.focusWidget()
        return widget is not None and self.panel_area.isAncestorOf(widget)

    def _focus_current_panel(self) -> None:
        """Keep the keyboard in the panels, e.g. for the next Delete, rather than in
        the text field that Qt gives it to when the panel that had it is removed."""
        if self.current_panel is not None:
            self.current_panel.graphics.setFocus()

    def change_layout(self, panel: Optional[Panel], action: str) -> None:
        """Add a panel below ``panel``, add a column after it, or remove it."""
        if panel is None:
            panel = self.panels[0] if self.panels else None
        self.sync_layout_to_config()
        before = self._location()
        focus = self._panels_have_focus()
        c, p = self._locate(panel) if panel is not None else (0, 0)
        columns = self.config.columns
        if action == "add_below":
            current = PanelConfig()
            columns[c].panels.insert(p + 1, current)
        elif action == "add_column":
            current = PanelConfig()
            columns.insert(c + 1, ColumnConfig([current]))
        elif action == "remove":
            del columns[c].panels[p]
            if not columns[c].panels:
                del columns[c]
            if not columns:
                columns.append(ColumnConfig([PanelConfig()]))
            # the panel that takes its place becomes the current one
            column = columns[min(c, len(columns) - 1)]
            current = column.panels[min(p, len(column.panels) - 1)]
        else:
            raise ValueError(f"Unknown layout change '{action}'")
        self.build_panels()
        self.set_current_panel(next(x for x in self.panels if x.config is current))
        if focus:
            self._focus_current_panel()
        self.refresh_all()
        description = {"add_below": "add panel", "add_column": "add column",
                       "remove": "remove panel"}[action]
        self._changed(description, current=before)

    def delete_in_current_panel(self) -> None:
        """Delete the signals of the current panel, or the panel if it has none
        (Delete, Backspace)."""
        panel = self.current_panel
        if panel is None:
            return
        if panel.config.signals:
            panel.clear_signals()
        elif len(self.panels) > 1:
            self.change_layout(panel, "remove")

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
        self._set_sources(specs)

    def add_sources(self, texts: List[str]) -> None:
        """Add sources (e.g. files dropped on the window) to those shown."""
        try:
            new = [parse_source(text) for text in texts]
        except IdscopeError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid source", str(exc))
            return
        known = {spec.uri for spec in self.config.sources}
        new = [spec for spec in new if spec.uri not in known]
        if new:
            self._set_sources(unique_labels(self.config.sources + new))

    def _set_sources(self, specs) -> None:
        self.config.sources = specs
        self.pool.keep_only(specs)
        self._show_sources()
        self.refresh_all()
        self.browser.set_sources(self.sources())
        self._record("sources")  # which can be undone, but is no change of the panels
        self._session_changed()

    def reload(self) -> None:
        """Read all data again."""
        self.pool.reload()
        self._logged_notes.clear()
        self.refresh_all()
        self.browser.set_sources(self.sources())

    def refresh_all(self) -> None:
        start = _time.perf_counter()
        errors, notes = [], []
        with _busy():
            sources = self.sources()
            for panel in self.panels:
                panel_errors, panel_notes = self._load_panel(panel, sources)
                errors += panel_errors
                notes += panel_notes
            self._update_time_grid()
            self._arrange_axes()
        self._report(errors, notes, _time.perf_counter() - start)

    def refresh_panel(self, panel: Panel) -> None:
        start = _time.perf_counter()
        with _busy():
            errors, notes = self._load_panel(panel, self.sources())
            self._update_time_grid()
            self._arrange_axes()
        self._report(errors, notes, _time.perf_counter() - start)

    def _load_panel(self, panel: Panel,
                    sources: List[DataSource]) -> Tuple[List[str], List[str]]:
        """Load a panel; returns its errors and what is not filled, for the log."""
        errors = panel.load(sources, self.time)
        name = panel.title()
        return ([f"[{name}] {error}" for error in errors],
                [f"[{name}] {note}" for note in panel.notes])

    def _report(self, errors: List[str], notes: List[str], seconds: float) -> None:
        for source in self.config.sources:
            for note in self.pool.get(source).notes:
                if note not in self._logged_notes:
                    self._logged_notes.add(note)
                    self.log(note)
        for message in errors + notes:
            self.log(message)
        n_curves = sum(len(panel.curves) for panel in self.panels)
        text = f"{n_curves} signals loaded in {seconds:.2f} s"
        counts = [f"{len(errors)} not shown"] if errors else []
        counts += [f"{len(notes)} not filled"] if notes else []
        if counts:
            text += f"; {', '.join(counts)} (see View → Log)"
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
        times = np.concatenate(slices + traces) if slices or traces else np.empty(0)
        times = np.unique(times[np.isfinite(times)])
        if len(times) == 1 and self.time != float(times[0]):
            # All the data is at one time (e.g. what a code gives for the time it
            # was run for): that is the time
            self.set_time(float(times[0]))
        elif self.time is None and len(grid):
            self.set_time(float(grid[len(grid) // 2]))
        else:  # e.g. the time of a configuration, which the panels were loaded at
            self.timebar.set_time(self.time)

    def set_time(self, t: Optional[float]) -> None:
        """Move the time cursor: profiles and maps show the slice closest to t."""
        self.time = t
        self.timebar.set_time(t)
        for panel in self.panels:
            panel.set_time(t)
        if self.mode == "point" and self.current_panel is not None and t is not None:
            values = self.current_panel.values_at_time(t)
            self.status_label.setText(" | ".join([f"t = {t:.5g} s"] + values))
        self._session_changed()

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

    def autoscale_current(self) -> None:
        if self.current_panel is not None:
            self.current_panel.autoscale()

    def select_panel(self, columns: int, rows: int) -> None:
        """Make the panel ``rows`` below (or ``columns`` to the right of) the current
        one the current panel, with the keyboard: in another column, the panel
        closest in height."""
        if self.current_panel is None:
            return
        c, p = self._locate(self.current_panel)
        if columns:
            target = min(max(c + columns, 0), len(self.column_panels) - 1)
            if target != c:
                middle = self._middle(self.current_panel)
                p = min(range(len(self.column_panels[target])),
                        key=lambda i: abs(self._middle(self.column_panels[target][i])
                                          - middle))
            c = target
        panels = self.column_panels[c]
        self.set_current_panel(panels[min(max(p + rows, 0), len(panels) - 1)])
        self._focus_current_panel()

    def _middle(self, panel: Panel) -> int:
        """Height of the middle of a panel, in the panel area."""
        return panel.mapTo(self.panel_area, QtCore.QPoint(0, panel.height() // 2)).y()

    def step_image(self, delta: int) -> None:
        if self.current_panel is not None:
            self.current_panel.step_image(delta)

    # ------------------------------------------------------------- keyboard

    def focus_sources(self) -> None:
        self.sources_edit.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        self.sources_edit.selectAll()

    def focus_filter(self) -> None:
        self.browser_dock.show()
        self.browser.filter.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        self.browser.filter.selectAll()

    def focus_time(self) -> None:
        self.timebar.spin.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        self.timebar.spin.selectAll()

    def focus_panels(self) -> None:
        """Give the keyboard back to the current panel (e.g. from a text field)."""
        self._focus_current_panel()

    def set_full_screen(self, full: bool) -> None:
        state = self.windowState() & ~QtCore.Qt.WindowState.WindowFullScreen
        if full:
            state |= QtCore.Qt.WindowState.WindowFullScreen
        self.setWindowState(state)

    def shortcut_rows(self) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """The keyboard shortcuts, as ``(group, [(keys, what they do), ...])``."""
        native = QtGui.QKeySequence.SequenceFormat.NativeText
        actions = [action for menu_action in self.menuBar().actions()
                   for action in menu_action.menu().actions()]
        actions += self.findChild(QtWidgets.QToolBar, "main_toolbar").actions()
        anywhere, seen = [], set()
        for action in actions:
            keys = [k.toString(native) for k in action.shortcuts()]
            text = action.text().split("\t")[0].replace("&", "").rstrip("…")
            if keys and text not in seen:
                seen.add(text)
                anywhere.append((", ".join(keys), text))
        on_panels = []
        for text, keys in self.plot_keys:
            shown, slots = [], []
            for key, slot in keys:
                if slot not in slots:
                    shown.append(QtGui.QKeySequence(key).toString(native))
                    slots.append(slot)
            on_panels.append(("  ".join(shown), text))
        return [("On the panels, after a click on one", on_panels),
                ("Anywhere (text fields keep the keys they use)", anywhere)]

    def show_shortcuts(self) -> None:
        """List the keyboard shortcuts (F1), in a window that can stay open."""
        if self._shortcuts_dialog is not None:
            self._shortcuts_dialog.show()
            self._shortcuts_dialog.raise_()
            self._shortcuts_dialog.activateWindow()
            return
        tables = []
        for group, keys in self.shortcut_rows():
            rows = "".join(f"<tr><td style='padding-right: 14px'><b>"
                           f"{key.replace('  ', '&nbsp;&nbsp; ')}</b></td>"
                           f"<td>{text}</td></tr>" for key, text in keys)
            tables.append(f"<p><b>{group}</b></p><table>{rows}</table>")
        dialog = QtWidgets.QDialog(self)  # not modal: keys can be tried meanwhile
        dialog.setWindowTitle("Keyboard shortcuts")
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel("<table><tr><td valign='top' style='padding-right: "
                                 f"28px'>{tables[0]}</td><td valign='top'>{tables[1]}"
                                 "</td></tr></table>")
        label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.rejected.connect(dialog.close)
        layout.addWidget(buttons)
        dialog.show()
        self._shortcuts_dialog = dialog

    def _set_share_x(self, shared: bool) -> None:
        self.config.share_x = shared
        self._arrange_axes()

    def _arrange_axes(self) -> None:
        """Share x axes and stack panels tightly, as in jScope.

        Panels that show the same x quantity share their x axis: time across the
        window, a profile coordinate within a column. In a column, a panel whose
        neighbour below shares its x axis does not repeat the x tick labels. Panels
        without a color bar keep room for one when others in the column have it, so
        that the plots line up.
        """
        share = self.config.share_x
        masters = {}
        for column, panels in enumerate(self.column_panels):
            for panel in panels:
                key = panel.x_key(column) if share else None
                master = masters.setdefault(key, panel) if key else panel
                panel.plot.setXLink(None if master is panel else master.plot)
            keys = [panel.x_key(column) if share else None for panel in panels]
            for i, panel in enumerate(panels):
                below = keys[i + 1] if i + 1 < len(panels) else None
                panel.set_x_tick_labels(keys[i] is None or keys[i] != below)
            padding = COLORBAR_SPACE if any(p.has_colorbar() for p in panels) else 0
            for panel in panels:
                panel.set_right_padding(padding)

    # ---------------------------------------------------------------- dialogs

    def open_setup(self, panel: Optional[Panel]) -> None:
        if panel is None:
            return
        self.set_current_panel(panel)
        dialog = SetupDialog(panel.config, [s.label for s in self.config.sources], self)
        before = copy.deepcopy(panel.config)

        def on_applied():
            nonlocal before
            if panel.config != before:  # e.g. not when OK follows Apply
                self._changed("panel setup")
                before = copy.deepcopy(panel.config)
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
        # Qt writes "*" in place of "[*]" while the window is modified
        self.setWindowTitle(f"idscope — {name}[*]" if name else "idscope[*]")

    def _update_modified(self) -> None:
        """Show * in the title when the panels differ from the configuration file."""
        self.setWindowModified(
            self._saved is None or _layout(self.current_config()) != self._saved
        )

    def _file_layout(self, path: Optional[Path]) -> Optional[dict]:
        if path is None:
            return None
        try:
            return _layout(_ensure_panels(load_config(path)))
        except IdscopeError:
            return None

    def _changed(self, description: str, *, current: Optional[Tuple[int, int]] = None,
                 merge: bool = False) -> None:
        """The user changed the panels (see _record)."""
        self._record(description, current=current, merge=merge)
        self._update_modified()
        self._session_changed()

    def show_config(self, config: ScopeConfig, config_path=None,
                    description: str = "open") -> None:
        """Show another scope, e.g. from a configuration file or a template.

        Undo goes back to the scope shown before, with its configuration file.
        """
        before = self._location()
        config = _ensure_panels(config)
        self._show(config, config_path, _layout(config))
        self._record(description, replaced=True, current=before)
        self._session_changed()

    def _show(self, config: ScopeConfig, config_path, saved: Optional[dict],
              current: Optional[Tuple[int, int]] = None) -> None:
        focus = self._panels_have_focus()
        sources_changed = config.sources != self.config.sources
        if config.dd_version != self.pool.dd_version:
            self.pool.close()
            self.pool = SourcePool(config.dd_version)
            self._logged_notes.clear()
            sources_changed = True
        else:
            self.pool.keep_only(config.sources)
        self.config = config
        self.config_path = Path(config_path) if config_path else None
        self._saved = saved
        self.time = config.time
        self.current_panel = None
        with QtCore.QSignalBlocker(self.share_x_action):
            self.share_x_action.setChecked(config.share_x)
        self._show_sources()
        self.member_colors.clear()  # the members of this scope, from the first color
        self.build_panels()
        panel = self._panel_at(current)
        if panel is not None:
            self.set_current_panel(panel)
        if focus:
            self._focus_current_panel()
        self.refresh_all()
        if sources_changed:
            self.browser.set_sources(self.sources())
        self._update_title()
        self._update_modified()

    # ------------------------------------------------------------------- undo

    def _snapshot(self, **fields) -> _Step:
        return _Step(copy.deepcopy(self.current_config()), self.config_path, self._saved,
                     **fields)

    def _record(self, description: str, *, replaced: bool = False,
                current: Optional[Tuple[int, int]] = None, merge: bool = False) -> None:
        """Keep the state before a change, so that Undo can go back to it.

        Args:
            description: what changed, e.g. "remove panel" (shown as "Undo remove
                panel").
            replaced: the change replaced the scope (Open, New, a template).
            current: the current panel before the change, if the change moved it.
            merge: add the change to the last one if that was the same, e.g. to count
                the steps of dragging a splitter as one change.
        """
        if not (merge and self._merging == description and self._undo):
            self._undo.append(dataclasses.replace(
                self._stable, description=description, replaced=replaced,
                current=current if current is not None else self._location(),
            ))
            del self._undo[:-MAX_UNDO]
        self._redo.clear()
        self._merging = description if merge else None
        self._stable = self._snapshot()
        self._update_undo_actions()

    def undo(self) -> None:
        """Go back to the state before the last change (Ctrl+Z)."""
        if self._undo:
            self._go_to(self._undo.pop(), self._redo)

    def redo(self) -> None:
        """Make the last undone change again."""
        if self._redo:
            self._go_to(self._redo.pop(), self._undo)

    def _go_to(self, step: _Step, other: List[_Step]) -> None:
        """Show ``step``, keeping the current state in ``other``, which undoes that."""
        other.append(self._snapshot(description=step.description,
                                    replaced=step.replaced, current=self._location()))
        config = copy.deepcopy(step.config)
        config.time = self.time  # the time is not part of the changes
        if step.replaced:  # back to that scope, with its configuration file
            path, saved = step.config_path, step.saved
        else:  # changes of the scope do not change the file it belongs to
            path, saved = self.config_path, self._saved
        self._show(config, path, saved, step.current)
        self._merging = None
        self._stable = self._snapshot()
        self._update_undo_actions()
        self._session_changed()

    def _update_undo_actions(self) -> None:
        for action, steps, verb in ((self.undo_action, self._undo, "Undo"),
                                    (self.redo_action, self._redo, "Redo")):
            action.setEnabled(bool(steps))
            action.setText(f"&{verb} {steps[-1].description}" if steps else f"&{verb}")

    # ------------------------------------------------------------------ files

    def load_config_file(self, path) -> None:
        self.show_config(load_config(path), path, f"open {Path(path).name}")
        self._add_recent_file(path)

    def new_from_template(self, name: str) -> None:
        """Replace the layout by a template, keeping the sources."""
        try:
            config = load_template(name)
        except IdscopeError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot load template", str(exc))
            return
        config.sources = self.config.sources
        config.dd_version = self.config.dd_version
        self.show_config(config, description=f"template {name}")

    def new_config(self) -> None:
        """Replace the layout by an empty panel, keeping the sources."""
        config = default_config()
        config.sources = self.config.sources
        config.dd_version = self.config.dd_version
        self.show_config(config, description="new")

    def open_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open configuration", str(self._dialog_dir()),
            "idscope configuration (*.yaml *.yml);;All files (*)",
        )
        if path:
            self._remember_dir(path)
            self.open_file(path)

    def open_file(self, path) -> None:
        """Open a configuration file, or tell why it cannot be opened."""
        try:
            self.load_config_file(path)
        except IdscopeError as exc:
            if not Path(path).is_file():
                self._remove_recent_file(path)
            QtWidgets.QMessageBox.warning(self, "Cannot open configuration", str(exc))

    def current_config(self) -> ScopeConfig:
        """The configuration with the current layout sizes and time."""
        self.sync_layout_to_config()
        self.config.time = None if self.time is None else float(self.time)
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
            self._remember_dir(path)
            self._save_to(Path(path))

    def _save_to(self, path: Path) -> None:
        try:
            save_config(self.current_config(), path)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot save", str(exc))
            return
        self.config_path = path
        self._saved = _layout(self.config)
        self._stable = self._snapshot()  # a later Open or New goes back to this file
        self._merging = None  # a drag after the save is a change of its own
        self._show_sources()
        self._update_title()
        self._update_modified()
        self._add_recent_file(path)
        self._session_changed()
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
            self._remember_dir(path)
            try:
                self.export_image(path)
            except IdscopeError as exc:
                QtWidgets.QMessageBox.warning(self, "Cannot export", str(exc))

    def _base_dir(self) -> Optional[Path]:
        """Relative source paths refer to the directory of the configuration."""
        return self.config_path.parent.resolve() if self.config_path else None

    def _show_sources(self) -> None:
        self.sources_edit.setText(format_source_list(self.config.sources, self._base_dir()))
        if self.config.sources:  # as they are, wherever the configuration is
            text = format_source_list(self.config.sources)
            self.recent_sources = ([text] + [t for t in self._read_list(
                "recent_sources", "sources") if t != text])[:MAX_RECENT_FILES]
            self._write_list("recent_sources", "sources", self.recent_sources)
            self._update_recent_sources_menu()

    def _update_recent_sources_menu(self) -> None:
        menu = self.recent_sources_menu
        menu.clear()
        for text in self.recent_sources:
            shown = text if len(text) <= 100 else f"{text[:48]}…{text[-48:]}"
            action = menu.addAction(shown.replace("&", "&&"))
            action.setToolTip(text)
            action.triggered.connect(lambda checked=False, t=text: self.use_sources(t))
        if self.recent_sources:
            menu.addSeparator()
            menu.addAction("Clear list").triggered.connect(self._clear_recent_sources)
        self.recent_sources_button.setEnabled(bool(self.recent_sources))

    def _clear_recent_sources(self) -> None:
        self.recent_sources = []
        self._write_list("recent_sources", "sources", [])
        self._update_recent_sources_menu()

    def use_sources(self, text: str) -> None:
        """Show these sources (e.g. recent ones), as if typed and applied."""
        self.sources_edit.setText(text)
        self.apply_sources()

    def eventFilter(self, watched, event) -> bool:
        if (watched is self.sources_edit and event.type() == QtCore.QEvent.Type.KeyPress
                and event.key() == QtCore.Qt.Key.Key_Down and self.recent_sources):
            self.recent_sources_menu.popup(self.sources_edit.mapToGlobal(
                QtCore.QPoint(0, self.sources_edit.height())
            ))
            return True
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------ dropped files

    @staticmethod
    def _dropped_paths(mime: QtCore.QMimeData) -> List[Path]:
        if not mime.hasUrls():
            return []
        return [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if self._dropped_paths(event.mimeData()):
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if self._dropped_paths(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Files and HDF5 entry directories dropped on the window are added as
        sources; a configuration (.yaml) is opened."""
        paths = self._dropped_paths(event.mimeData())
        if not paths:
            return
        event.acceptProposedAction()
        configs = [p for p in paths if p.suffix.lower() in (".yaml", ".yml")]
        if configs:
            self.open_file(str(configs[0]))
        others = [str(p) for p in paths if p not in configs]
        if others:
            self.add_sources(others)

    def _dialog_dir(self) -> Path:
        """Where file dialogs start: the directory of the configuration file, else the
        last one chosen in a dialog, else the current directory."""
        if self.config_path is not None:
            return self.config_path.parent
        if self._last_dir is not None and self._last_dir.is_dir():
            return self._last_dir
        return Path.cwd()

    def _remember_dir(self, path) -> None:
        self._last_dir = Path(os.path.abspath(path)).parent
        if self.settings is not None:
            self.settings.setValue("last_dir", str(self._last_dir))

    # ----------------------------------------------------------- recent files

    def _read_list(self, name: str, key: str) -> List[str]:
        """A list kept in the settings (recent files, recent sources), newest first;
        without settings, the list of this window."""
        if self.settings is None:
            return list(getattr(self, name, []))
        self.settings.sync()  # another idscope may have changed the list
        values: List[str] = []
        for i in range(self.settings.beginReadArray(name)):
            self.settings.setArrayIndex(i)
            value = self.settings.value(key)
            if value and str(value) not in values:
                values.append(str(value))
        self.settings.endArray()
        return values[:MAX_RECENT_FILES]

    def _write_list(self, name: str, key: str, values: List[str]) -> None:
        if self.settings is None:
            return
        self.settings.remove(name)
        self.settings.beginWriteArray(name, min(len(values), MAX_RECENT_FILES))
        for i, value in enumerate(values[:MAX_RECENT_FILES]):
            self.settings.setArrayIndex(i)
            self.settings.setValue(key, value)
        self.settings.endArray()
        self.settings.sync()

    def _read_recent_files(self) -> List[str]:
        return self._read_list("recent_files", "path")

    def _set_recent_files(self, files: List[str]) -> None:
        self.recent_files = files[:MAX_RECENT_FILES]
        self._write_list("recent_files", "path", self.recent_files)
        self._update_recent_menu()

    def _add_recent_file(self, path) -> None:
        path = os.path.abspath(path)
        self._set_recent_files([path] + [p for p in self._read_recent_files() if p != path])

    def _remove_recent_file(self, path) -> None:
        path = os.path.abspath(path)
        self._set_recent_files([p for p in self._read_recent_files() if p != path])

    def _update_recent_menu(self) -> None:
        menu = self.recent_menu
        menu.clear()
        home = str(Path.home())
        for number, path in enumerate(self.recent_files, 1):
            text = "~" + path[len(home):] if path.startswith(home + os.sep) else path
            text = text.replace("&", "&&")  # not a keyboard shortcut
            action = menu.addAction(f"&{number} {text}" if number < 10 else text)
            action.triggered.connect(lambda checked=False, p=path: self.open_file(p))
        if self.recent_files:
            menu.addSeparator()
            menu.addAction("Clear list").triggered.connect(
                lambda checked=False: self._set_recent_files([])
            )
        menu.setEnabled(bool(self.recent_files))

    # --------------------------------------------------- session and settings

    def _setting(self, key: str, default=None):
        value = self.settings.value(key) if self.settings is not None else None
        return default if value is None else value

    def _restore_window(self) -> None:
        geometry = self._setting("window/geometry")
        if isinstance(geometry, QtCore.QByteArray):
            self.restoreGeometry(geometry)
        state = self._setting("window/state")
        if isinstance(state, QtCore.QByteArray):
            self.restoreState(state, WINDOW_STATE_VERSION)

    def _save_settings(self) -> None:
        settings = self.settings
        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/state", self.saveState(WINDOW_STATE_VERSION))
        settings.setValue("mode", self.mode)
        ids_name = self.browser.ids_combo.currentText()
        if ids_name:
            settings.setValue("browser/ids", ids_name)
        settings.sync()

    def _session_changed(self) -> None:
        """Write the session a moment after the last change."""
        if self.persist:
            self._session_timer.start()

    def _write_session(self) -> None:
        """Keep the scope as the last session (see idscope.session)."""
        self._session_timer.stop()
        try:
            save_session(self.current_config(), self.config_path, self.isWindowModified())
        except Exception as exc:  # never fail, e.g. to close, because of the session
            logger.warning("Cannot save the session: %s", exc)
            self.log(f"Cannot save the session: {exc}")

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About idscope",
            f"<b>idscope</b> {__version__}<br>"
            "A lightweight jScope-like viewer for IMAS IDS data, "
            "built on imas-python and pyqtgraph.",
        )

    def changeEvent(self, event: QtCore.QEvent) -> None:
        action = getattr(self, "full_screen_action", None)  # once the menus exist
        if event.type() == QtCore.QEvent.Type.WindowStateChange and action is not None:
            # e.g. full screen left through the window manager
            with QtCore.QSignalBlocker(action):
                action.setChecked(
                    bool(self.windowState() & QtCore.Qt.WindowState.WindowFullScreen)
                )
        super().changeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.persist:
            self._write_session()
            self._save_settings()
        self.pool.close()
        super().closeEvent(event)
