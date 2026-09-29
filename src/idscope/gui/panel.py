"""A plot panel: one cell of the scope grid (a "wave" in jScope)."""

import html
import logging
import re
from typing import List, NamedTuple, Optional, Tuple

import contourpy
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from idscope.config import PanelConfig, SignalConfig
from idscope.data import MAP, PROFILE, SHAPE, TRACE, Channels, MapSeries, ProfileSeries
from idscope.data import Shapes, SignalData, parse_alternatives
from idscope.data.extract import ANNULAR, GEOMETRY, MARKERS, POINT, MeshSeries
from idscope.data.sources import DataSource
from idscope.errors import IdscopeError, NoDataError
from idscope.gui import style
from idscope.gui.mesh import MeshItem

logger = logging.getLogger(__name__)

MIME_SIGNAL = "application/x-idscope-signal"
"""Mime type of signal paths dragged from the IDS browser (one path per line)."""

LEFT_AXIS_WIDTH = 52
"""Width of the y tick labels (the axis label comes on top). The same for all panels
so that the plots of a column line up."""
COLORBAR_WIDTH = 72
"""Fixed width of color bars."""
COLORBAR_GAP = 4
"""Gap between a plot and its color bar."""
COLORBAR_SPACE = COLORBAR_WIDTH + COLORBAR_GAP
"""Room taken by a color bar; panels without one reserve it to stay aligned."""
EDGE_PADDING = 14
"""Space between the data and the ends of the x axis, in pixels."""

_DATA_KIND_TO_PANEL = {TRACE: "trace", PROFILE: "profile", MAP: "map", SHAPE: "map"}
_MAX_SHAPE_LABELS = 40
"""Shapes are named on the map (e.g. coils) when there are no more of them."""
_MAX_CATEGORY_LABELS = 24
_MIN_BAR_SLOTS = 4
"""Bars take the room of at least this many elements (else a single bar would
fill the panel)."""
_MAX_VALUE_LABELS = 16
"""Bars of data at one time are labeled with their values when there are no more."""
_TOP_ROOM = 44
"""Pixels above the highest labeled bar: for its value, and the title above it."""
_BOTTOM_ROOM = 20
"""Pixels below the lowest (negative) labeled bar, for its value."""
_MAX_CURSOR_ROWS = 10
"""Values that the crosshair of the Point mode lists, at most."""
_HINT = "Double-click to set up, or drop a signal from the IDS browser"
_FONT_SIZE = 8  # points


class _Axes(NamedTuple):
    """Labels, ranges and log scales of the axes (None: automatic)."""

    x_label: Optional[str]
    y_label: Optional[str]
    x_range: Optional[Tuple[float, float]]
    y_range: Optional[Tuple[float, float]]
    log_x: bool
    log_y: bool


class Curve(NamedTuple):
    """One signal of one source (or one member of a signal with wildcards), as
    plotted in a panel."""

    signal: SignalConfig
    signal_index: int
    source: str
    source_index: int
    data: SignalData
    name: str
    item: Optional[object]  # PlotDataItem / InfiniteLine / BarGraphItem, None for images
    label: Optional[str] = None
    """Text of its legend entry (None if it has none)."""

    def scaled(self, values: np.ndarray) -> np.ndarray:
        return values * self.signal.scale + self.signal.offset


def _is_uniform(x: np.ndarray) -> bool:
    if len(x) < 2:
        return True
    step = np.diff(x)
    return bool(np.all(step > 0) and np.allclose(step, step[0], rtol=1e-3, atol=0))


def _at_one_time(loaded) -> bool:
    """Whether the data of a panel that depends on time is all at one and the same
    time: every trace or profile a single sample or slice, at the same time. (A
    time graph of runs of a code at different times, one time each, says something.)
    """
    times = []
    for entry in loaded:
        data = entry[4]
        for member in data.members if isinstance(data, Channels) else [data]:
            if member.kind in (TRACE, PROFILE) and member.time is not None:
                times.append(member.time)
    if not times:
        return False
    times = np.concatenate(times)
    times = times[np.isfinite(times)]
    return times.size > 0 and bool(np.all(times == times[0]))


def _margin(room: float, padding: float = 0.1) -> float:
    """The margin beyond the data, as a fraction of their range, that leaves
    ``room`` (a fraction of the height of the plot) once the plot has added its
    own ``padding`` on both sides."""
    need = room * (1 + 2 * padding) - padding
    if need <= 0:
        return 0.0
    return min(need / max(1 - need, 0.25), 3.0)


def _edges(x: np.ndarray) -> np.ndarray:
    """Cell edges around the points ``x`` (for PColorMeshItem)."""
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    mid = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])


def contour_lines(x1: np.ndarray, x2: np.ndarray, z: np.ndarray, count: int):
    """``count`` evenly spaced contour levels of z(x1, x2).

    Returns:
        ``[(level, x, y), ...]`` where x and y hold all lines of a level, separated
        by NaN (to be drawn as one PlotDataItem).
    """
    finite = z[np.isfinite(z)]
    if count <= 0 or finite.size == 0 or finite.min() == finite.max():
        return []
    levels = np.linspace(finite.min(), finite.max(), count + 2)[1:-1]
    generator = contourpy.contour_generator(
        x1, x2, np.ma.masked_invalid(z.T), line_type="Separate"
    )
    result = []
    for level in levels:
        xs, ys = [], []
        for line in generator.lines(level):
            xs += [line[:, 0], [np.nan]]
            ys += [line[:, 1], [np.nan]]
        if xs:
            result.append((float(level), np.concatenate(xs), np.concatenate(ys)))
    return result


def _joined(outlines: List[np.ndarray], separate: bool) -> Tuple[np.ndarray, np.ndarray]:
    """x and y of several outlines, separated by NaN (drawn as one curve)."""
    parts = []
    for outline in outlines:
        if outline.size:
            parts.append(outline)
            if separate:
                parts.append(np.full((1, 2), np.nan))
    if not parts:
        return np.empty(0), np.empty(0)
    points = np.vstack(parts)
    if not separate:
        points = points[np.all(np.isfinite(points), axis=1)]
    return points[:, 0], points[:, 1]


def _mark_single_point(item, x, y) -> None:
    """Show a curve of a single point (e.g. one time sample) with a symbol, since
    a line through one point is invisible."""
    if item is None:
        return
    finite = np.isfinite(np.asarray(x, float)) & np.isfinite(np.asarray(y, float))
    single = int(finite.sum()) == 1
    if single != (item.opts.get("symbol") is not None):
        pen = item.opts["pen"]
        color = pg.mkPen(pen).color() if pen is not None else "k"
        item.setSymbol("o" if single else None)
        item.setSymbolSize(6)
        item.setSymbolBrush(color)
        item.setSymbolPen(color)


def _compact_exponent(text: str) -> str:
    """``2e+19`` as ``2e19`` and ``1e-05`` as ``1e-5``."""
    return re.sub(r"e\+?(-?)0*(\d)", r"e\1\2", text)


def _format(value: float) -> str:
    return f"{value:.5g}" if np.isfinite(value) else "nan"


def _round_to_view(value: float, span: float, log: bool = False) -> float:
    """A coordinate of the mouse in a view ``span`` wide, as the value it stands for
    (10**value on a log axis), to about a thousandth of the view: 2.2e-16 is 0."""
    if log:
        value = 10.0 ** value
        step = abs(value) * np.log(10.0) * span / 1000.0  # the same, in decades
    else:
        step = span / 1000.0
    if not (np.isfinite(value) and np.isfinite(step) and step > 0):
        return value
    step = 10.0 ** np.floor(np.log10(step))
    return float(np.round(value / step) * step) + 0.0  # not -0


def _value_at(xs, ys, x: float, log_x: bool, tolerance: float) -> Optional[float]:
    """The value of a curve y(xs) at ``x`` (in the coordinates of the view: log10 of
    x on a log axis), along its line; None away from it (more than ``tolerance``
    beyond its ends, or anywhere for a single point farther than that)."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if log_x:
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = np.log10(xs)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if xs.size == 0:
        return None
    if xs.size > 1 and np.any(np.diff(xs) < 0):
        order = np.argsort(xs, kind="stable")
        xs, ys = xs[order], ys[order]
    if not xs[0] - tolerance <= x <= xs[-1] + tolerance:
        return None
    return float(np.interp(x, xs, ys))


class Panel(QtWidgets.QFrame):
    """A panel showing traces, profiles, a map or a time-space image.

    Signals of the panel are defined by :attr:`config`; :meth:`load` reads them from
    the data sources and :meth:`set_time` moves the panel to another time.
    """

    activated = QtCore.Signal(object)
    """The user clicked in the panel (it becomes the current panel)."""
    timeRequested = QtCore.Signal(float)
    """The user picked a time (time line dragged, or click in Point mode)."""
    setupRequested = QtCore.Signal(object)
    layoutRequested = QtCore.Signal(object, str)
    """Layout change: ``add_below``, ``add_column`` or ``remove``."""
    signalsChanged = QtCore.Signal(object, str)
    """The signals of the panel changed (e.g. by a drop) and must be reloaded:
    ``(panel, description of the change)``."""
    hoverMessage = QtCore.Signal(str)

    def __init__(self, config: PanelConfig, parent=None, member_colors=None):
        """
        Args:
            member_colors: colors of the members of signals with wildcards, shared
                by the panels of a window so that e.g. a coil has the same color in
                all of them: ``(what, name, signal) -> color index``.
        """
        super().__init__(parent)
        self.config = config
        self.kind: Optional[str] = None
        """Resolved kind: trace, profile, map or spacetime (None if nothing loaded)."""
        self.curves: List[Curve] = []
        self.errors: List[str] = []
        self.notes: List[str] = []
        """What is not filled in a source (e.g. an IDS it does not have): said
        softly, since sources often hold different IDSs."""
        self._time: Optional[float] = None
        self._mode = "pan"
        self._image_curves: List[Curve] = []
        self._image = None
        self._colorbar = None
        self._contours: List[object] = []  # contour lines, one item per level
        self._image_grid: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._image_time: Optional[float] = None
        self._title_time: Optional[float] = None
        self._time_line = None
        self._bars: List[Curve] = []  # members of wildcard traces, as bars per element
        self._categories: List[str] = []  # the elements the bars stand for
        self._one_time: Optional[str] = None
        """The kind of a panel of time traces or time-space images whose data is all
        at one time, and which then shows it as it is at that time (see load)."""
        self._value_labels: List[object] = []  # values written on the bars
        self._cursor_items = None  # the crosshair of the Point mode, once needed
        self._placing_cursor = False
        self._labeled_heights: List[float] = []  # heights of the bars labeled
        self._bar_room = None  # an invisible line: the room the bars take
        self._member_colors: dict = member_colors if member_colors is not None else {}
        self._legend_entries: List[Tuple[object, str]] = []
        self._legend_proxies: List[object] = []  # legend samples of the line styles
        self._styled_sources = False  # sources are told apart by line style
        self._shape_labels: List[object] = []  # names written on shapes (e.g. coils)
        self._named_signals: set = set()  # signals whose shapes are named on the map

        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.set_current(False)
        self.setAcceptDrops(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Source selector for maps and time-space images (one image at a time)
        self.header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(self.header)
        header_layout.setContentsMargins(4, 0, 4, 0)
        header_layout.addStretch(1)
        show_label = QtWidgets.QLabel("Show:")
        show_label.setStyleSheet(f"font-size: {_FONT_SIZE}pt;")
        header_layout.addWidget(show_label)
        self.image_selector = QtWidgets.QComboBox()
        self.image_selector.setStyleSheet(f"font-size: {_FONT_SIZE}pt;")
        self.image_selector.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.image_selector.currentIndexChanged.connect(self._on_image_selected)
        header_layout.addWidget(self.image_selector)
        self.header.hide()
        layout.addWidget(self.header)

        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        self.message.setContentsMargins(4, 0, 4, 0)
        self.message.hide()
        layout.addWidget(self.message)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setMinimumSize(120, 60)
        self.graphics.ci.setContentsMargins(0, 0, 0, 0)
        self.graphics.ci.setSpacing(0)
        # Let drops reach the panel instead of the graphics view
        self.graphics.setAcceptDrops(False)
        self.graphics.viewport().setAcceptDrops(False)
        layout.addWidget(self.graphics, stretch=1)
        self.plot = self.graphics.addPlot(row=0, col=0)
        self._style_plot()
        self.legend = None
        self._new_legend()
        # The title is drawn inside the plot, like in jScope, to keep panels compact
        self.title_item = pg.TextItem(color="k", anchor=(0, 0),
                                      fill=pg.mkBrush(255, 255, 255, 150))
        title_font = QtGui.QFont()
        title_font.setPointSize(_FONT_SIZE + 1)
        title_font.setBold(True)
        self.title_item.setFont(title_font)
        self.title_item.setParentItem(self.plot.getViewBox())
        self.title_item.setPos(2, 1)
        self.title_item.setZValue(100)
        self._install_menu()
        # The room above bars for their values depends on the height of the plot
        self.plot.getViewBox().sigResized.connect(lambda view: self._make_bar_room())
        scene = self.graphics.scene()
        scene.sigMouseClicked.connect(self._on_click)
        scene.sigMouseMoved.connect(self._on_move)
        self.graphics.viewport().installEventFilter(self)  # the mouse leaving
        self.apply_config()

    def _style_plot(self) -> None:
        plot = self.plot
        plot.layout.setContentsMargins(0, 2, 2, 0)
        plot.layout.setHorizontalSpacing(0)
        plot.layout.setVerticalSpacing(0)
        plot.showGrid(x=True, y=True, alpha=0.25)
        font = QtGui.QFont()
        font.setPointSize(_FONT_SIZE)
        for name in ("left", "bottom", "right"):
            axis = plot.getAxis(name)
            # Ticks point into the plot, as in jScope; values are shown as they are
            # (pyqtgraph would otherwise rescale them, e.g. "(x0.001)")
            # Ticks and grid lines at the major tick values only (no minor grid)
            axis.setStyle(tickFont=font, tickLength=-4, tickTextOffset=2, maxTickLevel=0)
            axis.enableAutoSIPrefix(False)
        plot.getAxis("left").setWidth(LEFT_AXIS_WIDTH)
        plot.getViewBox().suggestPadding = self._suggest_padding

    def _new_legend(self) -> None:
        """Replace the legend by an empty one (pyqtgraph's clear() keeps the rows
        of the removed entries, which then stretch the next ones)."""
        old = self.plot.legend
        if old is not None:
            if old.scene() is not None:
                old.scene().removeItem(old)
            # LegendItem.setParentItem would anchor itself to the missing parent
            QtWidgets.QGraphicsWidget.setParentItem(old, None)
            self.plot.legend = None
        self.legend = self.plot.addLegend(
            offset=(-5, 3), labelTextSize=f"{_FONT_SIZE}pt",
            brush=pg.mkBrush(255, 255, 255, 200), verSpacing=-6,
        )

    def _suggest_padding(self, axis: int) -> float:
        """Auto-range padding: a few pixels along x, enough for the tick labels at
        both ends (pyqtgraph hides labels that stick out of the plot), and room for
        the title on y."""
        if axis == 1 and self.kind != "map":
            return 0.1
        view = self.plot.getViewBox()
        size = view.width() if axis == 0 else view.height()
        if size <= 0:
            return 0.02
        return min(max(EDGE_PADDING / size, 0.005), 0.05)

    # ------------------------------------------------------------------ config

    def title(self) -> str:
        return self.config.title or (
            self.config.signals[0].label or self.config.signals[0].path
            if self.config.signals
            else "empty panel"
        )

    def set_current(self, current: bool) -> None:
        """Frame the current panel; the others have no frame (their axes tell them
        apart), which keeps them clear of gray lines."""
        color = style.CURRENT_PANEL_COLOR if current else "transparent"
        self.setStyleSheet(f"Panel {{ border: 1px solid {color}; }}")

    def apply_config(self) -> None:
        """Apply the settings that do not require reloading data."""
        cfg = self.config
        axes = self._axes()
        self.plot.setLogMode(x=axes.log_x, y=axes.log_y)
        self.legend.setVisible(cfg.legend and bool(self.legend.items))
        self.plot.getViewBox().setAspectLocked(cfg.equal_aspect)
        self._update_title()
        self._update_labels()
        self.apply_ranges()

    def apply_ranges(self) -> None:
        axes = self._axes()
        view = self.plot.getViewBox()
        if axes.x_range:
            view.setXRange(*axes.x_range, padding=0)
        else:
            view.enableAutoRange(axis=pg.ViewBox.XAxis)
        if axes.y_range:
            view.setYRange(*axes.y_range, padding=0)
        else:
            view.enableAutoRange(axis=pg.ViewBox.YAxis)

    def _axes(self) -> _Axes:
        """The axis settings of the configuration, as they apply to what is shown.

        Data at one time is shown as it is at that time (see load): what the
        configuration says of the time axis no longer applies, and the y axis of a
        time-space image becomes the x axis of its profiles.
        """
        cfg = self.config
        if self._one_time == "trace":  # bars
            return _Axes(None, cfg.y_label, None, cfg.y_range, False, False)
        if self._one_time == "spacetime":  # profiles
            return _Axes(cfg.y_label, None, cfg.y_range, None, cfg.log_y, False)
        return _Axes(cfg.x_label, cfg.y_label, cfg.x_range, cfg.y_range, cfg.log_x,
                     cfg.log_y)

    def autoscale(self) -> None:
        self.plot.getViewBox().enableAutoRange()

    def add_signal(self, path: str) -> None:
        self.config.signals.append(SignalConfig(path))
        self.signalsChanged.emit(self, "add signal")

    def clear_signals(self) -> None:
        self.config.signals.clear()
        self.signalsChanged.emit(self, "clear signals")

    # ------------------------------------------------------ layout (main window)

    def x_key(self, column: int) -> Optional[str]:
        """Identifies the x quantity, for sharing x axes between panels.

        Time is shared across the window; a profile coordinate only within a column.
        """
        if self.kind in ("trace", "spacetime"):
            return "time"
        if self.kind == "profile" and self.curves and not self._bars:
            return f"{column}:{self.curves[0].data.x_axis.title()}"
        return None

    def set_x_tick_labels(self, visible: bool) -> None:
        """Show or hide the x tick labels and label (hidden in stacked panels)."""
        axis = self.plot.getAxis("bottom")
        axis.setStyle(showValues=visible)
        axis.showLabel(visible)

    def has_colorbar(self) -> bool:
        return self._colorbar is not None

    def set_right_padding(self, width: int) -> None:
        """Reserve ``width`` pixels on the right, to line up with color bars."""
        if width > 0 and self._colorbar is None:
            self.plot.showAxis("right")
            axis = self.plot.getAxis("right")
            axis.setStyle(showValues=False)
            axis.setWidth(width)
        else:
            self.plot.hideAxis("right")

    # ----------------------------------------------------------------- loading

    def clear(self) -> None:
        """Remove all plotted items (the configuration is kept)."""
        for curve in self.curves:
            if curve.item is not None:
                self.plot.removeItem(curve.item)
        self.curves = []
        self._new_legend()
        self._legend_entries = []
        self._legend_proxies = []
        self._styled_sources = False
        for label in self._shape_labels:
            self.plot.removeItem(label)
        self._shape_labels = []
        self._named_signals = set()
        self._bars = []
        self._categories = []
        self._remove_value_labels()
        if self._bar_room is not None:
            self.plot.removeItem(self._bar_room)
            self._bar_room = None
        self.plot.getAxis("bottom").setTicks(None)
        self._remove_image()
        if self._time_line is not None:
            self.plot.removeItem(self._time_line)
            self._time_line = None
        self.kind = None
        self._one_time = None
        self.errors = []
        self.notes = []
        self._image_curves = []
        self._image_time = self._title_time = None
        with QtCore.QSignalBlocker(self.image_selector):
            self.image_selector.clear()
        self.header.hide()
        self._hide_cursor()

    def load(self, sources: List[DataSource], time: Optional[float]) -> List[str]:
        """Read all signals of this panel from ``sources`` and draw them.

        Returns:
            Error messages for the signals that could not be shown; what is not
            filled in a source goes to :attr:`notes`.
        """
        shown_image = self.image_selector.currentText()
        self.clear()
        errors, notes = [], []
        loaded = []
        for signal_index, signal in enumerate(self.config.signals):
            try:
                paths = parse_alternatives(signal.path)
                x_paths = parse_alternatives(signal.x) if signal.x else [None]
            except IdscopeError as exc:
                errors.append(str(exc))
                continue
            for source_index, source in enumerate(sources):
                if signal.sources is not None and source.label not in signal.sources:
                    continue
                try:
                    data = source.first_signal(paths, x_paths)
                except NoDataError as exc:
                    notes.append(f"{source.label}: {exc}")
                    continue
                except IdscopeError as exc:
                    errors.append(f"{source.label}: {exc}")
                    continue
                loaded.append((signal_index, signal, source_index, source.label, data))

        self.kind = self._resolve_kind(loaded)
        if self.kind in ("trace", "spacetime") and _at_one_time(loaded):
            # Against time, data that is all at one time (e.g. what a code gives
            # for the time it was run for) is a lone point, or an image one time
            # wide: it is shown as it is at that time instead, traces as bars of
            # their values and time-space images as the profiles
            self._one_time, self.kind = self.kind, "profile"
        axes = self._axes()
        self.plot.setLogMode(x=axes.log_x, y=axes.log_y)
        accepted = []
        for entry in loaded:
            problem = self._incompatibility(entry[4])
            if problem:
                errors.append(f"{entry[3]}: {problem}")
            else:
                accepted.append(entry)
        sources_shown = sorted({(entry[2], entry[3]) for entry in accepted})
        n_signals = len(self.config.signals)
        if self._one_time == "trace":
            self._add_bars_at_one_time(accepted, sources_shown)
        else:
            for entry in accepted:
                self._add_curves(*entry, n_sources=len(sources_shown),
                                 n_signals=n_signals)
        if len(sources_shown) > 1 and self._styled_sources:
            # Colors tell the signals (or members) apart: say which line style is
            # which source
            for source_index, source in sources_shown:
                proxy = pg.PlotDataItem(pen=style.source_pen(source_index))
                self._legend_proxies.append(proxy)
                self._legend_entries.append((proxy, source))
        self._fill_legend()
        if self._categories:
            self._set_categories()
        if self.kind in ("map", "spacetime"):
            self._setup_images(shown_image)
        if self.kind in ("trace", "spacetime"):
            self._time_line = pg.InfiniteLine(
                angle=90,
                movable=True,
                pen=pg.mkPen(style.TIME_LINE_COLOR, width=1,
                             style=QtCore.Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen(style.CURRENT_PANEL_COLOR, width=2),
            )
            self._time_line.setZValue(10)
            self._time_line.sigDragged.connect(
                lambda line: self.timeRequested.emit(float(line.value()))
            )
            self.plot.addItem(self._time_line, ignoreBounds=True)
        self.errors, self.notes = errors, notes
        # An empty legend would still draw its (white) box
        self.legend.setVisible(self.config.legend and bool(self.legend.items))
        self._show_message()
        self._update_labels()
        self.set_time(time)
        self.apply_ranges()
        return errors

    def _resolve_kind(self, loaded) -> Optional[str]:
        if self.config.kind != "auto":
            return self.config.kind
        for entry in loaded:
            data = entry[4]
            if isinstance(data, Channels) and data.kind == TRACE:
                # a value per element, e.g. per coil: bars, unless they are traces
                return "trace" if data.is_time_dependent else "profile"
            return _DATA_KIND_TO_PANEL[data.kind]
        return None

    def _incompatibility(self, data: SignalData) -> Optional[str]:
        kind = self.kind
        if isinstance(data, Shapes):
            if kind == "map":
                return None
            return (f"'{data.path.text}' is a shape in the (R, Z) plane and can only be "
                    "shown in a map panel.")
        if self._one_time == "trace":  # bars of the values of traces
            if data.kind == TRACE:
                return None
            return (f"'{data.path.text}' is a {data.kind} and cannot be shown in a "
                    "trace panel.")
        if isinstance(data, Channels):
            if kind == "profile" and data.kind == TRACE:
                return None  # a bar per element, at the selected time
            if kind == "map" and data.kind == PROFILE:
                return None  # curves over the map
            if kind == "spacetime":
                return (f"'{data.path.text}' has wildcards and cannot be shown as a "
                        "time-space image.")
        if kind == "spacetime":
            if not isinstance(data, ProfileSeries) or data.time is None:
                return (f"'{data.path.text}' is not a time-dependent profile and cannot "
                        "be shown as a time-space image.")
            return None
        if kind == "map" and isinstance(data, ProfileSeries):
            return None  # a curve drawn over the map, e.g. the plasma boundary
        if _DATA_KIND_TO_PANEL[data.kind] != kind:
            return (f"'{data.path.text}' is a {data.kind} and cannot be shown in a "
                    f"{kind} panel.")
        return None

    def _add_curves(self, signal_index: int, signal: SignalConfig, source_index: int,
                    source: str, data: SignalData, n_sources: int, n_signals: int) -> None:
        """Create the curves (or bars, or shapes) of one signal of one source.

        With several sources, curves take the color of their signal (or member) and
        the line style of their source when that is enough to tell them apart; the
        legend then has an entry per signal and one per source.
        """
        base = signal.label or data.path.label
        suffix = f" [{source}]" if n_sources > 1 else ""
        if isinstance(data, Channels) and data.kind == TRACE and self.kind == "profile":
            label = source if n_signals == 1 and n_sources > 1 else base + suffix
            self._add_bars(signal, signal_index, source, source_index, data,
                           base + suffix, label)
            return
        styled = n_sources > 1 and (n_signals > 1 or isinstance(data, Channels))
        self._styled_sources |= styled
        if isinstance(data, Channels):
            overlay = self.kind == "map"  # e.g. the parts of a wall: one pen
            for k, (member, member_name) in enumerate(zip(data.members, data.names)):
                name = member_name if n_signals == 1 else f"{member_name} {base}"
                if overlay:
                    pen = style.curve_pen(signal_index, source_index, n_signals,
                                          signal.color)
                    label = (base if styled else base + suffix) if k == 0 else None
                else:
                    # the same member, e.g. "source nbi", has one color in the window;
                    # in a panel with several signals, a color per signal
                    key = (data.axis_label, member_name,
                           signal_index if n_signals > 1 else None)
                    color = self._member_colors.setdefault(key, len(self._member_colors))
                    pen = style.member_pen(color, source_index, signal.color)
                    label = name
                item = self._make_item(member, signal, name + suffix, pen)
                self._add_curve(Curve(signal, signal_index, source, source_index, member,
                                      name + suffix, item, label))
            return
        pen = style.curve_pen(signal_index, source_index, n_signals, signal.color)
        name = base + suffix
        if isinstance(data, Shapes):
            item = self._make_shape_item(data, pen, source_index)
            if data.shape == GEOMETRY and signal_index not in self._named_signals:
                self._named_signals.add(signal_index)
                self._name_shapes(data)
        else:
            item = self._make_item(data, signal, name, pen)
        self._add_curve(Curve(signal, signal_index, source, source_index, data, name,
                              item, base if styled else name))

    def _add_curve(self, curve: Curve) -> None:
        self.curves.append(curve)
        item = curve.item
        if curve.label is None or item is None or isinstance(item, pg.InfiniteLine):
            return
        if all(label != curve.label for _, label in self._legend_entries):
            self._legend_entries.append((item, curve.label))

    def _fill_legend(self) -> None:
        entries = self._legend_entries
        self.legend.setColumnCount(1 if len(entries) <= 4 else 2 if len(entries) <= 12
                                   else 3)
        for item, label in entries:
            self.legend.addItem(item, label)

    def _make_item(self, data: SignalData, signal: SignalConfig, name: str, pen):
        if self.kind == "trace":
            if data.time is None:  # constant: a horizontal line
                line = pg.InfiniteLine(
                    pos=float(signal.scale * data.values + signal.offset), angle=0,
                    pen=pen, label=name,
                    labelOpts={"position": 0.05, "color": pen.color(), "movable": True},
                )
                self.plot.addItem(line)
                return line
            values = data.values * signal.scale + signal.offset
            item = pg.PlotDataItem(data.time, values, pen=pen)
            # Add before enabling clipping: pyqtgraph needs the ViewBox for it
            self.plot.addItem(item)
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
            _mark_single_point(item, data.time, values)
            return item
        if isinstance(data, ProfileSeries) and self.kind in ("profile", "map"):
            if self.kind == "map":
                pen.setWidthF(2.0)  # stand out on the image
            item = pg.PlotDataItem(pen=pen)
            item.setZValue(20)
            self.plot.addItem(item)
            return item
        return None  # images are created in _setup_images

    def _make_shape_item(self, data: Shapes, pen, source_index: int):
        if data.shape == POINT:
            symbol = style.SYMBOLS[source_index % len(style.SYMBOLS)]
            item = pg.PlotDataItem(pen=None, symbol=symbol, symbolSize=10,
                                   symbolPen=pg.mkPen(pen.color(), width=2),
                                   symbolBrush=None)
        elif data.shape == MARKERS:  # often thousands: small dots, no outline
            item = pg.PlotDataItem(pen=None, symbol="o", symbolSize=3, symbolPen=None,
                                   symbolBrush=pg.mkBrush(pen.color()))
        elif data.shape in (GEOMETRY, ANNULAR):  # filled (a vessel layer: its ring)
            fill = QtGui.QColor(pen.color())
            fill.setAlpha(60)
            item = pg.PlotDataItem(pen=pen, connect="finite", fillLevel="enclosed",
                                   fillBrush=fill)
        else:
            pen.setWidthF(2.0)
            item = pg.PlotDataItem(pen=pen, connect="finite")
        item.setZValue(15)
        self.plot.addItem(item)
        return item

    def _name_shapes(self, data: Shapes) -> None:
        """Write the name of every shape (e.g. coil) on the map."""
        groups: dict = {}
        for outline, name in zip(data.slices[0] if data.slices else [], data.names):
            if name and outline.size:
                groups.setdefault(name.split("/")[0], []).append(outline)
        if not groups or len(groups) > _MAX_SHAPE_LABELS:
            return
        font = QtGui.QFont()
        font.setPointSize(_FONT_SIZE - 1)
        for name, outlines in groups.items():
            points = np.vstack(outlines)
            center = np.nanmean(points, axis=0)
            label = pg.TextItem(name, color="#303030", anchor=(0.5, 0.5))
            label.setFont(font)
            label.setPos(float(center[0]), float(center[1]))
            label.setZValue(30)
            self.plot.addItem(label, ignoreBounds=True)
            self._shape_labels.append(label)

    # -------------------------------------------------------------------- bars

    def _add_bars(self, signal: SignalConfig, signal_index: int, source: str,
                  source_index: int, data: Channels, name: str, label: Optional[str],
                  color: Optional[str] = None) -> None:
        """Values of the members of a signal with wildcards (e.g. the current of
        every coil) as bars, at the selected time."""
        for member_name in data.names:
            if member_name not in self._categories:
                self._categories.append(member_name)
        color = color or signal.color or style.PALETTE[len(self._bars) % len(style.PALETTE)]
        item = pg.BarGraphItem(x=[0.0], height=[0.0], width=0.8, brush=color,
                               pen=pg.mkPen(QtGui.QColor(color).darker(140)))
        item.setVisible(False)  # until the first set_time
        self.plot.addItem(item)
        curve = Curve(signal, signal_index, source, source_index, data, name, item, label)
        self._bars.append(curve)
        self._add_curve(curve)

    def _add_bars_at_one_time(self, accepted, sources_shown) -> None:
        """Time traces that are all at one time, as bars of their values: a bar per
        signal, or per member of a signal with wildcards (e.g. per coil).

        The bars of a source have its color, and the legend says which source is
        which; when signals share their members (e.g. the currents and the
        voltages of the coils), the bars have a color per signal and source instead,
        as those of signals with wildcards otherwise do.
        """
        n_sources = len(sources_shown)
        rank = {index: k for k, (index, _) in enumerate(sources_shown)}
        bars, signals_of = [], {}
        for signal_index, signal, source_index, source, data in accepted:
            base = signal.label or data.path.label
            if not isinstance(data, Channels):  # a bar of its own, named after it
                data = Channels(data.path, [data], [base], data.value_axis,
                                data.documentation)
            for name in data.names:
                signals_of.setdefault(name, set()).add(signal_index)
            bars.append((signal_index, signal, source_index, source, data, base))
        shared = any(len(indices) > 1 for indices in signals_of.values())
        for signal_index, signal, source_index, source, data, base in bars:
            suffix = f" [{source}]" if n_sources > 1 else ""
            if shared:
                color, label = None, base + suffix
            else:
                color = style.PALETTE[rank[source_index] % len(style.PALETTE)]
                if n_sources == 1 and signal.color:
                    color = signal.color
                label = source if n_sources > 1 else None  # the bars are named
            self._add_bars(signal, signal_index, source, source_index, data,
                           base + suffix, label, color)

    def _set_categories(self) -> None:
        """Name the bars on the x axis (some of them when there are many)."""
        step = int(np.ceil(len(self._categories) / _MAX_CATEGORY_LABELS))
        ticks = [(float(i), name) for i, name in enumerate(self._categories)]
        self.plot.getAxis("bottom").setTicks([ticks[::step]])
        self._bar_room = pg.PlotDataItem(pen=pg.mkPen(0, 0, 0, 0))  # transparent
        self.plot.addItem(self._bar_room)

    def _update_bars(self, t: Optional[float]) -> List[Tuple[Curve, Optional[float]]]:
        """Heights of the bars at time t; returns the time of the values shown.

        The bars that stand for the same element (e.g. a coil in two sources) are
        side by side around it, as wide as the element with the most bars allows.
        """
        shown, drawn = [], []
        at: dict = {}  # element -> its bars
        for curve in self._bars:
            for name in curve.data.names:
                at.setdefault(name, []).append(curve)
        width = 0.8 / max((len(group) for group in at.values()), default=1)
        for curve in self._bars:
            values, t_slice = curve.data.values_at(t)
            x = np.empty(len(curve.data.names))
            for k, name in enumerate(curve.data.names):
                group = at[name]
                rank = next(i for i, other in enumerate(group) if other is curve)
                x[k] = self._categories.index(name) + (rank - (len(group) - 1) / 2) * width
            heights = curve.scaled(values)
            valid = np.isfinite(heights)
            curve.item.setOpts(x=x[valid], height=heights[valid], width=width)
            curve.item.setVisible(bool(valid.any()))
            shown.append((curve, t_slice))
            drawn += zip(x[valid], heights[valid])
        self._remove_value_labels()
        labeled = self._one_time == "trace" and len(drawn) <= _MAX_VALUE_LABELS
        if labeled:
            self._label_values(drawn)
        self._labeled_heights = [float(height) for _, height in drawn] if labeled else []
        self._make_bar_room()
        return shown

    def _make_bar_room(self) -> None:
        """Keep the room of a few elements at least (a single bar would fill the
        panel), and, for the height of the panel, room for the values written at
        the ends of the bars and for the title above them."""
        if self._bar_room is None:
            return
        n = len(self._categories)
        center, half = (n - 1) / 2, max(n, _MIN_BAR_SLOTS) / 2
        low = high = 0.0
        if self._labeled_heights:
            top = max(max(self._labeled_heights), 0.0)
            bottom = min(min(self._labeled_heights), 0.0)
            span = (top - bottom) or 1.0
            size = self.plot.getViewBox().height()
            size = size if size > 0 else 150.0  # not laid out yet
            if top > 0:
                high = top + _margin(_TOP_ROOM / size) * span
            if bottom < 0:
                low = bottom - _margin(_BOTTOM_ROOM / size) * span
        self._bar_room.setData([center - half, center + half], [low, high])

    def _label_values(self, bars) -> None:
        """Write the value of every bar ``(x, height)`` at its end."""
        font = QtGui.QFont()
        font.setPointSize(_FONT_SIZE - 1)
        for x, height in bars:
            label = pg.TextItem(f"{height:.4g}", color="#303030",
                                anchor=(0.5, 1.0) if height >= 0 else (0.5, 0.0))
            label.setFont(font)
            label.setPos(float(x), float(height))
            label.setZValue(20)
            self.plot.addItem(label, ignoreBounds=True)
            self._value_labels.append(label)

    def _remove_value_labels(self) -> None:
        for label in self._value_labels:
            self.plot.removeItem(label)
        self._value_labels = []

    # ------------------------------------------------------------------ images

    def _setup_images(self, shown: str = "") -> None:
        """Fill the image selector, keeping the image ``shown`` before a reload."""
        self._image_curves = [
            curve for curve in self.curves
            if self.kind == "spacetime" or isinstance(curve.data, (MapSeries, MeshSeries))
        ]
        with QtCore.QSignalBlocker(self.image_selector):
            for curve in self._image_curves:
                self.image_selector.addItem(curve.name)
            self.image_selector.setCurrentIndex(max(self.image_selector.findText(shown), 0))
        self.header.setVisible(len(self._image_curves) > 1)
        if self.kind == "spacetime":
            self._draw_spacetime()

    def _on_image_selected(self, index: int) -> None:
        if self.kind == "map":
            self._draw_map()
        elif self.kind == "spacetime":
            self._draw_spacetime()
        self._update_labels()

    def step_image(self, delta: int) -> None:
        """Show the next (or previous) image of the Show selector, round the end."""
        count = self.image_selector.count()
        if count > 1:
            self.image_selector.setCurrentIndex(
                (self.image_selector.currentIndex() + delta) % count
            )

    def _selected_image(self) -> Optional[Curve]:
        if not self._image_curves:
            return None
        index = max(self.image_selector.currentIndex(), 0)
        return self._image_curves[min(index, len(self._image_curves) - 1)]

    def _draw_map(self) -> None:
        curve = self._selected_image()
        if curve is None:
            self._image_time = None
            return
        data = curve.data
        if isinstance(data, MeshSeries):
            mesh, values, t = data.at(self._time)
            self._set_mesh(mesh, curve.scaled(values))
        else:
            x1, x2, z, t = data.at(self._time)
            self._set_image(x1, x2, curve.scaled(z))
        self._image_time = t

    def _draw_spacetime(self) -> None:
        curve = self._selected_image()
        if curve is None:
            return
        time, grid, z = curve.data.as_image()
        self._set_image(time, grid, curve.scaled(z))

    def _set_image(self, x1: np.ndarray, x2: np.ndarray, z: np.ndarray) -> None:
        """Show z(x1, x2) as an image, or as contour lines when ``contours`` is set."""
        if z.size == 0 or len(x1) != z.shape[0] or len(x2) != z.shape[1]:
            self._remove_image()
            return
        # Coordinates must increase for both image types
        if len(x1) > 1 and x1[0] > x1[-1]:
            x1, z = x1[::-1], z[::-1]
        if len(x2) > 1 and x2[0] > x2[-1]:
            x2, z = x2[::-1], z[:, ::-1]
        finite = z[np.isfinite(z)]
        levels = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if levels[0] == levels[1]:
            levels = (levels[0] - 0.5, levels[1] + 0.5)
        if self.config.contours > 0:
            self._draw_contour_lines(x1, x2, z, levels)
        else:
            self._draw_image(x1, x2, z, levels)
        self._image_grid = (x1, x2, z)
        self._update_colorbar_label()  # the color bar may be new

    def _set_mesh(self, mesh, values: np.ndarray) -> None:
        """Show the value of every cell of an unstructured grid, as filled cells
        (also when ``contours`` is set)."""
        if mesh is None or not len(values):
            self._remove_image()
            return
        finite = values[np.isfinite(values)]
        levels = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if levels[0] == levels[1]:
            levels = (levels[0] - 0.5, levels[1] + 0.5)
        if not isinstance(self._image, MeshItem):
            self._remove_image()  # another image type, or contour lines
            self._image = MeshItem(pg.colormap.get(style.COLORMAP))
            self.plot.addItem(self._image)
        self._image.setData(mesh, values, levels)
        if self._colorbar is None:
            self._colorbar = pg.ColorBarItem(
                colorMap=pg.colormap.get(style.COLORMAP), interactive=True, width=10,
            )
            self._colorbar.setImageItem(self._image, insert_in=self.plot)
            self._fit_colorbar()
        self._colorbar.setLevels(levels)
        self._update_colorbar_label()  # the color bar may be new

    def _image_value_at(self, x: float, y: float) -> Optional[float]:
        """The value of the image shown at (x, y): of the grid point closest to it,
        or of the cell it is in."""
        if isinstance(self._image, MeshItem):
            return self._image.value_at(x, y)
        if self._image_grid is not None:
            x1, x2, z = self._image_grid
            if len(x1) and len(x2):
                return float(z[int(np.argmin(np.abs(x1 - x))), int(np.argmin(np.abs(x2 - y)))])
        return None

    def _draw_image(self, x1, x2, z, levels) -> None:
        uniform = _is_uniform(x1) and _is_uniform(x2)
        wanted = pg.ImageItem if uniform else pg.PColorMeshItem
        if not isinstance(self._image, wanted):
            # Another image type, or contour lines: start over (with the color bar)
            self._remove_image()
        if uniform:
            if self._image is None:
                self._image = pg.ImageItem(axisOrder="col-major")
                self.plot.addItem(self._image)
            self._image.setImage(z, autoLevels=False, levels=levels)
            dx = x1[1] - x1[0] if len(x1) > 1 else 1.0
            dy = x2[1] - x2[0] if len(x2) > 1 else 1.0
            self._image.setRect(QtCore.QRectF(
                x1[0] - dx / 2, x2[0] - dy / 2, dx * len(x1), dy * len(x2)
            ))
        else:
            # Irregular grid (e.g. non-uniform time steps): draw cells
            ex, ey = np.meshgrid(_edges(x1), _edges(x2), indexing="ij")
            z_filled = np.nan_to_num(z, nan=levels[0])
            if self._image is None:
                self._image = pg.PColorMeshItem(
                    ex, ey, z_filled, colorMap=pg.colormap.get(style.COLORMAP),
                    levels=levels, enableAutoLevels=False,
                )
                self.plot.addItem(self._image)
            else:
                self._image.setData(ex, ey, z_filled)
        if self._colorbar is None:
            self._colorbar = pg.ColorBarItem(
                colorMap=pg.colormap.get(style.COLORMAP), interactive=True, width=10,
            )
            self._colorbar.setImageItem(self._image, insert_in=self.plot)
            self._fit_colorbar()
        self._colorbar.setLevels(levels)

    def _draw_contour_lines(self, x1, x2, z, levels) -> None:
        """Contour lines colored by level, without filling (no image)."""
        if self._image is not None:
            self._remove_image()  # the color bar of an image is interactive
        self._remove_contours()
        colormap = pg.colormap.get(style.CONTOUR_COLORMAP)
        for level, x, y in contour_lines(x1, x2, z, self.config.contours):
            fraction = (level - levels[0]) / (levels[1] - levels[0])
            color = colormap.map(fraction, mode="qcolor")
            item = pg.PlotDataItem(x, y, pen=pg.mkPen(color, width=1.5), connect="finite")
            item.setZValue(5)
            self.plot.addItem(item)
            self._contours.append(item)
        if self._colorbar is None:
            self._colorbar = pg.ColorBarItem(
                values=levels, colorMap=colormap, interactive=False, width=10,
            )
            self.plot.layout.addItem(self._colorbar, 2, 5)
            self._fit_colorbar()
        self._colorbar.setLevels(levels)

    def _fit_colorbar(self) -> None:
        self._colorbar.setFixedWidth(COLORBAR_WIDTH)
        # The color bar goes in column 5 of the plot layout, after a gap column
        self.plot.layout.setColumnFixedWidth(4, COLORBAR_GAP)
        axis = self._colorbar.getAxis("right")
        axis.setStyle(tickFont=self.plot.getAxis("left").style["tickFont"], maxTickLevel=0)
        # Values such as densities (2e+19) fit its width as 2e19
        default = type(axis).tickStrings
        axis.tickStrings = lambda values, scale, spacing: [
            _compact_exponent(text) for text in default(axis, values, scale, spacing)]

    def _remove_contours(self) -> None:
        for item in self._contours:
            self.plot.removeItem(item)
        self._contours = []

    def _remove_image(self) -> None:
        self._remove_contours()
        if self._image is not None:
            self.plot.removeItem(self._image)
            self._image = None
        if self._colorbar is not None:
            self.plot.layout.removeItem(self._colorbar)
            self.plot.layout.setColumnFixedWidth(4, 0)
            scene = self._colorbar.scene()
            if scene is not None:
                scene.removeItem(self._colorbar)
            self._colorbar = None
        self._image_grid = None

    # -------------------------------------------------------------------- time

    def set_time(self, t: Optional[float]) -> None:
        """Show the data at time ``t`` (profiles and maps) and move the time line."""
        self._time = t
        if self._time_line is not None:
            self._time_line.setVisible(t is not None)
            if t is not None:
                self._time_line.setValue(t)
        shown = []
        for curve in self.curves:
            if isinstance(curve.data, ProfileSeries) and curve.item is not None:
                x, y, t_slice = curve.data.at(t)
                y = curve.scaled(y)
                curve.item.setData(x, y)
                _mark_single_point(curve.item, x, y)
                shown.append((curve, t_slice))
            elif isinstance(curve.data, Shapes):
                outlines, t_slice = curve.data.at(t)
                x, y = _joined(outlines, curve.data.shape not in (POINT, MARKERS))
                curve.item.setData(x, y)
                if curve.data.is_time_dependent:
                    shown.append((curve, t_slice))
        shown += self._update_bars(t)
        if self.kind == "map":
            self._draw_map()
        # The time of the slices is shown once in the title when it is the same for
        # all curves (and the map), otherwise in the legend of every curve
        times = [t_slice for _, t_slice in shown if t_slice is not None]
        if self.kind == "map" and self._image_time is not None:
            times.insert(0, self._image_time)
        common = bool(times) and bool(np.allclose(times, times[0], rtol=1e-9, atol=0))
        self._title_time = times[0] if times and (common or self.kind == "map") else None
        for curve, t_slice in shown:
            self._set_legend_text(curve, None if common else t_slice)
        self._update_title()
        self._refresh_cursor()

    def _set_legend_text(self, curve: Curve, t_slice: Optional[float]) -> None:
        if curve.label is None:
            return
        text = curve.label if t_slice is None else f"{curve.label}  @ {t_slice:.4g} s"
        for sample, label in self.legend.items:
            if sample.item is curve.item:
                label.setText(text)

    def time_arrays(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Time vectors of the shown signals: ``(slices, traces)``.

        Slices come from profiles and maps; stepping through them is what the time
        slider does.
        """
        slices, traces = [], []
        for curve in self.curves:
            if isinstance(curve.data, Channels):  # bars: stepped through like slices
                # (unless they are traces at one time: their time is a trace's)
                times = [m.time for m in curve.data.members if m.time is not None]
                if self._one_time == "trace":
                    traces += times
                else:
                    slices += times
                continue
            time = curve.data.time
            if time is None:
                continue
            (traces if curve.data.kind == TRACE else slices).append(time)
        return slices, traces

    def values_at_time(self, t: float) -> List[str]:
        """``name = value`` of every trace at time ``t`` (nearest sample)."""
        result = []
        for curve in self.curves:
            data = curve.data
            if data.kind != TRACE or isinstance(data, Channels):
                continue
            if data.time is None:
                value = float(data.values)
            else:
                index = data.time_index(t)
                if index is None:
                    continue
                value = float(data.values[index])
            result.append(f"{curve.name} = {_format(curve.scaled(value))}")
        return result

    # ------------------------------------------------------------------ labels

    def _update_title(self) -> None:
        """The title, with the time of the data, and in gray how many signals are not
        filled in some source (their notes when hovered)."""
        parts = [self.config.title] if self.config.title else []
        if self.kind in ("profile", "map") and self._title_time is not None:
            parts.append(f"t = {self._title_time:.4g} s")
        text = html.escape("  ".join(parts)).replace("  ", "&nbsp; ")
        tip = ""
        if self.notes and self.curves:  # without curves, the message says it
            n = len(self.notes)
            note = f"{n} signal{'s' if n > 1 else ''} not filled"
            text += (f"&nbsp;&nbsp; <span style='color: {style.NOTE_COLOR}; "
                     f"font-weight: normal'>{note}</span>")
            tip = "\n".join(self.notes)
        self.title_item.setHtml(text)
        for item in (self.title_item, self.title_item.textItem):
            item.setToolTip(tip)
        self.title_item.setVisible(bool(text))

    def _update_labels(self) -> None:
        x_title = y_title = ""
        shown = self._selected_image() if self.kind in ("map", "spacetime") else None
        curves = [shown] if shown is not None else self.curves
        if curves:
            data = curves[0].data
            if self.kind == "trace":
                x_title = "time [s]"
            elif self.kind == "profile" and self._bars:
                x_title = self._bars[0].data.axis_label
            elif self.kind == "profile":
                x_title = data.x_axis.title()
            elif self.kind == "map" and hasattr(data, "x2_axis"):
                x_title, y_title = data.x_axis.title(), data.x2_axis.title()
            elif self.kind == "map":  # only curves over an empty map
                x_title, y_title = data.x_axis.title(), data.value_axis.title()
            elif self.kind == "spacetime":
                x_title, y_title = "time [s]", data.x_axis.title()
            if self.kind in ("trace", "profile"):
                units = {c.data.value_axis.units for c in curves}
                scaled = any(c.signal.scale != 1.0 for c in curves)
                if len(units) == 1 and not scaled:
                    only = units.pop()
                    y_title = (
                        curves[0].data.value_axis.title()
                        if len(self.config.signals) == 1
                        else (f"[{only}]" if only else "")
                    )
        self._update_colorbar_label()
        style_args = {"font-size": f"{_FONT_SIZE}pt"}
        axes = self._axes()
        x_text = axes.x_label if axes.x_label is not None else x_title
        y_text = axes.y_label if axes.y_label is not None else y_title
        self.plot.setLabel("bottom", x_text, **style_args)
        # Always show a y label (a space when empty): its width is added to the
        # fixed axis width, and must be the same in all panels to line them up
        self.plot.setLabel("left", y_text or " ", **style_args)
        self._update_title()

    def _update_colorbar_label(self) -> None:
        shown = self._selected_image()
        if self._colorbar is None or shown is None:
            return
        scaled = shown.signal.scale != 1.0 or shown.signal.offset != 0.0
        # Scaled values no longer have the units of the Data Dictionary
        label = shown.name if scaled else shown.data.value_axis.title()
        self._colorbar.setLabel("left", label, **{"font-size": f"{_FONT_SIZE}pt"})

    def _show_message(self) -> None:
        if not self.config.signals:
            text, color, tip = _HINT, style.NOTE_COLOR, ""
        elif self.errors and not self.curves:
            text, color = self.errors[0], style.ERROR_COLOR
            tip = "\n".join(self.errors + self.notes)
        elif self.errors:
            n = len(self.errors)
            text = f"{n} signal{'s' if n > 1 else ''} not shown (hover for details)"
            color, tip = style.ERROR_COLOR, "\n".join(self.errors + self.notes)
        elif self.notes and not self.curves:
            text, color, tip = self.notes[0], style.NOTE_COLOR, "\n".join(self.notes)
        else:  # what is not filled is said in the title, if anything
            self.message.hide()
            return
        self.message.setText(text)
        self.message.setToolTip(tip)
        self.message.setStyleSheet(f"color: {color}; font-size: {_FONT_SIZE}pt;")
        self.message.show()

    # ------------------------------------------------------------------- mouse

    def set_mode(self, mode: str) -> None:
        """``zoom`` (rectangle zoom), ``pan`` or ``point`` (pick time, read values)."""
        self._mode = mode
        view = self.plot.getViewBox()
        if mode == "point":
            view.setMouseEnabled(x=False, y=False)
            self.graphics.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            view.setMouseEnabled(x=True, y=True)
            view.setMouseMode(pg.ViewBox.RectMode if mode == "zoom" else pg.ViewBox.PanMode)
            self.graphics.unsetCursor()
            self._hide_cursor()

    def _on_click(self, event) -> None:
        self.activated.emit(self)
        if event.double():
            self.setupRequested.emit(self)
            return
        if (
            self._mode == "point"
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.kind in ("trace", "spacetime")
        ):
            position = self.plot.getViewBox().mapSceneToView(event.scenePos())
            self.timeRequested.emit(float(position.x()))

    def _on_move(self, scene_pos) -> None:
        view = self.plot.getViewBox()
        if not view.sceneBoundingRect().contains(scene_pos):
            self._hide_cursor()
            return
        position = view.mapSceneToView(scene_pos)
        x, y = position.x(), position.y()
        if self._mode == "point" and self.kind in ("trace", "spacetime"):
            buttons = QtWidgets.QApplication.mouseButtons()
            if buttons & QtCore.Qt.MouseButton.LeftButton:
                self.timeRequested.emit(float(x))
        if self._mode == "point" and self.curves:
            self._show_cursor(x, y)
        self.hoverMessage.emit(self._describe_position(x, y))

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QtCore.QEvent.Type.Leave:
            self._hide_cursor()
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------- crosshair

    def _cursor(self):
        """The crosshair of the Point mode: two lines, dots on the curves and a label
        of their values (made when first needed)."""
        if self._cursor_items is None:
            pen = pg.mkPen(style.CURSOR_COLOR, width=1, style=QtCore.Qt.PenStyle.DotLine)
            lines = [pg.InfiniteLine(angle=angle, pen=pen, movable=False)
                     for angle in (90, 0)]
            dots = pg.ScatterPlotItem(size=7, pen=pg.mkPen("w", width=1))
            label = pg.TextItem(color="k", anchor=(0, 1), border=pg.mkPen("#b0b0b0"),
                                fill=pg.mkBrush("w"))
            font = QtGui.QFont()
            font.setPointSize(_FONT_SIZE)
            label.setFont(font)
            for z, item in enumerate(lines + [dots]):
                item.setZValue(40 + z)
                item.hide()
                self.plot.addItem(item, ignoreBounds=True)
            # The label is placed in pixels, over the legend and the title (which are
            # not in the coordinates of the view either)
            label.hide()
            label.setParentItem(self.plot.getViewBox())
            label.setZValue(110)
            self._cursor_items = (lines, dots, label)
            self.plot.getViewBox().sigTransformChanged.connect(
                lambda view: self._refresh_cursor())
        return self._cursor_items

    def _hide_cursor(self) -> None:
        if self._cursor_items is not None:
            lines, dots, label = self._cursor_items
            for item in (*lines, dots, label):
                item.hide()

    def _refresh_cursor(self) -> None:
        """The crosshair again where it is, with the values of the data now shown
        (after the time changed, e.g. with the keys) and its label where the view
        now shows that place."""
        if (self._cursor_items is not None and self._cursor_items[0][0].isVisible()
                and not self._placing_cursor):
            lines = self._cursor_items[0]
            self._show_cursor(lines[0].value(), lines[1].value())

    def _show_cursor(self, x: float, y: float) -> None:
        """The crosshair at (x, y), in the coordinates of the view, with the values of
        the curves there (of the bars of the element, or of the image)."""
        lines, dots, label = self._cursor()
        header, rows, points = self._values_at(x, y)
        lines[0].setValue(x)
        lines[1].setValue(y)
        dots.setData([p[0] for p in points], [p[1] for p in points],
                     brush=[pg.mkBrush(p[2]) for p in points])
        text = [f"<b>{html.escape(header)}</b>"] if header else []
        for color, name, value in rows[:_MAX_CURSOR_ROWS]:
            mark = f"<span style='color: {color}'>&#9632;</span> " if color else ""
            text.append(f"{mark}{html.escape(name)} = {_format(value)}")
        if len(rows) > _MAX_CURSOR_ROWS:
            text.append(f"… {len(rows) - _MAX_CURSOR_ROWS} more")
        label.setHtml("<br>".join(text))
        for item in (*lines, dots, label):
            item.setVisible(item is not label or bool(text))
        # Beside the cursor, on the side with more room, and within the plot (once
        # visible: a hidden text item does not update its size)
        view = self.plot.getViewBox()
        self._placing_cursor = True  # mapping may update the view, and call back
        try:
            point = view.mapFromView(QtCore.QPointF(x, y))  # pixels, y down
        finally:
            self._placing_cursor = False
        bounds = view.boundingRect()
        label.setAnchor((1.05 if point.x() > bounds.center().x() else -0.05,
                         -0.05 if point.y() < bounds.center().y() else 1.05))
        label.setPos(point)
        label.updateTransform(force=True)
        rect = label.mapRectToParent(label.boundingRect())
        dx = max(bounds.left() - rect.left(), 0.0) - max(rect.right() - bounds.right(), 0.0)
        dy = max(bounds.top() - rect.top(), 0.0) - max(rect.bottom() - bounds.bottom(), 0.0)
        if dx or dy:
            label.setPos(point + QtCore.QPointF(dx, dy))

    def _values_at(self, x: float, y: float):
        """What the crosshair says at (x, y): ``(header, [(color, name, value)],
        [(x, y, color) of the dots on the curves])``."""
        axes = self._axes()
        (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
        x_shown = _round_to_view(x, x1 - x0, axes.log_x)
        y_shown = _round_to_view(y, y1 - y0, axes.log_y)
        if self.kind in ("map", "spacetime"):
            header = (f"{self._axis_name('bottom', 'x')} = {_format(x_shown)}, "
                      f"{self._axis_name('left', 'y')} = {_format(y_shown)}")
            shown = self._selected_image()
            value = self._image_value_at(x, y) if shown is not None else None
            if value is None:  # no image, or outside of the cells of a grid
                return header, [], []
            return header, [(None, shown.name, value)], []
        if self._bars:
            index = int(round(x))
            if not 0 <= index < len(self._categories):
                return "", [], []
            category = self._categories[index]
            rows = []
            for curve in self._bars:
                if category in curve.data.names:
                    values, _ = curve.data.values_at(self._time)
                    value = curve.scaled(values)[curve.data.names.index(category)]
                    color = curve.item.opts["brush"]
                    rows.append((pg.mkBrush(color).color().name(), curve.name, value))
            return category, rows, []
        tolerance = 0.01 * abs(x1 - x0)
        header = (f"t = {_format(x_shown)} s" if self.kind == "trace" else
                  f"{self._axis_name('bottom', 'x')} = {_format(x_shown)}")
        rows, points = [], []
        for curve in self.curves:
            item = curve.item
            if isinstance(item, pg.InfiniteLine):  # a constant
                rows.append((item.pen.color().name(), curve.name, float(item.value())))
                continue
            if not isinstance(item, pg.PlotDataItem) or item.xData is None:
                continue
            value = _value_at(item.xData, item.yData, x, axes.log_x, tolerance)
            if value is None:
                continue
            pen = item.opts["pen"]
            color = pg.mkPen(pen).color().name() if pen is not None else "#000000"
            rows.append((color, curve.name, value))
            if axes.log_y and value <= 0:
                continue
            points.append((x, np.log10(value) if axes.log_y else value, color))
        return header, rows, points

    def _axis_name(self, name: str, default: str) -> str:
        return self.plot.getAxis(name).labelText.strip() or default

    def _describe_position(self, x: float, y: float) -> str:
        axes = self._axes()
        (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
        text = (f"x = {_format(_round_to_view(x, x1 - x0, axes.log_x))}, "
                f"y = {_format(_round_to_view(y, y1 - y0, axes.log_y))}")
        value = self._image_value_at(x, y)
        if value is not None:
            text += f", value = {_format(value)}"
        return text

    # ------------------------------------------------------------ context menu

    def _install_menu(self) -> None:
        menu = self.plot.getViewBox().menu
        menu.addSeparator()
        actions = [
            ("Setup…", lambda: self.setupRequested.emit(self)),
            ("Autoscale", self.autoscale),
            ("Clear signals", self.clear_signals),
            (None, None),
            ("Add panel below", lambda: self.layoutRequested.emit(self, "add_below")),
            ("Add column", lambda: self.layoutRequested.emit(self, "add_column")),
            ("Remove panel", lambda: self.layoutRequested.emit(self, "remove")),
        ]
        for text, slot in actions:
            if text is None:
                menu.addSeparator()
            else:
                menu.addAction(text).triggered.connect(slot)

    # ------------------------------------------------------------- drag & drop

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime.hasUrls() and all(url.isLocalFile() for url in mime.urls()):
            event.ignore()  # files: sources for the window, not signal paths
        elif mime.hasFormat(MIME_SIGNAL) or mime.hasText():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        mime = event.mimeData()
        if mime.hasFormat(MIME_SIGNAL):
            text = bytes(mime.data(MIME_SIGNAL)).decode()
        else:
            text = mime.text()
        paths = [line.strip() for line in text.splitlines() if line.strip()]
        if not paths:
            return
        event.acceptProposedAction()
        self.activated.emit(self)
        for path in paths:
            self.config.signals.append(SignalConfig(path))
        self.signalsChanged.emit(self, "drop signals")
