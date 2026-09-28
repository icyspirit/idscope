"""A plot panel: one cell of the scope grid (a "wave" in jScope)."""

import logging
from typing import List, NamedTuple, Optional, Tuple

import contourpy
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from idscope.config import PanelConfig, SignalConfig
from idscope.data import MAP, PROFILE, TRACE, MapSeries, ProfileSeries, SignalData
from idscope.data import parse_alternatives
from idscope.data.sources import DataSource
from idscope.errors import IdscopeError
from idscope.gui import style

logger = logging.getLogger(__name__)

MIME_SIGNAL = "application/x-idscope-signal"
"""Mime type of signal paths dragged from the IDS browser (one path per line)."""

LEFT_AXIS_WIDTH = 52
"""Width of the y tick labels (the axis label comes on top). The same for all panels
so that the plots of a column line up."""
COLORBAR_WIDTH = 64
"""Fixed width of color bars."""
COLORBAR_GAP = 4
"""Gap between a plot and its color bar."""
COLORBAR_SPACE = COLORBAR_WIDTH + COLORBAR_GAP
"""Room taken by a color bar; panels without one reserve it to stay aligned."""
EDGE_PADDING = 14
"""Space between the data and the ends of the x axis, in pixels."""

_DATA_KIND_TO_PANEL = {TRACE: "trace", PROFILE: "profile", MAP: "map"}
_HINT = "Double-click to set up, or drop a signal from the IDS browser"
_FONT_SIZE = 8  # points


class Curve(NamedTuple):
    """One signal of one source, as plotted in a panel."""

    signal: SignalConfig
    signal_index: int
    source: str
    source_index: int
    data: SignalData
    name: str
    item: Optional[object]  # PlotDataItem / InfiniteLine, None for images

    def scaled(self, values: np.ndarray) -> np.ndarray:
        return values * self.signal.scale + self.signal.offset


def _is_uniform(x: np.ndarray) -> bool:
    if len(x) < 2:
        return True
    step = np.diff(x)
    return bool(np.all(step > 0) and np.allclose(step, step[0], rtol=1e-3, atol=0))


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


def _format(value: float) -> str:
    return f"{value:.5g}" if np.isfinite(value) else "nan"


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
    signalsChanged = QtCore.Signal(object)
    """The signals of the panel changed (e.g. by a drop) and must be reloaded."""
    hoverMessage = QtCore.Signal(str)

    def __init__(self, config: PanelConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.kind: Optional[str] = None
        """Resolved kind: trace, profile, map or spacetime (None if nothing loaded)."""
        self.curves: List[Curve] = []
        self.errors: List[str] = []
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
        self.legend = self.plot.addLegend(
            offset=(-5, 3), labelTextSize=f"{_FONT_SIZE}pt",
            brush=pg.mkBrush(255, 255, 255, 200), verSpacing=-6,
        )
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
        scene = self.graphics.scene()
        scene.sigMouseClicked.connect(self._on_click)
        scene.sigMouseMoved.connect(self._on_move)
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
        color = style.CURRENT_PANEL_COLOR if current else "#c8c8c8"
        self.setStyleSheet(f"Panel {{ border: 1px solid {color}; }}")

    def apply_config(self) -> None:
        """Apply the settings that do not require reloading data."""
        cfg = self.config
        self.plot.setLogMode(x=cfg.log_x, y=cfg.log_y)
        self.legend.setVisible(cfg.legend and bool(self.legend.items))
        self.plot.getViewBox().setAspectLocked(cfg.equal_aspect)
        self._update_title()
        self._update_labels()
        self.apply_ranges()

    def apply_ranges(self) -> None:
        cfg = self.config
        view = self.plot.getViewBox()
        if cfg.x_range:
            view.setXRange(*cfg.x_range, padding=0)
        else:
            view.enableAutoRange(axis=pg.ViewBox.XAxis)
        if cfg.y_range:
            view.setYRange(*cfg.y_range, padding=0)
        else:
            view.enableAutoRange(axis=pg.ViewBox.YAxis)

    def autoscale(self) -> None:
        self.plot.getViewBox().enableAutoRange()

    def add_signal(self, path: str) -> None:
        self.config.signals.append(SignalConfig(path))
        self.signalsChanged.emit(self)

    # ------------------------------------------------------ layout (main window)

    def x_key(self, column: int) -> Optional[str]:
        """Identifies the x quantity, for sharing x axes between panels.

        Time is shared across the window; a profile coordinate only within a column.
        """
        if self.kind in ("trace", "spacetime"):
            return "time"
        if self.kind == "profile" and self.curves:
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
        self.legend.clear()
        self._remove_image()
        if self._time_line is not None:
            self.plot.removeItem(self._time_line)
            self._time_line = None
        self.kind = None
        self.errors = []
        self._image_curves = []
        self._image_time = self._title_time = None
        with QtCore.QSignalBlocker(self.image_selector):
            self.image_selector.clear()
        self.header.hide()

    def load(self, sources: List[DataSource], time: Optional[float]) -> List[str]:
        """Read all signals of this panel from ``sources`` and draw them.

        Returns:
            Error messages for the signals that could not be shown.
        """
        shown_image = self.image_selector.currentText()
        self.clear()
        errors = []
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
                except IdscopeError as exc:
                    errors.append(f"{source.label}: {exc}")
                    continue
                loaded.append((signal_index, signal, source_index, source.label, data))

        self.kind = self._resolve_kind(loaded)
        accepted = []
        for entry in loaded:
            problem = self._incompatibility(entry[4])
            if problem:
                errors.append(f"{entry[3]}: {problem}")
            else:
                accepted.append(entry)
        n_sources = len({entry[3] for entry in accepted})
        n_signals = len(self.config.signals)
        for signal_index, signal, source_index, source, data in accepted:
            name = signal.label or data.path.label
            if n_sources > 1:
                name += f" [{source}]"
            pen = style.curve_pen(signal_index, source_index, n_signals, signal.color)
            item = self._make_item(data, signal, name, pen)
            self.curves.append(
                Curve(signal, signal_index, source, source_index, data, name, item)
            )
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
        self.errors = errors
        self.legend.setColumnCount(2 if len(self.legend.items) > 4 else 1)
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
            return _DATA_KIND_TO_PANEL[entry[4].kind]
        return None

    def _incompatibility(self, data: SignalData) -> Optional[str]:
        kind = self.kind
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
            item = pg.PlotDataItem(
                data.time, data.values * signal.scale + signal.offset, pen=pen, name=name
            )
            # Add before enabling clipping: pyqtgraph needs the ViewBox for it
            self.plot.addItem(item)
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
            return item
        if isinstance(data, ProfileSeries) and self.kind in ("profile", "map"):
            if self.kind == "map":
                pen.setWidthF(2.0)  # stand out on the image
            item = pg.PlotDataItem(pen=pen, name=name)
            item.setZValue(20)
            self.plot.addItem(item)
            return item
        return None  # images are created in _setup_images

    # ------------------------------------------------------------------ images

    def _setup_images(self, shown: str = "") -> None:
        """Fill the image selector, keeping the image ``shown`` before a reload."""
        self._image_curves = [
            curve for curve in self.curves
            if self.kind == "spacetime" or isinstance(curve.data, MapSeries)
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
        data: MapSeries = curve.data
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
        self._colorbar.getAxis("right").setStyle(
            tickFont=self.plot.getAxis("left").style["tickFont"]
        )

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
                curve.item.setData(x, curve.scaled(y))
                shown.append((curve, t_slice))
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

    def _set_legend_text(self, curve: Curve, t_slice: Optional[float]) -> None:
        text = curve.name if t_slice is None else f"{curve.name}  @ {t_slice:.4g} s"
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
            if data.kind != TRACE:
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
        parts = [self.config.title] if self.config.title else []
        if self.kind in ("profile", "map") and self._title_time is not None:
            parts.append(f"t = {self._title_time:.4g} s")
        text = "  ".join(parts)
        self.title_item.setText(text)
        self.title_item.setVisible(bool(text))

    def _update_labels(self) -> None:
        cfg = self.config
        x_title = y_title = ""
        shown = self._selected_image() if self.kind in ("map", "spacetime") else None
        curves = [shown] if shown is not None else self.curves
        if curves:
            data = curves[0].data
            if self.kind == "trace":
                x_title = "time [s]"
            elif self.kind == "profile":
                x_title = data.x_axis.title()
            elif self.kind == "map":
                x_title, y_title = data.x_axis.title(), data.x2_axis.title()
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
        x_text = cfg.x_label if cfg.x_label is not None else x_title
        y_text = cfg.y_label if cfg.y_label is not None else y_title
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
            text, color, tip = _HINT, "#808080", ""
        elif self.errors and not self.curves:
            text, color = self.errors[0], style.ERROR_COLOR
            tip = "\n".join(self.errors)
        elif self.errors:
            n = len(self.errors)
            text = f"{n} signal{'s' if n > 1 else ''} not shown (hover for details)"
            color, tip = style.ERROR_COLOR, "\n".join(self.errors)
        else:
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
            return
        position = view.mapSceneToView(scene_pos)
        x, y = position.x(), position.y()
        if self._mode == "point" and self.kind in ("trace", "spacetime"):
            buttons = QtWidgets.QApplication.mouseButtons()
            if buttons & QtCore.Qt.MouseButton.LeftButton:
                self.timeRequested.emit(float(x))
        self.hoverMessage.emit(self._describe_position(x, y))

    def _describe_position(self, x: float, y: float) -> str:
        text = f"x = {_format(x)}, y = {_format(y)}"
        if self._image_grid is not None:
            x1, x2, z = self._image_grid
            if len(x1) and len(x2):
                i = int(np.argmin(np.abs(x1 - x)))
                j = int(np.argmin(np.abs(x2 - y)))
                text += f", value = {_format(z[i, j])}"
        return text

    # ------------------------------------------------------------ context menu

    def _install_menu(self) -> None:
        menu = self.plot.getViewBox().menu
        menu.addSeparator()
        actions = [
            ("Setup…", lambda: self.setupRequested.emit(self)),
            ("Autoscale", self.autoscale),
            ("Clear signals", self._clear_signals),
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

    def _clear_signals(self) -> None:
        self.config.signals.clear()
        self.signalsChanged.emit(self)

    # ------------------------------------------------------------- drag & drop

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime.hasFormat(MIME_SIGNAL) or mime.hasText():
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
        self.signalsChanged.emit(self)
