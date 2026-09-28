"""Colors and line styles of plotted curves."""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

# matplotlib "tab10": readable on the white background used by jScope
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
LINE_STYLES = [
    QtCore.Qt.PenStyle.SolidLine,
    QtCore.Qt.PenStyle.DashLine,
    QtCore.Qt.PenStyle.DotLine,
    QtCore.Qt.PenStyle.DashDotLine,
]
MEMBER_PALETTE = PALETTE + [
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
]
"""Colors of the members of a signal with wildcards (one per coil, channel, ...)."""
SYMBOLS = ["x", "+", "o", "t"]
"""Symbols of points (e.g. the magnetic axis), one per source."""
TIME_LINE_COLOR = "#808080"
ERROR_COLOR = "#c62828"
CURRENT_PANEL_COLOR = "#1f77b4"
COLORMAP = "viridis"
CONTOUR_COLORMAP = "turbo"
"""Colors of contour lines: dark at both ends, so that lines show on white."""


def configure_pyqtgraph() -> None:
    """jScope-like look: white background, black axes, antialiased lines."""
    pg.setConfigOptions(background="w", foreground="k", antialias=True)


def curve_pen(signal_index: int, source_index: int, n_signals: int, color=None):
    """Pen for a signal/source combination.

    With a single signal in a panel, sources are told apart by color (like shots in
    jScope). With several signals, color identifies the signal and the line style
    identifies the source.
    """
    if n_signals <= 1:
        default = PALETTE[source_index % len(PALETTE)]
        style = QtCore.Qt.PenStyle.SolidLine
    else:
        default = PALETTE[signal_index % len(PALETTE)]
        style = LINE_STYLES[source_index % len(LINE_STYLES)]
    return pg.mkPen(color or default, width=1.5, style=style)


def member_pen(color_index: int, source_index: int, color=None):
    """Pen of a member of a signal with wildcards: the color tells the members
    apart (unless the signal has a color of its own), the line style the sources."""
    default = MEMBER_PALETTE[color_index % len(MEMBER_PALETTE)]
    style = LINE_STYLES[source_index % len(LINE_STYLES)]
    return pg.mkPen(color or default, width=1.5, style=style)


def source_pen(source_index: int):
    """Black pen with the line style of a source, for legends."""
    return pg.mkPen("k", width=1.5, style=LINE_STYLES[source_index % len(LINE_STYLES)])
