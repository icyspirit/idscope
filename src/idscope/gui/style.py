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
TIME_LINE_COLOR = "#808080"
ERROR_COLOR = "#c62828"
CURRENT_PANEL_COLOR = "#1f77b4"
COLORMAP = "viridis"


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
