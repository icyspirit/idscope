"""The cells of an unstructured grid, filled with the colors of their values."""

from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from idscope.data.extract import Mesh

_COLORS = 256
"""Colors of the color map that the values are drawn with."""


class MeshItem(pg.GraphicsObject):
    """The cells of a :class:`~idscope.data.extract.Mesh`, each filled with the color
    of its value (cells without a value are left empty).

    A :class:`pyqtgraph.ColorBarItem` controls it as it does an image
    (``setLevels``, ``setColorMap``). The cells of a color are drawn as one path,
    and the drawing is kept until the values, the levels or the colors change.
    """

    def __init__(self, colormap: pg.ColorMap):
        super().__init__()
        self._colormap = colormap
        self._levels: Tuple[float, float] = (0.0, 1.0)
        self._mesh: Optional[Mesh] = None
        self._values = np.empty(0)
        self._outlines: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._picture: Optional[QtGui.QPicture] = None
        self._bounds = QtCore.QRectF()

    # ------------------------------------------------------ as an image item

    def setData(self, mesh: Mesh, values: np.ndarray,
                levels: Optional[Tuple[float, float]] = None) -> None:
        if mesh is not self._mesh:
            self._mesh = mesh
            self._outlines = _outlines(mesh)
            finite = mesh.points[np.isfinite(mesh.points).all(axis=1)]
            self.prepareGeometryChange()
            self._bounds = QtCore.QRectF() if not len(finite) else QtCore.QRectF(
                QtCore.QPointF(*finite.min(axis=0)), QtCore.QPointF(*finite.max(axis=0)))
        self._values = np.asarray(values, dtype=float)
        if levels is not None:
            self._levels = (float(levels[0]), float(levels[1]))
        self._redraw()

    def setLevels(self, levels, update: bool = True) -> None:
        self._levels = (float(levels[0]), float(levels[1]))
        self._redraw()

    def getLevels(self) -> Tuple[float, float]:
        return self._levels

    def setColorMap(self, colormap: pg.ColorMap) -> None:
        self._colormap = colormap
        self._redraw()

    def getColorMap(self) -> pg.ColorMap:
        return self._colormap

    def value_at(self, x: float, y: float) -> Optional[float]:
        """The value of the cell at (x, y), or None outside of the cells."""
        if self._mesh is None:
            return None
        cell = self._mesh.cell_at(x, y)
        if cell is None or cell >= len(self._values):
            return None
        return float(self._values[cell])

    # ----------------------------------------------------------- drawing

    def boundingRect(self) -> QtCore.QRectF:
        return self._bounds

    def paint(self, painter, *args) -> None:
        if self._picture is None:
            self._picture = self._render()
        self._picture.play(painter)

    def _redraw(self) -> None:
        self._picture = None
        self.update()

    def _render(self) -> QtGui.QPicture:
        picture = QtGui.QPicture()
        if self._outlines is None or not len(self._values):
            return picture
        x, y, cell_of = self._outlines
        low, high = self._levels
        span = high - low if high > low else 1.0
        with np.errstate(invalid="ignore"):
            color = np.floor((self._values - low) / span * _COLORS)
        # Beyond the levels, the color of the end; no value, not drawn
        color = np.clip(np.nan_to_num(color, nan=0.0), 0, _COLORS - 1).astype(int)
        color[~np.isfinite(self._values)] = -1
        # The points of the cells of each color, in order (cell by cell)
        of_point = color[cell_of]
        order = np.argsort(of_point, kind="stable")
        of_point = of_point[order]
        table = self._colormap.getLookupTable(nPts=_COLORS, alpha=True)
        painter = QtGui.QPainter(picture)
        for index in np.unique(of_point):
            if index < 0:
                continue
            start, stop = np.searchsorted(of_point, [index, index + 1])
            chosen = order[start:stop]
            path = pg.arrayToQPath(x[chosen], y[chosen], connect="finite")
            qcolor = QtGui.QColor(*[int(v) for v in table[index]])
            painter.fillPath(path, pg.mkBrush(qcolor))
            # Over the seams that filling leaves between neighbouring cells
            pen = pg.mkPen(qcolor, width=0)
            pen.setCosmetic(True)
            painter.strokePath(path, pen)
        painter.end()
        return picture


def _outlines(mesh: Mesh):
    """The outline of every cell, closed and followed by a NaN: ``(x, y, the cell
    of every point)``."""
    parts, owners = [], []
    for index, cell in enumerate(mesh.cells):
        if not cell.size:
            continue
        corners = mesh.points[cell]
        parts.append(np.vstack([corners, corners[:1], [[np.nan, np.nan]]]))
        owners.append(np.full(len(cell) + 2, index))
    if not parts:
        return None
    points = np.vstack(parts)
    return points[:, 0], points[:, 1], np.concatenate(owners)
