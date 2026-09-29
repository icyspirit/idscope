"""The time bar: the time cursor shared by all panels."""

from typing import Optional

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets


class TimeBar(QtWidgets.QWidget):
    """Slider over the time slices of the loaded data, plus an exact time field.

    The slider steps through ``grid`` (the time slices of the profiles and maps);
    the spin box accepts any time. :attr:`timeChanged` is emitted on user changes.
    """

    timeChanged = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = np.empty(0)
        self._time: Optional[float] = None
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(QtWidgets.QLabel("Time"))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(4)
        self.spin.setRange(-1e9, 1e9)
        self.spin.setSuffix(" s")
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(110)
        self.spin.valueChanged.connect(self._on_spin)
        layout.addWidget(self.spin)
        self.prev_button = QtWidgets.QToolButton()
        self.prev_button.setArrowType(QtCore.Qt.ArrowType.LeftArrow)
        self.prev_button.setToolTip("Previous time slice")
        self.prev_button.clicked.connect(lambda: self.step(-1))
        layout.addWidget(self.prev_button)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider, stretch=1)
        self.next_button = QtWidgets.QToolButton()
        self.next_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.next_button.setToolTip("Next time slice")
        self.next_button.clicked.connect(lambda: self.step(1))
        layout.addWidget(self.next_button)
        self.info = QtWidgets.QLabel()
        self.info.setMinimumWidth(140)
        layout.addWidget(self.info)
        self._update_enabled()

    @property
    def time(self) -> Optional[float]:
        return self._time

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def set_grid(self, grid: np.ndarray) -> None:
        """Set the time slices that the slider steps through."""
        grid = np.unique(np.asarray(grid, dtype=float))
        self._grid = grid[np.isfinite(grid)]
        with QtCore.QSignalBlocker(self.slider):
            self.slider.setRange(0, max(len(self._grid) - 1, 0))
        if len(self._grid) > 1:
            self.info.setText(
                f"{len(self._grid)} slices, {self._grid[0]:.4g} … {self._grid[-1]:.4g} s"
            )
        elif len(self._grid):
            self.info.setText(f"1 slice, t = {self._grid[0]:.4g} s")
        else:
            self.info.setText("no time slices")
        self._update_enabled()
        if self._time is not None:
            self._show(self._time)

    def set_time(self, t: Optional[float]) -> None:
        """Set the time without emitting :attr:`timeChanged`."""
        self._time = t
        self._show(t)

    def step(self, delta: int) -> None:
        if not len(self._grid):
            return
        index = self._nearest(self._time) if self._time is not None else 0
        # If the time is between slices, a step goes to the neighbouring slice
        if self._time is not None and self._grid[index] != self._time:
            if (delta > 0) == (self._grid[index] > self._time):
                delta -= int(np.sign(delta))
        index = int(np.clip(index + delta, 0, len(self._grid) - 1))
        self._emit(float(self._grid[index]))

    def go_to(self, index: int) -> None:
        """Go to time slice ``index`` (e.g. -1: the last one)."""
        if len(self._grid):
            self._emit(float(self._grid[index]))

    def _nearest(self, t: float) -> int:
        return int(np.argmin(np.abs(self._grid - t)))

    def _show(self, t: Optional[float]) -> None:
        with QtCore.QSignalBlocker(self.spin):
            if t is not None:
                self.spin.setValue(t)
        if t is not None and len(self._grid):
            with QtCore.QSignalBlocker(self.slider):
                self.slider.setValue(self._nearest(t))

    def _emit(self, t: float) -> None:
        self.set_time(t)
        self.timeChanged.emit(t)

    def _on_slider(self, index: int) -> None:
        if len(self._grid):
            self._emit(float(self._grid[index]))

    def _on_spin(self, value: float) -> None:
        self._emit(float(value))

    def _update_enabled(self) -> None:
        steps = len(self._grid) > 1  # nothing to step through in a single slice
        for widget in (self.slider, self.prev_button, self.next_button):
            widget.setEnabled(steps)
