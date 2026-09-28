"""Panel setup dialog (the "wave setup" of jScope)."""

from dataclasses import fields
from typing import List, Optional

from pyqtgraph.Qt import QtCore, QtWidgets

from idscope.config import KINDS, PanelConfig, SignalConfig
from idscope.data import parse_alternatives
from idscope.errors import IdscopeError

COLUMNS = ["Path", "Label", "Scale", "Offset", "X path", "Color", "Sources"]
_PATH, _LABEL, _SCALE, _OFFSET, _X, _COLOR, _SOURCES = range(len(COLUMNS))


class SetupDialog(QtWidgets.QDialog):
    """Edit the signals and appearance of a panel.

    Changes are written into the :class:`PanelConfig` on OK/Apply, after which
    :attr:`applied` is emitted.
    """

    applied = QtCore.Signal()

    def __init__(self, config: PanelConfig, source_labels: List[str], parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle(f"Setup: {config.title or 'panel'}")
        self.resize(900, 460)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QGridLayout()
        self.title = QtWidgets.QLineEdit(config.title)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(KINDS)
        self.kind.setCurrentText(config.kind)
        self.kind.setToolTip(
            "auto: from the data; trace: y(t); profile: y(x) at the selected time;\n"
            "map: z(x1, x2) at the selected time; spacetime: a profile as image z(t, x)"
        )
        self.x_label = self._optional_text(config.x_label)
        self.y_label = self._optional_text(config.y_label)
        self.x_min, self.x_max = self._range_edits(config.x_range)
        self.y_min, self.y_max = self._range_edits(config.y_range)
        self.log_x = QtWidgets.QCheckBox("log x")
        self.log_x.setChecked(config.log_x)
        self.log_y = QtWidgets.QCheckBox("log y")
        self.log_y.setChecked(config.log_y)
        self.legend = QtWidgets.QCheckBox("legend")
        self.legend.setChecked(config.legend)
        self.equal_aspect = QtWidgets.QCheckBox("equal aspect")
        self.equal_aspect.setChecked(config.equal_aspect)
        self.contours = QtWidgets.QSpinBox()
        self.contours.setRange(0, 100)
        self.contours.setValue(config.contours)
        self.contours.setToolTip("Number of contour lines on maps and time-space images")

        form.addWidget(QtWidgets.QLabel("Title"), 0, 0)
        form.addWidget(self.title, 0, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("Kind"), 0, 4)
        form.addWidget(self.kind, 0, 5)
        form.addWidget(QtWidgets.QLabel("X label"), 1, 0)
        form.addWidget(self.x_label, 1, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("X range"), 1, 4)
        form.addLayout(self._pair(self.x_min, self.x_max), 1, 5)
        form.addWidget(QtWidgets.QLabel("Y label"), 2, 0)
        form.addWidget(self.y_label, 2, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("Y range"), 2, 4)
        form.addLayout(self._pair(self.y_min, self.y_max), 2, 5)
        checks = QtWidgets.QHBoxLayout()
        for box in (self.log_x, self.log_y, self.legend, self.equal_aspect):
            checks.addWidget(box)
        checks.addSpacing(12)
        checks.addWidget(QtWidgets.QLabel("contours"))
        checks.addWidget(self.contours)
        checks.addStretch(1)
        form.addLayout(checks, 3, 1, 1, 5)
        layout.addLayout(form)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(_PATH, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        sources_tip = "Source labels, comma separated (empty: all sources)"
        if source_labels:
            sources_tip += f"\nAvailable: {', '.join(source_labels)}"
        tips = {
            _PATH: "e.g. core_profiles/profiles_1d/electrons/temperature\n"
                   "Alternatives, tried in order: path_a | path_b",
            _SCALE: "plotted value = value × scale + offset",
            _OFFSET: "plotted value = value × scale + offset",
            _X: "x values of a profile (default: its coordinate), "
                "e.g. core_profiles/profiles_1d/grid/psi",
            _COLOR: "e.g. #d62728 or red (empty: automatic)",
            _SOURCES: sources_tip,
        }
        for column, tip in tips.items():
            self.table.horizontalHeaderItem(column).setToolTip(tip)
        for signal in config.signals:
            self._add_row(signal)
        layout.addWidget(self.table, stretch=1)

        buttons = QtWidgets.QHBoxLayout()
        for text, slot in (
            ("Add", lambda: self._add_row(SignalConfig(""), edit=True)),
            ("Remove", self._remove_rows),
            ("Up", lambda: self._move(-1)),
            ("Down", lambda: self._move(1)),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        box.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.apply
        )
        buttons.addWidget(box)
        layout.addLayout(buttons)

    # --------------------------------------------------------------- widgets

    @staticmethod
    def _optional_text(value: Optional[str]) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(value or "")
        edit.setPlaceholderText("automatic")
        return edit

    @staticmethod
    def _range_edits(value):
        edits = []
        for i in range(2):
            edit = QtWidgets.QLineEdit("" if value is None else f"{value[i]:g}")
            edit.setPlaceholderText("auto")
            edit.setMaximumWidth(90)
            edits.append(edit)
        return edits

    @staticmethod
    def _pair(first, second):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(first)
        layout.addWidget(QtWidgets.QLabel("–"))
        layout.addWidget(second)
        return layout

    def _add_row(self, signal: SignalConfig, edit: bool = False) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            signal.path,
            signal.label or "",
            f"{signal.scale:g}",
            f"{signal.offset:g}",
            signal.x or "",
            signal.color or "",
            ", ".join(signal.sources) if signal.sources is not None else "",
        ]
        for column, value in enumerate(values):
            self.table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        if edit:
            self.table.setCurrentCell(row, _PATH)
            self.table.editItem(self.table.item(row, _PATH))

    def _remove_rows(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _move(self, delta: int) -> None:
        row = self.table.currentRow()
        target = row + delta
        if row < 0 or not 0 <= target < self.table.rowCount():
            return
        for column in range(self.table.columnCount()):
            a, b = self.table.takeItem(row, column), self.table.takeItem(target, column)
            self.table.setItem(row, column, b)
            self.table.setItem(target, column, a)
        self.table.setCurrentCell(target, self.table.currentColumn())

    # ---------------------------------------------------------------- values

    def _cell(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        return item.text().strip() if item is not None else ""

    @staticmethod
    def _float(text: str, default, what: str):
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            raise IdscopeError(f"{what}: '{text}' is not a number.") from None

    def _range(self, low: QtWidgets.QLineEdit, high: QtWidgets.QLineEdit, what: str):
        lo = self._float(low.text().strip(), None, what)
        hi = self._float(high.text().strip(), None, what)
        if lo is None and hi is None:
            return None
        if lo is None or hi is None or lo >= hi:
            raise IdscopeError(f"{what}: give both limits, with min < max.")
        return (lo, hi)

    def read_config(self) -> PanelConfig:
        """The configuration as currently entered in the dialog.

        Raises:
            IdscopeError: when an entry is invalid.
        """
        signals = []
        for row in range(self.table.rowCount()):
            path = self._cell(row, _PATH)
            if not path:
                continue
            where = f"Row {row + 1}"
            try:
                parse_alternatives(path)
                x = self._cell(row, _X) or None
                if x:
                    parse_alternatives(x)
            except IdscopeError as exc:
                raise IdscopeError(f"{where}: {exc}") from None
            sources = self._cell(row, _SOURCES)
            signals.append(SignalConfig(
                path=path,
                label=self._cell(row, _LABEL) or None,
                scale=self._float(self._cell(row, _SCALE), 1.0, f"{where}, scale"),
                offset=self._float(self._cell(row, _OFFSET), 0.0, f"{where}, offset"),
                x=x,
                color=self._cell(row, _COLOR) or None,
                sources=[s.strip() for s in sources.split(",") if s.strip()] or None,
            ))
        return PanelConfig(
            title=self.title.text().strip(),
            kind=self.kind.currentText(),
            signals=signals,
            x_label=self.x_label.text() or None,
            y_label=self.y_label.text() or None,
            x_range=self._range(self.x_min, self.x_max, "X range"),
            y_range=self._range(self.y_min, self.y_max, "Y range"),
            log_x=self.log_x.isChecked(),
            log_y=self.log_y.isChecked(),
            legend=self.legend.isChecked(),
            equal_aspect=self.equal_aspect.isChecked(),
            contours=self.contours.value(),
            height=self.config.height,
        )

    def apply(self) -> bool:
        try:
            new = self.read_config()
        except IdscopeError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid setup", str(exc))
            return False
        for f in fields(PanelConfig):
            setattr(self.config, f.name, getattr(new, f.name))
        self.applied.emit()
        return True

    def accept(self) -> None:
        if self.apply():
            super().accept()
