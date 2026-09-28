"""The IDS browser: lists the filled quantities of a source (like jScope's
"Browse signals")."""

import logging
from typing import Dict, List, Optional

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from idscope.data.browse import BrowseItem, list_signals
from idscope.data.sources import DataSource
from idscope.errors import IdscopeError
from idscope.gui.panel import MIME_SIGNAL

logger = logging.getLogger(__name__)

_PATH_ROLE = QtCore.Qt.ItemDataRole.UserRole
_KIND_COLORS = {"trace": "#1f77b4", "profile": "#2ca02c", "map": "#9467bd",
                "shape": "#8c564b"}


class _SignalTree(QtWidgets.QTreeWidget):
    """Tree whose leaves can be dragged onto panels."""

    def mimeData(self, items):
        paths = [item.data(0, _PATH_ROLE) for item in items if item.data(0, _PATH_ROLE)]
        mime = QtCore.QMimeData()
        text = "\n".join(paths)
        mime.setData(MIME_SIGNAL, QtCore.QByteArray(text.encode()))
        mime.setText(text)
        return mime

    def mimeTypes(self):
        return [MIME_SIGNAL, "text/plain"]


class BrowserWidget(QtWidgets.QWidget):
    """Browse the IDSs of a source and add their quantities to panels."""

    addRequested = QtCore.Signal(str, bool)
    """``(signal_path, new_panel)``: add a signal to the current or a new panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sources: List[DataSource] = []
        self._items: List[BrowseItem] = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.currentIndexChanged.connect(self._on_source)
        self.ids_combo = QtWidgets.QComboBox()
        self.ids_combo.currentIndexChanged.connect(self._on_ids)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter, e.g. temperature")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        form = QtWidgets.QFormLayout()
        form.addRow("Source", self.source_combo)
        form.addRow("IDS", self.ids_combo)
        layout.addLayout(form)
        layout.addWidget(self.filter)
        self.tree = _SignalTree()
        self.tree.setHeaderLabels(["Quantity", "Kind", "Units"])
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)
        self.tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.tree.setUniformRowHeights(True)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.setIndentation(14)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        self.setMinimumWidth(280)
        layout.addWidget(self.tree, stretch=1)
        self.status = QtWidgets.QLabel(
            "Double-click or drag a quantity onto a panel."
        )
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #606060; font-size: 8pt;")
        layout.addWidget(self.status)

    # ----------------------------------------------------------------- sources

    def set_sources(self, sources: List[DataSource]) -> None:
        """Show the given sources, keeping the selection when possible."""
        previous = self.source_combo.currentText()
        self._sources = list(sources)
        with QtCore.QSignalBlocker(self.source_combo):
            self.source_combo.clear()
            for source in self._sources:
                self.source_combo.addItem(source.label)
                self.source_combo.setItemData(
                    self.source_combo.count() - 1, source.uri,
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )
            index = self.source_combo.findText(previous)
            self.source_combo.setCurrentIndex(max(index, 0))
        self._on_source()

    def current_source(self) -> Optional[DataSource]:
        index = self.source_combo.currentIndex()
        return self._sources[index] if 0 <= index < len(self._sources) else None

    def _on_source(self, *_) -> None:
        previous = self.ids_combo.currentText()
        with QtCore.QSignalBlocker(self.ids_combo):
            self.ids_combo.clear()
        self.tree.clear()
        source = self.current_source()
        if source is None:
            self.status.setText("No data source. Enter one in the Sources field.")
            return
        try:
            available = source.available_ids()
        except IdscopeError as exc:
            self.status.setText(str(exc))
            return
        with QtCore.QSignalBlocker(self.ids_combo):
            for name, occurrence in available:
                text = name if occurrence == 0 else f"{name}:{occurrence}"
                self.ids_combo.addItem(text, (name, occurrence))
            index = self.ids_combo.findText(previous)
            self.ids_combo.setCurrentIndex(max(index, 0))
        if not available:
            self.status.setText(f"No IDS found in '{source.label}'.")
        self._on_ids()

    def _on_ids(self, *_) -> None:
        self.tree.clear()
        self._items = []
        source = self.current_source()
        selection = self.ids_combo.currentData()
        if source is None or selection is None:
            return
        name, occurrence = selection
        try:
            self._items = list_signals(source, name, occurrence)
        except IdscopeError as exc:
            self.status.setText(str(exc))
            return
        self._fill_tree(name, self._items)
        self.status.setText(
            f"{len(self._items)} quantities. Double-click or drag onto a panel."
        )
        self._apply_filter(self.filter.text())

    def _fill_tree(self, ids_name: str, items: List[BrowseItem]) -> None:
        self.tree.setUpdatesEnabled(False)
        nodes: Dict[str, QtWidgets.QTreeWidgetItem] = {}
        for item in items:
            parts = item.dd_path.split("/")
            parent = self.tree.invisibleRootItem()
            for depth in range(len(parts) - 1):
                key = "/".join(parts[: depth + 1])
                node = nodes.get(key)
                if node is None:
                    node = QtWidgets.QTreeWidgetItem(parent, [parts[depth]])
                    node.setFlags(node.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled)
                    nodes[key] = node
                parent = node
            # A shape (e.g. a coil geometry) is also the parent of its quantities
            leaf = nodes.get(item.dd_path)
            if leaf is None:
                leaf = QtWidgets.QTreeWidgetItem(parent, [parts[-1]])
                nodes[item.dd_path] = leaf
            leaf.setText(1, item.kind)
            leaf.setText(2, item.units)
            leaf.setFlags(leaf.flags() | QtCore.Qt.ItemFlag.ItemIsDragEnabled)
            leaf.setData(0, _PATH_ROLE, item.signal_path)
            doc = item.documentation
            leaf.setToolTip(0, f"{item.signal_path}\n\n{doc}" if doc else item.signal_path)
            color = _KIND_COLORS.get(item.kind)
            if color:
                leaf.setForeground(1, QtGui.QBrush(QtGui.QColor(color)))
        self.tree.setUpdatesEnabled(True)

    def _apply_filter(self, text: str) -> None:
        text = text.strip().lower()

        def visit(node: QtWidgets.QTreeWidgetItem, prefix: str) -> bool:
            path = f"{prefix}/{node.text(0)}" if prefix else node.text(0)
            visible = not text or (
                bool(node.data(0, _PATH_ROLE)) and text in path.lower()
            )
            for i in range(node.childCount()):
                visible |= visit(node.child(i), path)
            node.setHidden(not visible)
            if text and node.childCount():
                node.setExpanded(visible)
            return visible

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            visit(root.child(i), "")

    # ---------------------------------------------------------------- actions

    def selected_paths(self) -> List[str]:
        return [
            item.data(0, _PATH_ROLE)
            for item in self.tree.selectedItems()
            if item.data(0, _PATH_ROLE)
        ]

    def _on_double_click(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        path = item.data(0, _PATH_ROLE)
        if path:
            self.addRequested.emit(path, False)

    def _context_menu(self, position) -> None:
        paths = self.selected_paths()
        if not paths:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Add to current panel").triggered.connect(
            lambda: [self.addRequested.emit(p, False) for p in paths]
        )
        menu.addAction("Add to a new panel").triggered.connect(
            lambda: self._add_to_new_panel(paths)
        )
        menu.addAction("Copy path").triggered.connect(
            lambda: QtWidgets.QApplication.clipboard().setText("\n".join(paths))
        )
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _add_to_new_panel(self, paths: List[str]) -> None:
        self.addRequested.emit(paths[0], True)
        for path in paths[1:]:
            self.addRequested.emit(path, False)
