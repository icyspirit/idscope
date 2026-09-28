"""Graphical user interface of idscope.

Needs a Qt binding supported by pyqtgraph (PySide6, PyQt6 or PyQt5).
"""

import sys
from typing import Optional

from idscope.config import ScopeConfig


def create_application(argv=None):
    """Create (or get) the QApplication and configure pyqtgraph."""
    from pyqtgraph.Qt import QtWidgets

    from idscope.gui.style import configure_pyqtgraph

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(list(argv) if argv is not None else sys.argv[:1])
    app.setApplicationName("idscope")
    configure_pyqtgraph()
    return app


def create_window(config: Optional[ScopeConfig] = None, config_path=None,
                  time: Optional[float] = None, modified: bool = False,
                  persist: bool = False):
    """Create a main window (the QApplication must exist).

    Args:
        modified: the panels have changes that are not saved to ``config_path``.
        persist: restore the window's settings, and keep them and the scope as the
            last session (see :mod:`idscope.session`).
    """
    from idscope.gui.mainwindow import MainWindow

    window = MainWindow(config, config_path, modified=modified, persist=persist)
    if time is not None:
        window.set_time(time)
    return window


def run(config: Optional[ScopeConfig] = None, config_path=None,
        time: Optional[float] = None, modified: bool = False) -> int:
    """Open a scope window, which keeps the session, and run the Qt event loop."""
    app = create_application()
    window = create_window(config, config_path, time, modified=modified, persist=True)
    window.show()
    return app.exec()
