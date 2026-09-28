"""The last session and where idscope keeps its settings.

When idscope starts without a configuration file or a template, it shows the scope of
the last session: its panels, sources, sizes and time. The window writes it on
closing, and a moment after each change, to ``session.yaml`` in :func:`config_dir`.
The file is a configuration, which can also be opened as one, with a ``session`` key
that holds the configuration file the scope came from and whether it has changes
that are not saved to that file.
"""

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple, Optional

import yaml

from idscope.config import ScopeConfig, config_from_dict, config_to_dict, dump_yaml
from idscope.errors import ConfigError

SESSION_FILE = "session.yaml"
SETTINGS_FILE = "settings.ini"
"""Settings of the window (size, docks, mode, recent files), written by Qt."""

_HEADER = (
    "# idscope session: the scope shown when idscope starts without a configuration\n"
)


def config_dir() -> Path:
    """The directory of idscope's settings and last session.

    ``$IDSCOPE_CONFIG_DIR`` if set, otherwise ``idscope`` in ``%APPDATA%`` on Windows
    and in ``$XDG_CONFIG_HOME`` (by default ``~/.config``) elsewhere.
    """
    directory = os.environ.get("IDSCOPE_CONFIG_DIR")
    if directory:
        return Path(directory).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "idscope"


def session_path() -> Path:
    return config_dir() / SESSION_FILE


class Session(NamedTuple):
    config: ScopeConfig
    config_path: Optional[Path]
    """The configuration file the scope was opened from or saved to."""
    modified: bool
    """The panels have changes that are not saved to ``config_path``."""


def load_session(path=None) -> Optional[Session]:
    """The last session, or None if there is none.

    Raises:
        ConfigError: the session file cannot be read.
    """
    path = Path(path) if path is not None else session_path()
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            return None
        config = config_from_dict(data, path.parent.resolve())
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ConfigError) as exc:
        raise ConfigError(f"Cannot read the last session {path}: {exc}") from None
    info = data.get("session")
    info = info if isinstance(info, dict) else {}
    config_path = info.get("file")
    # Forget a configuration file that was moved or deleted since: Save then asks
    # where to write, instead of creating it again
    if not isinstance(config_path, str) or not Path(config_path).is_file():
        config_path = None
    return Session(
        config, Path(config_path) if config_path else None, info.get("modified") is True
    )


def save_session(config: ScopeConfig, config_path=None, modified: bool = False,
                 path=None) -> Path:
    """Write the session file, replacing it at once so that it is never half written.

    Raises:
        OSError: the file cannot be written.
    """
    path = Path(path) if path is not None else session_path()
    data = config_to_dict(config, path.parent)
    info = {}
    if config_path is not None:
        info["file"] = os.path.abspath(config_path)
    if modified:
        info["modified"] = True
    if info:
        data = {"idscope": data.pop("idscope"), "session": info, **data}
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".session-", suffix=".yaml")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(_HEADER + dump_yaml(data))
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return path
