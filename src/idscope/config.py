"""The idscope configuration: layout, panels, signals and default sources.

Configurations are stored as YAML. A minimal example::

    title: Shot overview
    sources:
      - A=/data/shot_a.nc
      - ref=imas:hdf5?path=/data/entries/134173/106
    columns:
      - panels:
          - title: Plasma current
            signals:
              - equilibrium/time_slice/global_quantities/ip
          - title: Electron temperature
            signals:
              - path: core_profiles/profiles_1d/electrons/temperature
                scale: 1.0e-3
            y_label: Te [keV]

Like the jScope shot field, the sources can be replaced when opening the file (see
``idscope --help``), so one configuration can be applied to other data.
"""

import difflib
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from idscope.data.sources import SourceSpec, parse_source, unique_labels
from idscope.errors import ConfigError, IdscopeError

KINDS = ("auto", "trace", "profile", "map", "spacetime")
FORMAT_VERSION = 1


@dataclass
class SignalConfig:
    path: str
    """Signal path. Wildcards select several elements, e.g.
    ``pf_active/coil[*]/current/data`` (a curve per coil). Alternatives separated by
    ``|`` are tried in order, e.g. ``core_profiles/profiles_1d/j_phi |
    core_profiles/profiles_1d/j_tor`` for data in Data Dictionary 4 or 3."""
    label: Optional[str] = None
    """Legend label (default: the end of the path)."""
    scale: float = 1.0
    offset: float = 0.0
    """Plotted value is ``value * scale + offset``."""
    x: Optional[str] = None
    """Path of the x values of a profile (default: its Data Dictionary coordinate).
    Alternatives are separated by ``|``, e.g. ``core_profiles/profiles_1d/grid/rho_tor_norm
    | norm(core_profiles/profiles_1d/grid/psi)`` to use psi_n when rho_tor_norm is not
    filled."""
    color: Optional[str] = None
    sources: Optional[List[str]] = None
    """Labels of the sources to plot this signal for (default: all)."""


@dataclass
class PanelConfig:
    title: str = ""
    kind: str = "auto"
    """``auto``, ``trace``, ``profile``, ``map`` or ``spacetime``."""
    signals: List[SignalConfig] = field(default_factory=list)
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    log_x: bool = False
    log_y: bool = False
    legend: bool = True
    equal_aspect: bool = False
    """Same scale on both axes, e.g. for maps in the (R, Z) plane."""
    contours: int = 0
    """Draw maps and time-space images as this many contour lines instead of an image
    (0: image)."""
    height: float = 1.0
    """Relative height within the column."""


@dataclass
class ColumnConfig:
    panels: List[PanelConfig] = field(default_factory=list)
    width: float = 1.0
    """Relative width within the window."""


@dataclass
class ScopeConfig:
    title: str = ""
    sources: List[SourceSpec] = field(default_factory=list)
    columns: List[ColumnConfig] = field(default_factory=list)
    time: Optional[float] = None
    """Initial time of the time cursor."""
    share_x: bool = True
    """Panels that show the same x quantity share their x axis: time across the
    window, profile coordinates within a column. Stacked panels that share their x
    axis are drawn without x tick labels, except for the lowest one."""
    dd_version: Optional[str] = None
    """Data Dictionary version to open the sources with (default: imas-python's)."""

    def panels(self) -> List[PanelConfig]:
        return [panel for column in self.columns for panel in column.panels]


def default_config(columns: int = 2, rows: int = 2) -> ScopeConfig:
    """An empty layout of ``columns`` x ``rows`` panels."""
    return ScopeConfig(
        columns=[ColumnConfig([PanelConfig() for _ in range(rows)]) for _ in range(columns)]
    )


# ---------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------


def _check_keys(data: dict, cls, where: str) -> None:
    allowed = [f.name for f in fields(cls)]
    for key in data:
        if key not in allowed:
            msg = f"{where}: unknown key '{key}'"
            close = difflib.get_close_matches(str(key), allowed, n=1)
            if close:
                msg += f" (did you mean '{close[0]}'?)"
            raise ConfigError(msg)


def _typed(value, types, where: str, name: str):
    types = types if isinstance(types, tuple) else (types,)
    # bool is a subclass of int, but True is not a valid number here
    if not isinstance(value, types) or (isinstance(value, bool) and bool not in types):
        raise ConfigError(f"{where}: '{name}' has an invalid value {value!r}.")
    return value


def _number(data: dict, name: str, default, where: str):
    value = data.get(name, default)
    if value is None:
        return default
    return float(_typed(value, (int, float), where, name))


def _bool(data: dict, name: str, default: bool, where: str) -> bool:
    return _typed(data.get(name, default), bool, where, name)


def _count(data: dict, name: str, where: str) -> int:
    value = data.get(name, 0)
    if value is None:
        return 0
    value = _typed(value, int, where, name)
    if value < 0:
        raise ConfigError(f"{where}: '{name}' must not be negative.")
    return value


def _opt_str(data: dict, name: str, where: str) -> Optional[str]:
    value = data.get(name)
    if value is None:
        return None
    return str(_typed(value, (str, int, float), where, name))


def _range(data: dict, name: str, where: str) -> Optional[Tuple[float, float]]:
    value = data.get(name)
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        raise ConfigError(f"{where}: '{name}' must be [min, max], got {value!r}.")
    return float(value[0]), float(value[1])


def _signal(data, where: str) -> SignalConfig:
    if isinstance(data, str):
        return SignalConfig(data.strip())
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: a signal must be a path or a mapping.")
    _check_keys(data, SignalConfig, where)
    if not data.get("path"):
        raise ConfigError(f"{where}: 'path' is required.")
    sources = data.get("sources")
    if sources is not None:
        if isinstance(sources, str):
            sources = [sources]
        if not isinstance(sources, list):
            raise ConfigError(f"{where}: 'sources' must be a list of source labels.")
        sources = [str(s) for s in sources]
    return SignalConfig(
        path=str(data["path"]).strip(),
        label=_opt_str(data, "label", where),
        scale=_number(data, "scale", 1.0, where),
        offset=_number(data, "offset", 0.0, where),
        x=_opt_str(data, "x", where),
        color=_opt_str(data, "color", where),
        sources=sources,
    )


def _panel(data, where: str) -> PanelConfig:
    if data is None:
        return PanelConfig()
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: a panel must be a mapping.")
    _check_keys(data, PanelConfig, where)
    kind = data.get("kind", "auto")
    if kind not in KINDS:
        raise ConfigError(f"{where}: 'kind' must be one of {', '.join(KINDS)}.")
    signals = data.get("signals") or []
    if not isinstance(signals, list):
        raise ConfigError(f"{where}: 'signals' must be a list.")
    return PanelConfig(
        title=_opt_str(data, "title", where) or "",
        kind=kind,
        signals=[_signal(s, f"{where}.signals[{i}]") for i, s in enumerate(signals)],
        x_label=_opt_str(data, "x_label", where),
        y_label=_opt_str(data, "y_label", where),
        x_range=_range(data, "x_range", where),
        y_range=_range(data, "y_range", where),
        log_x=_bool(data, "log_x", False, where),
        log_y=_bool(data, "log_y", False, where),
        legend=_bool(data, "legend", True, where),
        equal_aspect=_bool(data, "equal_aspect", False, where),
        contours=_count(data, "contours", where),
        height=_number(data, "height", 1.0, where),
    )


def _column(data, where: str) -> ColumnConfig:
    if isinstance(data, list):  # a column can be given as a bare list of panels
        data = {"panels": data}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: a column must be a mapping with 'panels'.")
    _check_keys(data, ColumnConfig, where)
    panels = data.get("panels") or []
    if not isinstance(panels, list):
        raise ConfigError(f"{where}: 'panels' must be a list.")
    return ColumnConfig(
        panels=[_panel(p, f"{where}.panels[{i}]") for i, p in enumerate(panels)],
        width=_number(data, "width", 1.0, where),
    )


def _source(data, where: str, base_dir: Optional[Path]) -> SourceSpec:
    """Parse a source; relative file paths refer to the configuration directory."""
    if isinstance(data, str):
        return parse_source(data, base_dir)
    if isinstance(data, dict):
        unknown = set(data) - {"uri", "label"}
        if unknown or "uri" not in data:
            raise ConfigError(f"{where}: a source needs 'uri' and optionally 'label'.")
        spec = parse_source(str(data["uri"]), base_dir)
        if data.get("label"):
            spec = SourceSpec(spec.uri, str(data["label"]))
        return spec
    raise ConfigError(f"{where}: a source must be a string or a mapping.")


def config_from_dict(data: Any, base_dir: Optional[Path] = None) -> ScopeConfig:
    """Build a configuration from parsed YAML."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("The configuration must be a mapping.")
    # "idscope" holds the format version, the other keys are ScopeConfig fields
    _check_keys({k: v for k, v in data.items() if k != "idscope"}, ScopeConfig, "config")
    version = data.get("idscope", FORMAT_VERSION)
    if version != FORMAT_VERSION:
        raise ConfigError(
            f"Unsupported configuration format version {version!r} "
            f"(this idscope reads version {FORMAT_VERSION})."
        )
    sources = data.get("sources") or []
    if isinstance(sources, str):
        sources = [s for s in sources.split(",") if s.strip()]
    if not isinstance(sources, list):
        raise ConfigError("config: 'sources' must be a list.")
    columns = data.get("columns") or []
    if not isinstance(columns, list):
        raise ConfigError("config: 'columns' must be a list.")
    time = data.get("time")
    if time is not None:
        time = float(_typed(time, (int, float), "config", "time"))
    try:
        specs = [_source(s, f"sources[{i}]", base_dir) for i, s in enumerate(sources)]
    except IdscopeError as exc:
        raise ConfigError(str(exc)) from None
    return ScopeConfig(
        title=_opt_str(data, "title", "config") or "",
        sources=unique_labels(specs),
        columns=[_column(c, f"columns[{i}]") for i, c in enumerate(columns)],
        time=time,
        share_x=_bool(data, "share_x", True, "config"),
        dd_version=_opt_str(data, "dd_version", "config"),
    )


def load_config(path) -> ScopeConfig:
    """Read a YAML configuration file."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(f"Cannot read {path}: {exc.strerror}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from None
    try:
        return config_from_dict(data, path.parent.resolve())
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


# ---------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------


def _default_of(f):
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return MISSING


def _without_defaults(obj, skip=()) -> Dict[str, Any]:
    result = {}
    for f in fields(obj):
        if f.name in skip:
            continue
        value = getattr(obj, f.name)
        if value == _default_of(f):
            continue
        if isinstance(value, tuple):
            value = list(value)
        result[f.name] = value
    return result


def _signal_to_data(signal: SignalConfig):
    data = _without_defaults(signal)
    if list(data) == ["path"]:
        return signal.path  # short form
    return data


def config_to_dict(config: ScopeConfig, base_dir: Optional[Path] = None) -> dict:
    """Convert a configuration to plain data, leaving out default values."""
    data: Dict[str, Any] = {"idscope": FORMAT_VERSION}
    if config.title:
        data["title"] = config.title
    if config.sources:
        # Files next to the configuration are stored with a relative path
        data["sources"] = [spec.format(base_dir) for spec in config.sources]
    if config.time is not None:
        data["time"] = config.time
    if not config.share_x:
        data["share_x"] = False
    if config.dd_version:
        data["dd_version"] = config.dd_version
    columns = []
    for column in config.columns:
        col: Dict[str, Any] = {}
        if column.width != 1.0:
            col["width"] = column.width
        panels = []
        for panel in column.panels:
            pdata = _without_defaults(panel, skip=("signals",))
            if panel.signals:
                pdata["signals"] = [_signal_to_data(s) for s in panel.signals]
            panels.append(pdata)
        col["panels"] = panels
        columns.append(col)
    data["columns"] = columns
    return data


def save_config(config: ScopeConfig, path) -> None:
    """Write a configuration as YAML (relative paths for files next to it)."""
    path = Path(path)
    data = config_to_dict(config, path.parent)
    text = "# idscope configuration\n" + yaml.safe_dump(
        data, sort_keys=False, default_flow_style=None, allow_unicode=True, width=100
    )
    path.write_text(text)
