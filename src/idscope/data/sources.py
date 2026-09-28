"""Data sources: IMAS URIs and IDS netCDF files.

A source is written as a URI (``imas:hdf5?path=/data/entry``), a path to an IDS
netCDF file (``shot.nc``) or a path to an HDF5 data entry directory. It can be given
a short label with ``label=``, e.g. ``ref=imas:hdf5?path=/data/entry``. Several
sources are separated by commas, like the shot field of jScope.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import imas

from idscope.data.extract import SignalData, apply_x, extract_signal
from idscope.data.paths import SignalPath
from idscope.errors import IdscopeError, NoDataError, PathError, SourceError

logger = logging.getLogger(__name__)

_ALIAS = re.compile(r"([A-Za-z_][\w.\-]*)\s*=\s*(.+)", re.DOTALL)


@dataclass(frozen=True)
class SourceSpec:
    """A data source as written by the user."""

    uri: str
    """IMAS URI or path to a netCDF file."""
    label: str
    """Short name shown in legends."""

    def format(self, base_dir: Optional[Path] = None) -> str:
        """Format for a source list, omitting the label when it is the default.

        Files inside ``base_dir`` are written with a relative path.
        """
        uri = self.uri
        if base_dir is not None and not uri.startswith("imas:"):
            try:
                uri = str(Path(uri).resolve().relative_to(Path(base_dir).resolve()))
            except ValueError:
                pass
        if self.label == default_label(self.uri):
            return uri
        return f"{self.label}={uri}"


def normalize_uri(text: str) -> str:
    """Turn a user-provided source into something ``imas.DBEntry`` can open.

    - IMAS URIs (``imas:...``) are kept as they are.
    - netCDF file paths get ``~`` expanded.
    - A directory holding an HDF5 data entry (``master.h5``) becomes an
      ``imas:hdf5?path=...`` URI.
    """
    spec = text.strip()
    if not spec:
        raise SourceError("Empty data source.")
    if spec.startswith("imas:"):
        return spec
    path = Path(spec).expanduser()
    if path.is_dir() and (path / "master.h5").is_file():
        return f"imas:hdf5?path={path.resolve()}"
    if spec.endswith(".nc"):
        return str(path)
    return spec


def default_label(uri: str) -> str:
    """A short label for a source, e.g. the file name or the pulse/run numbers."""
    if uri.startswith("imas:"):
        query = uri.split("?", 1)[1] if "?" in uri else ""
        params = {}
        for item in re.split("[;&]", query):
            if "=" in item:
                key, value = item.split("=", 1)
                params[key.strip()] = value.strip()
        if params.get("path"):
            return Path(params["path"].rstrip("/")).name or params["path"]
        if "pulse" in params:
            return f"{params['pulse']}/{params.get('run', '0')}"
        return uri
    path = Path(uri)
    return path.stem if path.suffix == ".nc" else (path.name or uri)


def parse_source(text: str, base_dir: Optional[Path] = None) -> SourceSpec:
    """Parse one source, with an optional ``label=`` prefix.

    Args:
        base_dir: directory that relative file paths refer to (default: the
            current directory).
    """
    text = text.strip()
    match = _ALIAS.fullmatch(text)
    if match:
        label, spec = match[1], match[2].strip()
    else:
        label, spec = None, text
    if base_dir is not None and spec and not spec.startswith("imas:"):
        path = Path(spec).expanduser()
        if not path.is_absolute():
            spec = str(Path(base_dir) / path)
    uri = normalize_uri(spec)
    return SourceSpec(uri, label or default_label(uri))


def parse_source_list(text: str, base_dir: Optional[Path] = None) -> List[SourceSpec]:
    """Parse a comma-separated list of sources, making labels unique."""
    specs = [parse_source(item, base_dir) for item in text.split(",") if item.strip()]
    return unique_labels(specs)


def unique_labels(specs: List[SourceSpec]) -> List[SourceSpec]:
    """Append ``#2``, ``#3``, ... to repeated labels."""
    seen: Dict[str, int] = {}
    result = []
    for spec in specs:
        count = seen.get(spec.label, 0) + 1
        seen[spec.label] = count
        if count > 1:
            spec = SourceSpec(spec.uri, f"{spec.label}#{count}")
        result.append(spec)
    return result


def format_source_list(specs: List[SourceSpec], base_dir: Optional[Path] = None) -> str:
    return ", ".join(spec.format(base_dir) for spec in specs)


class DataSource:
    """An opened data source with caches for IDSs and extracted signals.

    IDSs are loaded lazily: only the nodes that are plotted are read. The data entry
    therefore stays open until :meth:`close` is called.
    """

    def __init__(self, spec: SourceSpec, dd_version: Optional[str] = None):
        self.spec = spec
        self.dd_version = dd_version
        self._entry: Optional[imas.DBEntry] = None
        self._ids: Dict[Tuple[str, int], object] = {}
        self._autoconvert: Dict[Tuple[str, int], bool] = {}
        self._available: Optional[List[Tuple[str, int]]] = None
        self._filled: Dict[Tuple[str, int], List[str]] = {}
        self._signals: Dict[tuple, Union[SignalData, IdscopeError]] = {}
        self.notes: List[str] = []
        """Informational messages, e.g. about Data Dictionary conversions."""

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def uri(self) -> str:
        return self.spec.uri

    def __repr__(self) -> str:
        return f"DataSource({self.label!r}, {self.uri!r})"

    @property
    def entry(self) -> imas.DBEntry:
        """The underlying ``imas.DBEntry``, opened on first use."""
        if self._entry is None:
            kwargs = {"dd_version": self.dd_version} if self.dd_version else {}
            try:
                self._entry = imas.DBEntry(self.uri, "r", **kwargs)
            except Exception as exc:
                raise SourceError(
                    f"Cannot open '{self.label}' ({self.uri}): {exc}"
                ) from exc
        return self._entry

    def get_ids(self, ids_name: str, occurrence: int = 0):
        """Get a lazily loaded IDS.

        IDSs stored with another major Data Dictionary version than the one used by
        idscope cannot be converted automatically; they are loaded in their own
        version instead, so paths must use that version's names.
        """
        key = (ids_name, occurrence)
        if key not in self._ids:
            self._ids[key] = self._load_ids(ids_name, occurrence)
        return self._ids[key]

    def _load_ids(self, ids_name: str, occurrence: int):
        entry = self.entry
        if ids_name not in entry.factory.ids_names():
            raise NoDataError(
                f"'{ids_name}' is not an IDS in Data Dictionary {entry.dd_version}."
            )
        what = f"'{ids_name}'" + (f" (occurrence {occurrence})" if occurrence else "")
        try:
            ids = entry.get(ids_name, occurrence, lazy=True)
            self._autoconvert[(ids_name, occurrence)] = True
            return ids
        except RuntimeError as exc:
            if "major version" not in str(exc):
                raise NoDataError(f"{what} cannot be read from '{self.label}': {exc}")
        # Stored with another major DD version: load it as it is stored.
        try:
            ids = entry.get(ids_name, occurrence, lazy=True, autoconvert=False)
        except Exception as exc:
            raise NoDataError(
                f"{what} cannot be read from '{self.label}': {exc}"
            ) from exc
        self._autoconvert[(ids_name, occurrence)] = False
        note = (
            f"{ids_name} in '{self.label}' is stored in DD {ids._dd_version} and is "
            f"shown with that version's paths (idscope uses DD {entry.dd_version})."
        )
        logger.info(note)
        self.notes.append(note)
        return ids

    def available_ids(self) -> List[Tuple[str, int]]:
        """List the ``(ids_name, occurrence)`` pairs present in this source."""
        if self._available is None:
            entry = self.entry
            result = []
            for ids_name in entry.factory.ids_names():
                try:
                    occurrences = entry.list_all_occurrences(ids_name)
                except Exception as exc:
                    raise SourceError(
                        f"Cannot list the IDSs in '{self.label}': {exc}"
                    ) from exc
                result.extend((ids_name, occ) for occ in occurrences)
            self._available = result
        return self._available

    def filled_paths(self, ids_name: str, occurrence: int = 0) -> List[str]:
        """Data Dictionary paths (without indices) that hold data."""
        key = (ids_name, occurrence)
        if key not in self._filled:
            self.get_ids(ids_name, occurrence)  # determines autoconvert
            try:
                paths = self.entry.list_filled_paths(
                    ids_name, occurrence, autoconvert=self._autoconvert[key]
                )
            except Exception as exc:
                raise SourceError(
                    f"Cannot list the filled quantities of '{ids_name}' in "
                    f"'{self.label}': {exc}"
                ) from exc
            self._filled[key] = sorted(paths)
        return self._filled[key]

    def signal(
        self, path: SignalPath, x_path: Optional[SignalPath] = None
    ) -> SignalData:
        """Extract a signal, optionally with the x values taken from ``x_path``.

        Results (including errors) are cached until :meth:`close`.
        """
        key = (path.canonical(), x_path.canonical() if x_path else None)
        cached = self._signals.get(key)
        if cached is None:
            try:
                if x_path is None:
                    ids = self.get_ids(path.ids_name, path.occurrence)
                    cached = extract_signal(ids, path)
                else:  # both are cached: x is often shared by many signals
                    cached = apply_x(self.signal(path), self.signal(x_path))
            except IdscopeError as exc:
                cached = exc
            self._signals[key] = cached
        if isinstance(cached, IdscopeError):
            raise type(cached)(*cached.args)
        return cached

    def first_signal(
        self,
        paths: Sequence[SignalPath],
        x_paths: Sequence[Optional[SignalPath]] = (None,),
    ) -> SignalData:
        """Extract the first alternative that can be read.

        Every path that can be read is tried with every x path, in order. When none
        works, the error reported is the first one that is not about the path itself
        (e.g. data not filled), since that is the most informative.
        """
        x_paths = list(x_paths) or [None]
        errors: List[IdscopeError] = []
        x_tried = False
        for path in paths:
            if x_paths != [None]:
                try:
                    self.signal(path)  # without its values, no x path can help
                except IdscopeError as exc:
                    errors.append(exc)
                    continue
                x_tried = True
            for x_path in x_paths:
                try:
                    return self.signal(path, x_path)
                except IdscopeError as exc:
                    errors.append(exc)
        chosen = next((e for e in errors if not isinstance(e, PathError)), errors[0])
        tried = []
        if len(paths) > 1:
            tried.append(" | ".join(path.text for path in paths))
        if x_tried and len(x_paths) > 1:
            tried.append("x: " + " | ".join(x_path.text for x_path in x_paths))
        if tried:
            raise type(chosen)(f"{chosen} (tried {'; '.join(tried)})")
        raise chosen

    def close(self) -> None:
        """Close the data entry and drop all caches (the next access reopens it)."""
        self._ids.clear()
        self._autoconvert.clear()
        self._available = None
        self._filled.clear()
        self._signals.clear()
        self.notes.clear()
        if self._entry is not None:
            try:
                self._entry.close()
            except Exception:  # pragma: no cover - closing errors are not actionable
                logger.debug("Error while closing %s", self.uri, exc_info=True)
            self._entry = None


class SourcePool:
    """Keeps data sources open across updates, keyed by URI."""

    def __init__(self, dd_version: Optional[str] = None):
        self.dd_version = dd_version
        self._sources: Dict[str, DataSource] = {}

    def get(self, spec: SourceSpec) -> DataSource:
        source = self._sources.get(spec.uri)
        if source is None:
            source = self._sources[spec.uri] = DataSource(spec, self.dd_version)
        elif source.spec != spec:
            source.spec = spec  # same data, new label
        return source

    def sources(self) -> List[DataSource]:
        return list(self._sources.values())

    def keep_only(self, specs: List[SourceSpec]) -> None:
        """Close the sources that are not in ``specs``."""
        keep = {spec.uri for spec in specs}
        for uri in list(self._sources):
            if uri not in keep:
                self._sources.pop(uri).close()

    def reload(self) -> None:
        """Close all entries so that data is read again on next access."""
        for source in self._sources.values():
            source.close()

    def close(self) -> None:
        for source in self._sources.values():
            source.close()
        self._sources.clear()
