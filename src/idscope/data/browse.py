"""Listing of the plottable quantities in a data source (the IDS browser)."""

from typing import List, NamedTuple, Optional

from imas.ids_data_type import IDSDataType

from idscope.data.extract import MAP, PROFILE, SHAPE, TRACE, is_time_aos, shape_kind
from idscope.data.sources import DataSource

CONSTANT = "constant"
UNSUPPORTED = "unsupported"

# Subtrees that describe the IDS itself rather than physics quantities
_SKIPPED_ROOTS = ("ids_properties", "code")


class BrowseItem(NamedTuple):
    dd_path: str
    """Data Dictionary path without indices, e.g. ``profiles_1d/electrons/density``."""
    signal_path: str
    """Suggested signal path: with ``[*]`` (all elements) for the arrays of
    structures that do not depend on time, ``[0]`` for maps."""
    kind: str
    """``trace``, ``profile``, ``map``, ``shape``, ``constant`` or ``unsupported``."""
    time_dependent: bool
    units: str
    documentation: str


def classify(ids_meta, dd_path: str):
    """Find the kind of a Data Dictionary path and a signal path suggestion.

    Returns:
        ``(kind, time_dependent, relative_signal_path)``, or ``None`` if the path
        is not a numeric data node.
    """
    meta = ids_meta
    parts = []  # (name, is an array of structures to index)
    time_aos = False
    for name in dd_path.split("/"):
        try:
            meta = meta[name]
        except KeyError:
            return None
        if meta.data_type is IDSDataType.STRUCT_ARRAY:
            indexed = not (is_time_aos(meta) and not time_aos)
            time_aos |= not indexed
            parts.append((name, indexed))
        else:
            parts.append((name, False))
    if meta.data_type not in (IDSDataType.FLT, IDSDataType.INT):
        kind = SHAPE if shape_kind(meta) else None
        if kind is None:
            return None
        return kind, time_aos, _signal_path(parts, "*")
    has_time_dim = any(c.is_time_coordinate for c in meta.coordinates)
    time_dependent = time_aos or has_time_dim
    other_dims = meta.ndim - (has_time_dim and not time_aos)
    if other_dims == 0:
        kind = TRACE if time_dependent else CONSTANT
    elif other_dims == 1:
        kind = PROFILE
    elif other_dims == 2:
        kind = MAP
    else:
        kind = UNSUPPORTED
    # All elements, except for maps: one map is shown at a time
    return kind, time_dependent, _signal_path(parts, "0" if kind == MAP else "*")


def _signal_path(parts, index: str) -> str:
    return "/".join(f"{name}[{index}]" if indexed else name for name, indexed in parts)


def shape_of(ids_meta, dd_path: str) -> Optional[str]:
    """The Data Dictionary path of the outermost shape (e.g. a coil geometry, an
    outline or a point) that ``dd_path`` belongs to, if any."""
    meta = ids_meta
    parts = dd_path.split("/")
    for depth, name in enumerate(parts[:-1]):
        try:
            meta = meta[name]
        except KeyError:
            return None
        if shape_kind(meta):
            return "/".join(parts[: depth + 1])
    return None


def list_signals(
    source: DataSource,
    ids_name: str,
    occurrence: int = 0,
    include_all: bool = False,
) -> List[BrowseItem]:
    """List the filled numeric quantities of an IDS.

    Args:
        include_all: also list time vectors, ``ids_properties`` and ``code``.
    """
    ids = source.get_ids(ids_name, occurrence)
    head = ids_name + (f":{occurrence}" if occurrence else "")
    items = []
    shapes: List[str] = []
    for dd_path in source.filled_paths(ids_name, occurrence):
        shape = shape_of(ids.metadata, dd_path)
        if shape is not None and shape not in shapes:
            shapes.append(shape)
        if not include_all:
            if dd_path.split("/", 1)[0] in _SKIPPED_ROOTS:
                continue
            if dd_path == "time" or dd_path.endswith("/time"):
                continue
        result = classify(ids.metadata, dd_path)
        if result is None:
            continue
        kind, time_dependent, relative = result
        meta = ids.metadata[dd_path]
        items.append(
            BrowseItem(
                dd_path,
                f"{head}/{relative}",
                kind,
                time_dependent,
                meta.units,
                meta.documentation or "",
            )
        )
    for dd_path in shapes:
        kind, time_dependent, relative = classify(ids.metadata, dd_path)
        meta = ids.metadata[dd_path]
        items.append(BrowseItem(dd_path, f"{head}/{relative}", kind, time_dependent,
                                "m", meta.documentation or ""))
    return items


def find_item(items: List[BrowseItem], dd_path: str) -> Optional[BrowseItem]:
    for item in items:
        if item.dd_path == dd_path:
            return item
    return None
