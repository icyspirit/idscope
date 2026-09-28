"""Listing of the plottable quantities in a data source (the IDS browser)."""

from typing import List, NamedTuple, Optional

from imas.ids_data_type import IDSDataType

from idscope.data.extract import MAP, PROFILE, TRACE, is_time_aos
from idscope.data.sources import DataSource

CONSTANT = "constant"
UNSUPPORTED = "unsupported"

# Subtrees that describe the IDS itself rather than physics quantities
_SKIPPED_ROOTS = ("ids_properties", "code")


class BrowseItem(NamedTuple):
    dd_path: str
    """Data Dictionary path without indices, e.g. ``profiles_1d/electrons/density``."""
    signal_path: str
    """Suggested signal path, with ``[0]`` for arrays that do not depend on time."""
    kind: str
    """``trace``, ``profile``, ``map``, ``constant`` or ``unsupported``."""
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
    parts = []
    time_aos = False
    for name in dd_path.split("/"):
        try:
            meta = meta[name]
        except KeyError:
            return None
        if meta.data_type is IDSDataType.STRUCT_ARRAY:
            if is_time_aos(meta) and not time_aos:
                time_aos = True
                parts.append(name)
            else:
                parts.append(f"{name}[0]")
        else:
            parts.append(name)
    if meta.data_type not in (IDSDataType.FLT, IDSDataType.INT):
        return None
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
    return kind, time_dependent, "/".join(parts)


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
    for dd_path in source.filled_paths(ids_name, occurrence):
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
    return items


def find_item(items: List[BrowseItem], dd_path: str) -> Optional[BrowseItem]:
    for item in items:
        if item.dd_path == dd_path:
            return item
    return None
