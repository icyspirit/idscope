"""Parsing of idscope signal paths.

A signal path points at one quantity in an IDS, for example::

    core_profiles/profiles_1d/electrons/temperature
    equilibrium:1/time_slice/global_quantities/ip
    magnetics/flux_loop[3]/flux/data
    magnetics/flux_loop(4)/flux/data
    core_profiles/profiles_1d/electrons/temperature[0]
    core_profiles.profiles_1d[:].electrons.temperature

Rules:

- The first element is the IDS name, optionally followed by ``:occurrence``.
- Nodes are separated by ``/`` or ``.``.
- Indices are written Python style ``[i]`` (0-based) or Fortran style ``(i)``
  (1-based), as in the Data Dictionary documentation. The two styles cannot be
  mixed in one path.
- A time-dependent array of structures (``profiles_1d``, ``time_slice``, ...) can be
  left without index, or written as ``[:]``, ``(:)`` or ``(itime)``. It then becomes
  the time axis of the signal. Whether an array of structures is time-dependent is
  only known from the Data Dictionary, so it is checked during extraction.
- An index on the last (data) node selects along the first dimension of the data,
  e.g. ``temperature[0]`` is the value on the first radial grid point.
- ``norm(path)`` normalizes every profile between its first and last point,
  ``(y - y[0]) / (y[-1] - y[0])``: ``norm(equilibrium/time_slice/profiles_1d/psi)``
  is the normalized poloidal flux psi_n, since the profiles go from the magnetic
  axis to the boundary.
"""

import dataclasses
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from imas.ids_path import IDSPath

from idscope.errors import PathError


class _AllType:
    """Sentinel for an index that selects all elements (``[:]`` or ``(itime)``)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ALL"

    def __reduce__(self):
        return (_AllType, ())


ALL = _AllType()
"""Index value for ``[:]``, ``(:)`` or ``(itime)``."""

Index = Union[None, int, _AllType]

_HEAD = re.compile(
    r"#?(?P<ids>[a-z][a-z0-9_]*)(?::(?P<occ>\d+))?(?:(?P<sep>[/.])(?P<rest>.*))?",
    re.ASCII | re.DOTALL,
)
_FUNCTION = re.compile(r"(?P<function>[a-z]+)\s*\((?P<inner>.*)\)", re.ASCII | re.DOTALL)

TRANSFORMS = ("norm",)
"""Functions that can be applied to a path."""


@dataclass(frozen=True)
class SignalPath:
    """A parsed signal path."""

    text: str
    """The path as written by the user (whitespace stripped)."""
    ids_name: str
    occurrence: int
    parts: Tuple[str, ...]
    """Node names below the IDS toplevel."""
    indices: Tuple[Index, ...]
    """Index of every node in :attr:`parts` (0-based int, ``ALL`` or ``None``)."""
    transform: Optional[str] = None
    """Function applied to the data, e.g. ``norm``."""

    @property
    def dd_path(self) -> str:
        """The Data Dictionary path without indices, e.g. ``profiles_1d/zeff``."""
        return "/".join(self.parts)

    def canonical(self) -> str:
        """Normalized Python-style form of the path."""
        head = self.ids_name
        if self.occurrence:
            head += f":{self.occurrence}"
        path = "/".join([head] + _format_parts(self.parts, self.indices))
        return f"{self.transform}({path})" if self.transform else path

    @property
    def name(self) -> str:
        """Name of the quantity, e.g. ``temperature``, or ``psi_n`` for ``norm(psi)``."""
        return f"{self.parts[-1]}_n" if self.transform == "norm" else self.parts[-1]

    @property
    def label(self) -> str:
        """A short label for legends, e.g. ``electrons/temperature``.

        The label starts at the last explicitly indexed node when there is one, so
        that ``flux_loop[3]/flux/data`` does not collapse to ``flux/data``.
        """
        if self.transform:
            return self.name
        formatted = _format_parts(self.parts, self.indices)
        start = max(len(formatted) - 2, 0)
        for pos, index in enumerate(self.indices):
            if isinstance(index, int):
                start = min(start, pos)
        return "/".join(formatted[start:])

    def __str__(self) -> str:
        return self.text


def _format_parts(parts, indices):
    formatted = []
    for part, index in zip(parts, indices):
        if index is ALL:
            formatted.append(f"{part}[:]")
        elif isinstance(index, int):
            formatted.append(f"{part}[{index}]")
        else:
            formatted.append(part)
    return formatted


ALTERNATIVE_SEPARATOR = "|"


def parse_alternatives(text: str) -> List[SignalPath]:
    """Parse alternative paths ``a | b | ...``, to be tried in order.

    Alternatives let one configuration work with data that uses different names,
    e.g. ``core_profiles/profiles_1d/j_phi | core_profiles/profiles_1d/j_tor`` for
    Data Dictionary 4 and 3.

    Raises:
        PathError: when one of the paths is malformed.
    """
    parts = text.split(ALTERNATIVE_SEPARATOR)
    if len(parts) > 1 and any(not part.strip() for part in parts):
        raise PathError(f"Empty alternative in '{text.strip()}'.")
    return [parse_signal_path(part) for part in parts]


def parse_signal_path(text: str) -> SignalPath:
    """Parse a signal path.

    Raises:
        PathError: when the path is malformed.
    """
    stripped = text.strip()
    function = _FUNCTION.fullmatch(stripped)
    if function:  # an IDS name is always followed by "/", so this is a function
        name = function["function"]
        if name not in TRANSFORMS:
            raise PathError(
                f"Unknown function '{name}' in '{stripped}'; available: "
                f"{', '.join(TRANSFORMS)}."
            )
        if not function["inner"].strip():
            raise PathError(
                f"'{stripped}' needs a signal path, for example "
                f"'{name}(equilibrium/time_slice/profiles_1d/psi)'."
            )
        inner = parse_signal_path(function["inner"])
        if inner.transform:
            raise PathError(f"Functions cannot be nested: '{stripped}'.")
        return dataclasses.replace(inner, text=stripped, transform=name)
    match = _HEAD.fullmatch(stripped)
    if match is None:
        raise PathError(
            f"Invalid signal path '{text}': expected 'ids_name/path/to/quantity', "
            "for example 'core_profiles/profiles_1d/electrons/temperature'."
        )
    ids_name = match["ids"]
    occurrence = int(match["occ"]) if match["occ"] else 0
    rest = match["rest"] or ""
    if not rest:
        raise PathError(
            f"Signal path '{text}' only names the IDS '{ids_name}'; add the path "
            "to a quantity, for example 'equilibrium/time_slice/global_quantities/ip'."
        )
    # Node names never contain dots, so dots can safely be used as separators.
    rest = rest.replace(".", "/")
    try:
        ids_path = IDSPath(rest)
    except ValueError as exc:
        raise PathError(f"Invalid signal path '{text}': {exc}") from None

    indices = []
    for part, index in ids_path.items():
        if isinstance(index, int) and index < 0:
            raise PathError(
                f"Invalid index for '{part}' in '{text}': Fortran-style indices "
                "such as '(1)' start at 1."
            )
        if index is None or isinstance(index, int):
            indices.append(index)
        elif isinstance(index, slice):
            if index.start is None and index.stop is None:
                indices.append(ALL)
            else:
                raise PathError(
                    f"Invalid index for '{part}' in '{text}': only single indices "
                    "or ':' (all elements) are supported."
                )
        elif index == "itime":
            indices.append(ALL)
        else:
            raise PathError(
                f"Invalid index '{index}' for '{part}' in '{text}': use an explicit "
                f"index, for example '{part}[0]' or '{part}(1)'."
            )
    return SignalPath(stripped, ids_name, occurrence, ids_path.parts, tuple(indices))
