# idscope

A lightweight [jScope](https://www.mdsplus.org/index.php/Documentation:Tutorial:GUIs:jScope)-like
viewer for [IMAS](https://imas.iter.org) IDS data. It reads data with
[imas-python](https://github.com/iterorganization/IMAS-Python) and draws it with
[pyqtgraph](https://www.pyqtgraph.org):

- a grid of panels (columns × rows) with time traces, profiles and 2D maps, stacked
  tightly on shared x axes as in jScope,
- a time cursor: profiles and maps follow the time picked on a trace or on the time bar,
- several data sources overlaid in every panel, like shots in jScope,
- wildcards over arrays of structures: every coil, channel or ion species at once,
- coils, walls and plasma boundaries drawn in the poloidal (R, Z) plane,
- an IDS browser that lists the quantities filled in a source,
- configurations saved as YAML, which can be applied to other data,
- templates for `core_profiles`, `equilibrium` and `pf_active`.

![idscope showing two synthetic data sources](docs/idscope-demo.png)

## Installation

```bash
git clone https://github.com/icyspirit/idscope.git
cd idscope
pip install -e ".[qt]"
```

`[qt]` installs PySide6. idscope works with any Qt binding supported by pyqtgraph,
so on systems where PySide6 wheels are not available use `pip install -e . PyQt5`
instead (or a Qt binding from your environment modules).

imas-python ≥ 2.3 installs the IMAS Access Layer core (`imas-core`) from PyPI, which
reads HDF5 data entries. netCDF files are read by imas-python itself. Other backends
(MDSplus, UDA) need an IMAS-Core installation that includes them, e.g. from your
site's environment modules.

## Quick start

```bash
python -m idscope.demo demo/      # writes demo_A.nc, demo_B.nc and demo.yaml
idscope demo/demo.yaml
```

Other ways to start:

```bash
idscope -T equilibrium a.nc b.nc             # compare two sources with a template
idscope shot.nc                              # browse a netCDF file
idscope scope.yaml                           # open a configuration
idscope scope.yaml -s A=a.nc -s "ref=imas:hdf5?path=/data/entry"
                                             # same configuration, other data
idscope scope.yaml -t 12.5 -e scope.png      # save an image at t = 12.5 s, no window
idscope --list shot.nc                       # list the IDSs of a source
idscope --list shot.nc core_profiles         # list the quantities of an IDS
```

## Templates

Templates are ready-made configurations for common IDSs. Start from one with
`idscope -T NAME SOURCE ...` or File → New from template; `idscope --list-templates`
prints their names.

| template | shows |
|---|---|
| `core_profiles` | Ip, central Te/Ti/ne, loop voltage, li and beta over time; Te/Ti, ne/ni, q, current densities and Zeff profiles; Te and ne against (t, x) |
| `equilibrium` | profiles: q, pressure, current densities (j_phi, j_parallel), dp/dpsi, FF', elongation and triangularities |
| `pf_active` | current and voltage of every coil over time; the coil currents at the selected time, as bars; the coils in the poloidal plane, with the flux surfaces, the plasma boundary, the magnetic axis and the wall |

Profiles are drawn against rho_tor_norm or, when it is not filled, against the
normalized poloidal flux psi_n (`psi_norm` if filled, otherwise `norm(psi)`, see
below). The templates work with Data Dictionary 3 and 4 data thanks to alternative
paths. Panels whose quantities are not filled in a source say so, e.g. the coil
voltages of a pf_active written by an equilibrium code, which only has currents.
A template saved with File → Save as becomes a configuration that can be edited.

![equilibrium template on the ITER 134173/106 training data](docs/template-equilibrium.png)

![pf_active template comparing two synthetic sources](docs/template-pf_active.png)

## Using the scope

| jScope | idscope |
|---|---|
| Shot field + Apply | **Sources** field + Apply: comma-separated sources, optionally `label=source` |
| Expression (TDI) | Signal **path**, e.g. `equilibrium/time_slice/global_quantities/ip`, with scale and offset |
| Browse signals | **IDS browser** (left): double-click or drag a quantity onto a panel |
| Wave setup (double-click) | **Setup** dialog: double-click a panel |
| Zoom / Point / Pan modes | **Pan** (default) / **Zoom** / **Point** in the toolbar |
| Save / Load settings (`.jscp`) | File → Save / Open (YAML) |

- **Sources** are IDS netCDF files (`shot.nc`), IMAS URIs
  (`imas:hdf5?path=/data/entry`, `imas:mdsplus?...`) or HDF5 entry directories.
  Relative paths are relative to the configuration file.
- **Pan** (the default): drag to move; right-drag scales the axes, the wheel zooms.
  **Zoom**: drag a rectangle. **Point**: click or drag in a time trace to set the
  time; the status bar shows the values of the current panel at that time.
- The **time bar** at the bottom steps through the time slices of the profiles and
  maps (also Ctrl+Left / Ctrl+Right). The dashed line in time traces can be dragged.
- Panels that show the same x quantity **share their x axis** (View → Share x axes):
  time across the window, a profile coordinate within a column. Stacked panels that
  share their x axis are drawn without gaps, and only the lowest shows the x values.
- Right-click a panel for its menu: setup, autoscale, add/remove panels and columns,
  and pyqtgraph's *Export…* (PNG, SVG, CSV of the curves).
- Signals that cannot be shown are reported in the panel (hover for details) and in
  View → Log.

## Signal paths

A path names an IDS and a quantity inside it:

```text
core_profiles/profiles_1d/electrons/temperature      profile vs rho_tor_norm, per time slice
equilibrium/time_slice/global_quantities/ip          time trace
summary/global_quantities/ip/value                   time trace
magnetics/flux_loop[0]/flux/data                     time trace (0-based index)
magnetics/flux_loop(1)/flux/data                     the same, Fortran-style 1-based index
core_profiles/profiles_1d/electrons/temperature[0]   time trace of the first grid point
core_profiles/profiles_1d[3]/electrons/temperature   profile of time slice 3 only
core_profiles/profiles_1d[-1]/electrons/temperature  profile of the last time slice
equilibrium/time_slice/profiles_2d[0]/psi            2D map, per time slice
equilibrium/time_slice/profiles_2d[0]/psi[:, 40]     psi along R at the 41st Z, per time slice
equilibrium:1/time_slice/global_quantities/ip        occurrence 1 of the IDS
norm(equilibrium/time_slice/profiles_1d/psi)         psi normalized per time slice: psi_n
pf_active/coil[*]/current/data                       a time trace per coil
pf_active/coil[0:6]/current/data                     the first six coils (Fortran: coil(1:6))
pf_active/coil[*]/current/data[:, 0]                 the first current sample of every coil
core_profiles/profiles_1d/ion[*]/temperature         a profile per ion species
pf_active/coil[*]/element[*]/geometry                the coils in (R, Z)
wall/description_2d[0]/limiter/unit[*]/outline       the wall in (R, Z)
equilibrium/time_slice/boundary/outline              the plasma boundary in (R, Z), per time slice
equilibrium/time_slice/global_quantities/magnetic_axis   the magnetic axis, a point per time slice
```

- The time-dependent array of structures (`profiles_1d`, `time_slice`, ...) is left
  without index; it becomes the time axis (`[:]`, `(:)` and `(itime)` as in the Data
  Dictionary documentation are accepted too). An index selects one time slice, a
  range some.
- Other arrays of structures need an index: `ion[0]`, `flux_loop(1)`, `coil[-1]`
  (the last one), or a **wildcard**: `*` or `:` for all elements, `a:b` or `a:b:step`
  for some of them (Fortran ranges `(a:b)` include both ends). A path with wildcards
  has a member per element, named after its `name` (or `label`, `identifier`): a
  curve per coil in time traces, a bar per coil at the selected time in a profile
  panel. Wildcards can be nested (`coil[*]/element[*]`).
- An index on the last node selects numpy style along every dimension of the
  result: first one per wildcard, in the order of the path, then those of the data.
  So `coil[*]/current/data[:, 0]` is the first sample of every coil,
  `coil[*]/current/data[2]` the current of coil 2, and
  `profiles_1d/ion[*]/temperature[:, 0]` the central temperature of every ion
  species. The time array of structures is the time axis, not a wildcard, even
  written `profiles_1d[:]`: `profiles_1d[:]/electrons/temperature[0]` is the
  temperature on the first grid point, over time.
- A path to a structure that describes a shape draws it in the poloidal plane, over
  maps: the 2D geometry of coils and passive loops (outline, rectangle,
  parallelogram, arcs of circle, annulus or thick line), `r`/`z` outlines and
  `r`/`z` points. The IDS browser lists them as *shape*.
- The x axis and the time come from the Data Dictionary coordinates, for homogeneous
  and heterogeneous time alike. Another x can be chosen per signal (`x:` in the
  configuration, *X path* in the setup dialog), e.g. `core_profiles/profiles_1d/grid/psi`.
- The plotted value is `value × scale + offset`.
- Alternatives separated by `|` are tried in order, and the first that can be read is
  shown. They help with quantities renamed between Data Dictionary versions
  (`equilibrium/time_slice/profiles_1d/j_phi | equilibrium/time_slice/profiles_1d/j_tor`)
  or filled in different IDSs
  (`core_profiles/global_quantities/ip | equilibrium/time_slice/global_quantities/ip`).
- `norm(path)` normalizes every profile from its first to its last point,
  `(y - y[0]) / (y[-1] - y[0])`, and adds `_n` to its name. Since the profiles of
  `psi` go from the magnetic axis to the boundary, `norm(.../psi)` is psi_n. As x
  path, `core_profiles/profiles_1d/grid/rho_tor_norm | norm(core_profiles/profiles_1d/grid/psi)`
  draws against rho_tor_norm, or against psi_n when rho_tor_norm is not filled.

Data stored with another major Data Dictionary version than the one idscope uses
(e.g. DD 3 data with DD 4) is shown in its own version, so paths use that version's
names; this is noted in the log. `--dd-version` selects the version to use.

## Panel kinds

| kind | shows | from |
|---|---|---|
| `trace` | y(t) | scalars in a time array of structures, arrays with a time coordinate |
| `profile` | y(x) at the current time, or a bar per element | 1D quantities; values of paths with wildcards |
| `map` | z(x1, x2) at the current time, with shapes over it | 2D quantities, e.g. `profiles_2d[0]/psi`; shapes in (R, Z) |
| `spacetime` | a profile as an image z(t, x) | 1D quantities that depend on time |

`auto` (the default) picks the kind from the first signal; a value per element of a
wildcard (e.g. `coil[*]/element[*]/turns_with_sign`) makes bars. Maps and time-space
images show one source at a time; choose it with the *Show* selector of the panel.
Shapes and profiles added to a map are drawn over it for every source, e.g. the coils
and the plasma boundary over psi(R, Z); a map panel can also hold shapes only. With
`contours: N`, maps and time-space images are drawn as N contour lines colored by
level instead of an image.

When a panel compares several sources, curves take the color of their signal (or
member, e.g. coil) and the line style of their source; the legend has an entry for
each, instead of one for every combination.

## Configuration

```yaml
title: Shot overview
sources:
  - A=shot_a.nc
  - ref=imas:hdf5?path=/data/entries/134173/106
time: 5.0                 # initial time (optional)
columns:
  - panels:
      - title: Plasma current
        signals:
          - equilibrium/time_slice/global_quantities/ip    # short form
          - path: summary/global_quantities/ip/value
            label: Ip (summary)
            scale: 1.0e-6
            sources: [A]  # only some sources (default: all)
        y_label: Ip [MA]
  - width: 1.5            # relative column width
    panels:
      - title: Poloidal flux
        kind: map
        equal_aspect: true
        contours: 12
        signals:
          - equilibrium/time_slice/profiles_2d[0]/psi
          - path: equilibrium/time_slice/boundary/outline
            label: boundary
          - pf_active/coil[*]/element[*]/geometry
```

Panel options: `title`, `kind`, `signals`, `x_label`, `y_label`, `x_range`, `y_range`,
`log_x`, `log_y`, `legend`, `equal_aspect`, `contours`, `height`. Signal options:
`path`, `label`, `scale`, `offset`, `x`, `color`, `sources`. Other top-level options:
`share_x` (share x axes and stack panels tightly, default true) and `dd_version`.

## Development

```bash
pip install -e ".[qt,test]"
QT_QPA_PLATFORM=offscreen pytest
```

The tests create synthetic data (netCDF and HDF5) and use the ITER 134173/106 training
data bundled with imas-python.

Code layout: `idscope.data` (sources, signal paths, extraction; no Qt),
`idscope.config` (YAML configuration), `idscope.templates` (YAML templates),
`idscope.gui` (Qt widgets), `idscope.cli`.
