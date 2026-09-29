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
- the last session back when idscope starts again: panels, sources and time,
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
idscope                                      # continue the last session
idscope shot.nc                              # the panels of the last session, other data
idscope --new shot.nc                        # browse a netCDF file in an empty panel
idscope -T equilibrium a.nc b.nc             # compare two sources with a template
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
| Save / Load settings (`.jscp`) | File → Save / Open (YAML); the last session comes back by itself |

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
- The IDS browser asks the data entry which quantities are filled. Access Layers
  older than 5.7, and backends such as MDSplus and UDA, cannot tell: the browser then
  reads the data instead. Of long arrays of structures it reads only some elements
  (three time slices; sixteen coils, channels, ...), so a quantity filled in none of
  those is not listed, but can still be typed as a path.

## Sessions

idscope keeps the scope of the last session: its panels, sources, sizes and time.
Started without a configuration file or a template, it shows them again.

| command | panels | sources |
|---|---|---|
| `idscope` | of the last session | of the last session |
| `idscope shot.nc` | of the last session | `shot.nc` |
| `idscope --new shot.nc` | one empty panel | `shot.nc` |
| `idscope scope.yaml [shot.nc]` | of `scope.yaml` | of `scope.yaml`, or `shot.nc` |
| `idscope -T NAME shot.nc` | of the template | `shot.nc` |

Nothing is lost when idscope closes, so it does not ask to save. Instead, the title
shows `*` while the panels have changes that are not saved to the configuration file
(the sources and the time do not count), and File → Save writes them to it. File →
Open recent lists the configuration files opened or saved last. idscope also keeps
the size of its window, its docks, the mouse mode and the IDS shown in the browser.
An export (`-e`) shows the same panels as the window would, and leaves the session
as it is.

The session (`session.yaml`, a configuration that can also be opened as one) and the
settings (`settings.ini`) are in `~/.config/idscope`, or `%APPDATA%\idscope` on
Windows, or `$IDSCOPE_CONFIG_DIR` if set.

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
pf_active/coil[:]/current/data                       a time trace per coil
pf_active/coil[0:6]/current/data                     the first six coils (Fortran: coil(1:6))
pf_active/coil[:]/current/data[0]                    the first current sample of every coil
core_profiles/profiles_1d/ion[:]/temperature         a profile per ion species
core_profiles/profiles_1d/ion[:]/temperature[0]      the central temperature of every species
pf_active/coil[:]/element[:]/geometry                the coils in (R, Z)
wall/description_2d[0]/limiter/unit[:]/outline       the wall in (R, Z)
equilibrium/time_slice/boundary/outline              the plasma boundary in (R, Z), per time slice
equilibrium/time_slice/global_quantities/magnetic_axis   the magnetic axis, a point per time slice
```

- The time-dependent array of structures (`profiles_1d`, `time_slice`, ...) is left
  without index; it becomes the time axis (`[:]`, `(:)` and `(itime)` as in the Data
  Dictionary documentation are accepted too). An index selects one time slice, a
  range some.
- Other arrays of structures need an index: `ion[0]`, `flux_loop(1)`, `coil[-1]`
  (the last one), or a **wildcard**: `:` for all elements, `a:b` or `a:b:step`
  for some of them (Fortran ranges `(a:b)` include both ends). A path with wildcards
  has a member per element, named after its `name` (or `label`, `identifier`): a
  curve per coil in time traces, a bar per coil at the selected time in a profile
  panel. Elements with the same name are told apart as `nbi`, `nbi#2`, and a member
  has the same color in all the panels of the window. Wildcards can be nested
  (`coil[:]/element[:]`).
- An index applies to the node it is written on. On the data node it selects numpy
  style along the dimensions of the data, for every element of the wildcards: so
  `coil[:]/current/data[2]` is the third current sample of every coil (a bar per
  coil), `profiles_1d/ion[:]/temperature[0]` the central temperature of every ion
  species over time, and `profiles_2d[0]/psi[:, 40]` a column of the map. Which
  coils or species is written on the array of structures: `coil[0:6]`, `ion[1]`.
- **Dimensions** keep the order of the Data Dictionary coordinates, which the tooltip
  of the IDS browser shows (e.g. `psi`: `grid/dim1, grid/dim2`, and
  `bolometer/power_density/data`: `grid/dim1, grid/dim2, time`). An index selects
  along them in that order, and a map draws the first along x and the second along
  y. idscope never reorders them by itself: `transpose(path, axes...)` does, as
  `numpy.transpose` does on the data left by the index. `transpose(path, 1, 0, 2)`
  puts the second dimension first; without axes the order is reversed (`2, 1, 0`
  for three dimensions), which swaps the two dimensions of a map. Time is a
  dimension like the others: one dimension left makes a curve along its coordinate
  (a time trace when that is time), two make a map (`power_density/data[:, 4]` has
  dim1 along x and time along y), and more need an index
  (`power_density/data[:, :, 2]`). What follows the time bar is the data of the
  time-dependent arrays of structures (`profiles_1d`, `time_slice`, ...).
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
| `trace` | y(t) | scalars in a time array of structures, arrays along a time coordinate |
| `profile` | y(x), at the current time for profiles in a time array of structures, or a bar per element | 1D quantities; values of paths with wildcards |
| `map` | z(x1, x2), at the current time for maps in a time array of structures, with shapes over it | 2D quantities, e.g. `profiles_2d[0]/psi`; shapes in (R, Z) |
| `spacetime` | a profile as an image z(t, x) | 1D quantities in a time array of structures |

`auto` (the default) picks the kind from the first signal; a value per element of a
wildcard (e.g. `coil[:]/element[:]/turns_with_sign`) makes bars. Maps and time-space
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
          - pf_active/coil[:]/element[:]/geometry
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
`idscope.session` (the last session and where settings are kept),
`idscope.gui` (Qt widgets), `idscope.cli`.
