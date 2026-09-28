# idscope

A lightweight [jScope](https://www.mdsplus.org/index.php/Documentation:Tutorial:GUIs:jScope)-like
viewer for [IMAS](https://imas.iter.org) IDS data. It reads data with
[imas-python](https://github.com/iterorganization/IMAS-Python) and draws it with
[pyqtgraph](https://www.pyqtgraph.org):

- a grid of panels (columns × rows) with time traces, profiles and 2D maps,
- a time cursor: profiles and maps follow the time picked on a trace or on the time bar,
- several data sources overlaid in every panel, like shots in jScope,
- an IDS browser that lists the quantities filled in a source,
- configurations saved as YAML, which can be applied to other data.

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
idscope shot.nc                              # browse a netCDF file
idscope scope.yaml                           # open a configuration
idscope scope.yaml -s A=a.nc -s "ref=imas:hdf5?path=/data/entry"
                                             # same configuration, other data
idscope scope.yaml -t 12.5 -e scope.png      # save an image at t = 12.5 s, no window
idscope --list shot.nc                       # list the IDSs of a source
idscope --list shot.nc core_profiles         # list the quantities of an IDS
```

## Using the scope

| jScope | idscope |
|---|---|
| Shot field + Apply | **Sources** field + Apply: comma-separated sources, optionally `label=source` |
| Expression (TDI) | Signal **path**, e.g. `equilibrium/time_slice/global_quantities/ip`, with scale and offset |
| Browse signals | **IDS browser** (left): double-click or drag a quantity onto a panel |
| Wave setup (double-click) | **Setup** dialog: double-click a panel |
| Zoom / Point / Pan modes | **Zoom** / **Point** / **Pan** in the toolbar |
| Save / Load settings (`.jscp`) | File → Save / Open (YAML) |

- **Sources** are IDS netCDF files (`shot.nc`), IMAS URIs
  (`imas:hdf5?path=/data/entry`, `imas:mdsplus?...`) or HDF5 entry directories.
  Relative paths are relative to the configuration file.
- **Zoom**: drag a rectangle; right-drag scales the axes, the wheel zooms.
  **Pan**: drag to move. **Point**: click or drag in a time trace to set the time;
  the status bar shows the values of the current panel at that time.
- The **time bar** at the bottom steps through the time slices of the profiles and
  maps (also Ctrl+Left / Ctrl+Right). The dashed line in time traces can be dragged.
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
equilibrium/time_slice/profiles_2d[0]/psi            2D map, per time slice
equilibrium:1/time_slice/global_quantities/ip        occurrence 1 of the IDS
```

- The time-dependent array of structures (`profiles_1d`, `time_slice`, ...) is left
  without index; it becomes the time axis. `[:]`, `(:)` and `(itime)` are accepted too.
- Other arrays of structures need an index: `ion[0]`, `flux_loop(1)`.
- The x axis and the time come from the Data Dictionary coordinates, for homogeneous
  and heterogeneous time alike. Another x can be chosen per signal (`x:` in the
  configuration, *X path* in the setup dialog), e.g. `core_profiles/profiles_1d/grid/psi`,
  or `equilibrium/time_slice/boundary/outline/r` to draw the boundary in (R, Z).
- The plotted value is `value × scale + offset`.

Data stored with another major Data Dictionary version than the one idscope uses
(e.g. DD 3 data with DD 4) is shown in its own version, so paths use that version's
names; this is noted in the log. `--dd-version` selects the version to use.

## Panel kinds

| kind | shows | from |
|---|---|---|
| `trace` | y(t) | scalars in a time array of structures, arrays with a time coordinate |
| `profile` | y(x) at the current time | 1D quantities |
| `map` | z(x1, x2) at the current time | 2D quantities, e.g. `profiles_2d[0]/psi` |
| `spacetime` | a profile as an image z(t, x) | 1D quantities that depend on time |

`auto` (the default) picks the kind from the first signal. Maps and time-space images
show one source at a time; choose it with the *Show* selector of the panel.

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
        signals:
          - equilibrium/time_slice/profiles_2d[0]/psi
```

Panel options: `title`, `kind`, `signals`, `x_label`, `y_label`, `x_range`, `y_range`,
`log_x`, `log_y`, `legend`, `equal_aspect`, `height`. Signal options: `path`, `label`,
`scale`, `offset`, `x`, `color`, `sources`. Other top-level options: `link_time_axes`
(zoom all time axes together, default true) and `dd_version`.

## Development

```bash
pip install -e ".[qt,test]"
QT_QPA_PLATFORM=offscreen pytest
```

The tests create synthetic data (netCDF and HDF5) and use the ITER 134173/106 training
data bundled with imas-python.

Code layout: `idscope.data` (sources, signal paths, extraction; no Qt),
`idscope.config` (YAML configuration), `idscope.gui` (Qt widgets), `idscope.cli`.
