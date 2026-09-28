"""Synthetic demo data, to try idscope without access to real IMAS data.

``python -m idscope.demo DIRECTORY`` writes two netCDF files (``demo_A.nc`` and
``demo_B.nc``, the second one with 80 % of the amplitudes) and an example
configuration ``demo.yaml`` that plots them.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

N_TIME = 41
N_RHO = 51


def _profile_scale(t: np.ndarray) -> np.ndarray:
    """Central temperature: ramp-up to a flat top with sawteeth, in eV."""
    ramp = 0.5 + 0.5 * np.tanh((t - 2.0) / 0.8)
    sawtooth = 0.08 * ((t % 1.3) / 1.3 - 0.5)
    return 1.0e3 + 7.0e3 * ramp * (1.0 + sawtooth)


def core_profiles(factory, scale: float = 1.0, n_time: int = N_TIME):
    """core_profiles with homogeneous time, electrons and two ion species.

    The second ion species is missing in the last three time slices.
    """
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    cp = factory.core_profiles()
    cp.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    cp.ids_properties.comment = "idscope synthetic demo data"
    t = np.linspace(0.0, 10.0, n_time)
    cp.time = t
    rho = np.linspace(0.0, 1.0, N_RHO)
    te0 = scale * _profile_scale(t)
    cp.profiles_1d.resize(n_time)
    for i, p1d in enumerate(cp.profiles_1d):
        p1d.time = t[i]
        p1d.grid.rho_tor_norm = rho
        p1d.grid.psi = 20.0 * rho**2
        shape = (1.0 - rho**2) ** 1.5
        p1d.electrons.temperature = te0[i] * shape + 50.0
        p1d.electrons.density = scale * 5.0e19 * (1.0 - 0.7 * rho**2) + 2.0e18
        n_ion = 2 if i < n_time - 3 else 1
        p1d.ion.resize(n_ion)
        for j, ion in enumerate(p1d.ion):
            ion.z_ion = 1.0 + j
            ion.temperature = (0.8 - 0.2 * j) * te0[i] * shape + 40.0
    cp.global_quantities.ip = scale * 1.0e6 * np.minimum(t / 2.0, 1.0)
    return cp


def equilibrium(factory, scale: float = 1.0, n_time: int = N_TIME):
    """equilibrium with scalars, 1D profiles and a rectangular (R, Z) psi map."""
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    eq = factory.equilibrium()
    eq.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    t = np.linspace(0.0, 10.0, n_time)
    eq.time = t
    eq.vacuum_toroidal_field.r0 = 1.8
    eq.vacuum_toroidal_field.b0 = np.full(n_time, -2.5)
    r = np.linspace(1.2, 2.4, 45)
    z = np.linspace(-1.2, 1.2, 81)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    rho = np.linspace(0.0, 1.0, N_RHO)
    theta = np.linspace(0.0, 2.0 * np.pi, 101)
    eq.time_slice.resize(n_time)
    for i, ts in enumerate(eq.time_slice):
        ramp = min(t[i] / 2.0, 1.0)
        r0 = 1.8 + 0.03 * np.sin(t[i])
        kappa = 1.2 + 0.4 * ramp
        ts.time = t[i]
        ts.global_quantities.ip = -scale * 1.0e6 * ramp
        ts.global_quantities.li_3 = 0.9 + 0.1 * np.cos(0.5 * t[i])
        ts.profiles_1d.psi = rho**2
        ts.profiles_1d.rho_tor_norm = rho
        ts.profiles_1d.q = 1.0 + (2.0 + 1.0 * ramp) * rho**2
        ts.profiles_2d.resize(1)
        p2d = ts.profiles_2d[0]
        p2d.grid_type.index = 1
        p2d.grid.dim1 = r
        p2d.grid.dim2 = z
        p2d.psi = scale * (((rr - r0) / 0.5) ** 2 + (zz / (0.5 * kappa)) ** 2)
        ts.boundary.outline.r = r0 + 0.5 * np.cos(theta + 0.3 * np.sin(theta))
        ts.boundary.outline.z = 0.5 * kappa * np.sin(theta)
    return eq


def magnetics(factory, scale: float = 1.0):
    """magnetics with heterogeneous time: each signal has its own time base."""
    from imas.ids_defs import IDS_TIME_MODE_HETEROGENEOUS

    mag = factory.magnetics()
    mag.ids_properties.homogeneous_time = IDS_TIME_MODE_HETEROGENEOUS
    t = np.linspace(0.0, 10.0, 2001)
    mag.flux_loop.resize(3)
    for k, loop in enumerate(mag.flux_loop):
        loop.name = f"FL{k + 1}"
        loop.flux.time = t
        loop.flux.data = scale * (0.1 * (k + 1) * np.minimum(t / 2.0, 1.0)
                                  + 0.002 * np.sin(40.0 * t + k))
    mag.ip.resize(1)
    mag.ip[0].time = t[::10]
    mag.ip[0].data = scale * 1.0e6 * np.minimum(t[::10] / 2.0, 1.0)
    return mag


def summary(factory, scale: float = 1.0, n_time: int = N_TIME):
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    summ = factory.summary()
    summ.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    t = np.linspace(0.0, 10.0, n_time)
    summ.time = t
    summ.global_quantities.ip.value = scale * 1.0e6 * np.minimum(t / 2.0, 1.0)
    return summ


def write_entry(uri: str, scale: float = 1.0, dd_version: Optional[str] = None) -> None:
    """Write all demo IDSs to ``uri`` (a netCDF path or an IMAS URI)."""
    import imas

    kwargs = {"dd_version": dd_version} if dd_version else {}
    with imas.DBEntry(uri, "w", **kwargs) as entry:
        factory = entry.factory
        for ids in (core_profiles(factory, scale), equilibrium(factory, scale),
                    magnetics(factory, scale), summary(factory, scale)):
            entry.put(ids)


DEMO_CONFIG = """\
# idscope demo configuration (created by `python -m idscope.demo`)
title: idscope demo
sources:
  - A=demo_A.nc
  - B=demo_B.nc
columns:
  - panels:
      - title: Plasma current
        signals:
          - path: equilibrium/time_slice/global_quantities/ip
            label: Ip (equilibrium)
            scale: -1.0e-6
        y_label: Ip [MA]
      - title: Flux loop 2
        signals:
          - magnetics/flux_loop[1]/flux/data
      - title: Central Te
        signals:
          - path: core_profiles/profiles_1d/electrons/temperature[0]
            scale: 1.0e-3
        y_label: Te(0) [keV]
  - panels:
      - title: Electron temperature
        signals:
          - path: core_profiles/profiles_1d/electrons/temperature
            scale: 1.0e-3
        y_label: Te [keV]
      - title: Safety factor
        signals:
          - equilibrium/time_slice/profiles_1d/q
      - title: Te(rho, t)
        kind: spacetime
        signals:
          - core_profiles/profiles_1d/electrons/temperature
  - panels:
      - title: Poloidal flux
        kind: map
        equal_aspect: true
        signals:
          - equilibrium/time_slice/profiles_2d[0]/psi
"""


def write_demo(directory, with_config: bool = True) -> List[Path]:
    """Write ``demo_A.nc``, ``demo_B.nc`` (and ``demo.yaml``) to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, scale in (("demo_A.nc", 1.0), ("demo_B.nc", 0.8)):
        path = directory / name
        write_entry(str(path), scale)
        written.append(path)
    if with_config:
        config = directory / "demo.yaml"
        config.write_text(DEMO_CONFIG)
        written.append(config)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m idscope.demo", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("directory", nargs="?", default=".", help="output directory")
    args = parser.parse_args(argv)
    for path in write_demo(args.directory):
        print(f"wrote {path}")
    print(f"try: idscope {Path(args.directory) / 'demo.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
