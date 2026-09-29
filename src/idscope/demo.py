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


def _set_first(node, names, value) -> None:
    """Set the first of ``names`` that exists (quantities renamed between DD 3 and 4)."""
    for name in names:
        if hasattr(node, name):
            setattr(node, name, value)
            return


def core_profiles(factory, scale: float = 1.0, n_time: int = N_TIME,
                  rho_tor_norm: bool = True):
    """core_profiles with homogeneous time, electrons and two ion species.

    The second ion species is missing in the last three time slices.

    Args:
        rho_tor_norm: fill ``profiles_1d/grid/rho_tor_norm`` (profiles can otherwise
            be drawn against the normalized psi).
    """
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    cp = factory.core_profiles()
    cp.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    cp.ids_properties.comment = "idscope synthetic demo data"
    t = np.linspace(0.0, 10.0, n_time)
    cp.time = t
    rho = np.linspace(0.0, 1.0, N_RHO)
    te0 = scale * _profile_scale(t)
    ramp = np.minimum(t / 2.0, 1.0)
    cp.profiles_1d.resize(n_time)
    for i, p1d in enumerate(cp.profiles_1d):
        p1d.time = t[i]
        if rho_tor_norm:
            p1d.grid.rho_tor_norm = rho
        p1d.grid.psi = 20.0 * rho**2
        shape = (1.0 - rho**2) ** 1.5
        p1d.electrons.temperature = te0[i] * shape + 50.0
        density = scale * 5.0e19 * (1.0 - 0.7 * rho**2) + 2.0e18
        p1d.electrons.density = density
        n_ion = 2 if i < n_time - 3 else 1
        p1d.ion.resize(n_ion)
        for j, ion in enumerate(p1d.ion):
            _set_first(ion, ("name", "label"), ("D", "C")[j])
            ion.z_ion = 1.0 + j
            ion.temperature = (0.8 - 0.2 * j) * te0[i] * shape + 40.0
        p1d.t_i_average = 0.8 * te0[i] * shape + 40.0
        p1d.n_i_thermal_total = 0.9 * density
        p1d.zeff = 1.6 + 0.4 * rho**2
        p1d.q = 1.0 + (2.0 + ramp[i]) * rho**2
        j_total = scale * 1.5e6 * ramp[i] * (1.0 - rho**2) ** 1.2
        j_bootstrap = scale * 0.3e6 * ramp[i] * np.exp(-(((rho - 0.7) / 0.12) ** 2))
        p1d.j_total = j_total + j_bootstrap
        p1d.j_bootstrap = j_bootstrap
        p1d.j_ohmic = j_total
    cp.global_quantities.ip = scale * 1.0e6 * ramp
    cp.global_quantities.v_loop = 1.2 * np.exp(-t / 1.5) + 0.2
    cp.global_quantities.li_3 = 0.9 + 0.1 * np.cos(0.5 * t)
    cp.global_quantities.beta_pol = scale * (0.3 + 0.4 * ramp)
    return cp


def equilibrium(factory, scale: float = 1.0, n_time: int = N_TIME,
                rho_tor_norm: bool = True):
    """equilibrium with scalars, 1D profiles and a rectangular (R, Z) psi map.

    Args:
        rho_tor_norm: fill ``profiles_1d/rho_tor_norm`` (profiles can otherwise be
            drawn against the normalized psi).
    """
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
        gq = ts.global_quantities
        gq.ip = -scale * 1.0e6 * ramp
        gq.li_3 = 0.9 + 0.1 * np.cos(0.5 * t[i])
        gq.beta_pol = scale * (0.3 + 0.4 * ramp)
        _set_first(gq, ("beta_tor_norm", "beta_normal"), scale * 1.8 * ramp)
        gq.q_axis = 1.0
        gq.q_95 = 2.6 + 1.0 * ramp
        gq.magnetic_axis.r = r0 + 0.02
        gq.magnetic_axis.z = 0.01 * np.sin(2.0 * t[i])
        ts.boundary.elongation = kappa
        ts.boundary.triangularity = 0.25
        p1d = ts.profiles_1d
        psi = rho**2  # normalized: 0 on the magnetic axis, 1 on the boundary
        p1d.psi = scale * psi  # as the psi map below
        if rho_tor_norm:
            p1d.rho_tor_norm = rho
        p1d.q = 1.0 + (2.0 + 1.0 * ramp) * psi
        p1d.pressure = scale * 1.0e5 * ramp * (1.0 - psi) ** 2
        p1d.dpressure_dpsi = -scale * 2.0e5 * ramp * (1.0 - psi)
        p1d.f_df_dpsi = -scale * 0.5 * ramp * (1.0 - psi) ** 1.5
        j_phi = scale * 1.5e6 * ramp * (1.0 - psi) ** 1.2
        _set_first(p1d, ("j_phi", "j_tor"), j_phi)
        p1d.j_parallel = 1.05 * j_phi
        p1d.elongation = 1.0 + (kappa - 1.0) * rho
        p1d.triangularity_upper = 0.3 * rho**2
        p1d.triangularity_lower = 0.2 * rho**2
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


PF_COILS = [
    # name, R, Z, width, height [m], turns, current per turn at t = 0 and flat top [A]
    ("CS3U", 0.75, 1.25, 0.2, 0.5, 100, 12.0e3, -6.0e3),
    ("CS2U", 0.75, 0.75, 0.2, 0.5, 100, 15.0e3, -9.0e3),
    ("CS1U", 0.75, 0.25, 0.2, 0.5, 100, 18.0e3, -12.0e3),
    ("CS1L", 0.75, -0.25, 0.2, 0.5, 100, 18.0e3, -12.0e3),
    ("CS2L", 0.75, -0.75, 0.2, 0.5, 100, 15.0e3, -9.0e3),
    ("CS3L", 0.75, -1.25, 0.2, 0.5, 100, 12.0e3, -6.0e3),
    ("PF1", 1.25, 1.6, 0.3, 0.3, 50, 0.0, -4.0e3),
    ("PF2", 2.3, 1.4, 0.25, 0.25, 50, 0.0, -6.0e3),
    ("PF3", 2.85, 0.6, 0.2, 0.3, 50, 0.0, 8.0e3),
    ("PF4", 2.85, -0.6, 0.2, 0.3, 50, 0.0, 8.0e3),
    ("PF5", 2.3, -1.4, 0.25, 0.25, 50, 0.0, -6.0e3),
    ("PF6", 1.25, -1.6, 0.3, 0.3, 50, 0.0, -4.0e3),
]
"""Poloidal field coils of the demo machine: a central solenoid of six modules
(described as outlines) and six PF coils (described as rectangles)."""


def pf_active(factory, scale: float = 1.0, n_time: int = N_TIME,
              time: Optional[float] = None):
    """pf_active: the coils of :data:`PF_COILS`, one element each, with their
    current and voltage.

    Args:
        n_time: number of time points of the currents, from 0 to 10 s.
        time: write the currents at this single time instead, like an
            equilibrium code that solves for the coil currents.
    """
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    pf = factory.pf_active()
    pf.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    t = np.linspace(0.0, 10.0, n_time) if time is None else np.array([time])
    pf.time = t
    ramp = np.minimum(t / 2.0, 1.0)
    pf.coil.resize(len(PF_COILS))
    for coil, (name, r, z, width, height, turns, start, flat) in zip(pf.coil, PF_COILS):
        coil.name = name
        coil.element.resize(1)
        element = coil.element[0]
        element.turns_with_sign = float(turns)
        geometry = element.geometry
        if name.startswith("CS"):
            geometry.geometry_type = 1
            geometry.outline.r = np.array([r - width / 2, r + width / 2] * 2)[[0, 1, 3, 2]]
            geometry.outline.z = np.array([z - height / 2] * 2 + [z + height / 2] * 2)
        else:
            geometry.geometry_type = 2
            geometry.rectangle.r = r
            geometry.rectangle.z = z
            geometry.rectangle.width = width
            geometry.rectangle.height = height
        current = scale * (start + (flat - start) * ramp)
        coil.current.data = current
        # Inductive voltage while the current changes, resistive otherwise
        rate = np.gradient(current, t) if len(t) > 1 else np.zeros(len(t))
        coil.voltage.data = 0.02 * rate + 1.0e-3 * current
    return pf


def _switch_on(t: np.ndarray, t_on: float) -> np.ndarray:
    """0 before ``t_on``, rising to 1 within a second."""
    return 0.5 + 0.5 * np.tanh((t - t_on - 0.5) / 0.2)


def _identify(source, name: str) -> None:
    """Set the identifier of a core source from the Data Dictionary's list."""
    from imas.ids_identifiers import identifiers

    member = identifiers.core_source_identifier[name]
    source.identifier.name = member.name
    source.identifier.index = member.index
    source.identifier.description = member.description


def core_sources(factory, scale: float = 1.0, n_time: int = N_TIME):
    """core_sources: a neutral beam from 3 s, electron cyclotron waves from 5 s,
    ohmic heating that fades once the current has ramped up, and the collisional
    exchange from the electrons to the ions. Each has its profiles and totals."""
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    cs = factory.core_sources()
    cs.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    t = np.linspace(0.0, 10.0, n_time)
    cs.time = t
    rho = np.linspace(0.0, 1.0, N_RHO)
    ramp = np.minimum(t / 2.0, 1.0)
    on = {
        "nbi": scale * _switch_on(t, 3.0),
        "ec": scale * _switch_on(t, 5.0),
        "ohmic": scale * ramp * (0.3 + 0.7 * np.exp(-np.maximum(t - 2.0, 0.0) / 2.0)),
        "collisional_equipartition": scale * ramp * (0.2 + 0.8 * _switch_on(t, 3.0)),
    }

    def gauss(center: float, width: float) -> np.ndarray:
        return np.exp(-(((rho - center) / width) ** 2))

    # Profiles at full power: heating of the electrons and the ions [W m^-3],
    # particles [m^-3 s^-1], current [A m^-2] and torque [N m^-2] densities
    shapes = {
        "nbi": dict(electrons=0.25e6 * gauss(0.0, 0.45), ions=0.45e6 * gauss(0.0, 0.4),
                    particles=1.5e19 * gauss(0.0, 0.5), current=0.15e6 * gauss(0.0, 0.35),
                    torque=0.8 * gauss(0.0, 0.45)),
        "ec": dict(electrons=1.2e6 * gauss(0.3, 0.06), current=0.25e6 * gauss(0.32, 0.05)),
        "ohmic": dict(electrons=0.15e6 * (1.0 - rho**2) ** 3,
                      current=1.2e6 * (1.0 - rho**2) ** 1.5),
        "collisional_equipartition": dict(electrons=-0.1e6 * (1.0 - rho**2) ** 2,
                                          ions=0.1e6 * (1.0 - rho**2) ** 2),
    }
    # Totals at full power: power to the electrons and to the ions [W], driven
    # current [A] and torque [N m]
    totals = {
        "nbi": dict(electrons=2.0e6, ions=3.0e6, current=0.15e6, torque=4.0),
        "ec": dict(electrons=3.0e6, current=0.1e6),
        "ohmic": dict(electrons=1.5e6, current=0.8e6),
        "collisional_equipartition": dict(electrons=-0.5e6, ions=0.5e6),
    }
    cs.source.resize(len(on))
    for source, name in zip(cs.source, on):
        _identify(source, name)
        amplitude, shape, total = on[name], shapes[name], totals[name]
        source.profiles_1d.resize(n_time)
        source.global_quantities.resize(n_time)
        for i in range(n_time):
            p1d = source.profiles_1d[i]
            p1d.time = t[i]
            p1d.grid.rho_tor_norm = rho
            p1d.grid.psi = 20.0 * rho**2
            a = amplitude[i]
            p1d.electrons.energy = a * shape["electrons"]
            if "ions" in shape:
                p1d.total_ion_energy = a * shape["ions"]
            if "particles" in shape:
                p1d.electrons.particles = a * shape["particles"]
            if "current" in shape:
                p1d.j_parallel = a * shape["current"]
            if "torque" in shape:
                _set_first(p1d, ("momentum_phi", "momentum_tor"), a * shape["torque"])
            gq = source.global_quantities[i]
            gq.time = t[i]
            gq.electrons.power = a * total["electrons"]
            gq.total_ion_power = a * total.get("ions", 0.0)
            gq.power = a * (total["electrons"] + total.get("ions", 0.0))
            if "current" in total:
                gq.current_parallel = a * total["current"]
            if "torque" in total:
                _set_first(gq, ("torque_phi", "torque_tor"), a * total["torque"])
    return cs


def wall(factory):
    """wall: a D-shaped first wall around the plasma, as one limiter unit."""
    from imas.ids_defs import IDS_TIME_MODE_INDEPENDENT

    w = factory.wall()
    w.ids_properties.homogeneous_time = IDS_TIME_MODE_INDEPENDENT
    w.description_2d.resize(1)
    w.description_2d[0].limiter.unit.resize(1)
    unit = w.description_2d[0].limiter.unit[0]
    unit.name = "first wall"
    theta = np.linspace(0.0, 2.0 * np.pi, 121)[:-1]  # closed: the first point is not repeated
    unit.outline.r = 1.8 + 0.72 * np.cos(theta + 0.35 * np.sin(theta))
    unit.outline.z = 1.15 * np.sin(theta)
    return w


def summary(factory, scale: float = 1.0, n_time: int = N_TIME):
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    summ = factory.summary()
    summ.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    t = np.linspace(0.0, 10.0, n_time)
    summ.time = t
    summ.global_quantities.ip.value = scale * 1.0e6 * np.minimum(t / 2.0, 1.0)
    return summ


def write_entry(uri: str, scale: float = 1.0, dd_version: Optional[str] = None,
                rho_tor_norm: bool = True) -> None:
    """Write all demo IDSs to ``uri`` (a netCDF path or an IMAS URI).

    Args:
        rho_tor_norm: fill the rho_tor_norm grids of core_profiles and equilibrium.
    """
    import imas

    kwargs = {"dd_version": dd_version} if dd_version else {}
    with imas.DBEntry(uri, "w", **kwargs) as entry:
        factory = entry.factory
        for ids in (core_profiles(factory, scale, rho_tor_norm=rho_tor_norm),
                    equilibrium(factory, scale, rho_tor_norm=rho_tor_norm),
                    magnetics(factory, scale), summary(factory, scale),
                    pf_active(factory, scale), wall(factory),
                    core_sources(factory, scale)):
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
        contours: 10
        signals:
          - equilibrium/time_slice/profiles_2d[0]/psi
          - path: equilibrium/time_slice/boundary/outline
            label: boundary
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
