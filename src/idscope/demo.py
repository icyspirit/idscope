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
                  rho_tor_norm: bool = True, time: Optional[float] = None):
    """core_profiles with homogeneous time, electrons and two ion species.

    The second ion species is missing in the last three time slices.

    Args:
        rho_tor_norm: fill ``profiles_1d/grid/rho_tor_norm`` (profiles can otherwise
            be drawn against the normalized psi).
        time: write this single time instead, like a code run for one time (with
            both ion species).
    """
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    cp = factory.core_profiles()
    cp.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    cp.ids_properties.comment = "idscope synthetic demo data"
    t = np.linspace(0.0, 10.0, n_time) if time is None else np.array([time])
    n_time = len(t)
    cp.time = t
    rho = np.linspace(0.0, 1.0, N_RHO)
    te0 = scale * _profile_scale(t)
    ramp = np.minimum(t / 2.0, 1.0)
    beam = scale * _switch_on(t, 3.0)
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
        n_ion = 2 if i < n_time - 3 or time is not None else 1
        p1d.ion.resize(n_ion)
        for j, ion in enumerate(p1d.ion):
            _set_first(ion, ("name", "label"), ("D", "C")[j])
            ion.z_ion = 1.0 + j
            ion.temperature = (0.8 - 0.2 * j) * te0[i] * shape + 40.0
            if j == 0 and beam[i] > 1e-3:  # the fast deuterons of the beam
                fast = beam[i] * (1.0 - rho**2) ** 2
                ion.density_fast = 2.0e18 * fast
                ion.pressure_fast_parallel = 3.0e3 * fast
                ion.pressure_fast_perpendicular = 5.0e3 * fast
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
                rho_tor_norm: bool = True, time: Optional[float] = None):
    """equilibrium with scalars, 1D profiles and a rectangular (R, Z) psi map.

    Args:
        rho_tor_norm: fill ``profiles_1d/rho_tor_norm`` (profiles can otherwise be
            drawn against the normalized psi).
        time: write this single time slice instead, like a free-boundary solver.
    """
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    eq = factory.equilibrium()
    eq.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    t = np.linspace(0.0, 10.0, n_time) if time is None else np.array([time])
    n_time = len(t)
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
        # An X-point at the bottom of the boundary, and where a free-boundary
        # solver was asked to put it
        x_point = (r0 - 0.15, -0.5 * kappa)
        if hasattr(ts.boundary, "x_point"):  # Data Dictionary 3
            ts.boundary.x_point.resize(1)
            ts.boundary.x_point[0].r, ts.boundary.x_point[0].z = x_point
        else:  # Data Dictionary 4: the critical points of the contour tree, the
            # magnetic axis (here a minimum of psi) first and the X-point (a saddle)
            ts.contour_tree.node.resize(2)
            axis = (gq.magnetic_axis.r.value, gq.magnetic_axis.z.value)
            for node, critical_type, (r_node, z_node) in zip(
                ts.contour_tree.node, (0, 1), (axis, x_point)
            ):
                node.critical_type = critical_type
                node.r, node.z = r_node, z_node
        ts.constraints.x_point.resize(1)
        target = ts.constraints.x_point[0].position_measured
        target.r, target.z = x_point[0] - 0.02, x_point[1] - 0.04
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


def structured_grid(grid, axes) -> None:
    """Write a rectilinear GGD grid: a space of one dimension per axis, its nodes
    the points of the axis (as NuBDeC writes it). ``axes`` are pairs of a
    coordinate name (e.g. ``"r"``) and its points. Values on the nodes of such a
    grid run with the first axis fastest."""
    from imas.ids_identifiers import identifiers

    grid.identifier.name = "structured_spaces"
    grid.identifier.index = identifiers.ggd_identifier["structured_spaces"].index
    grid.space.resize(len(axes))
    for space, (coordinate, points) in zip(grid.space, axes):
        _coordinates(space, (coordinate,))
        space.objects_per_dimension.resize(1)
        nodes = space.objects_per_dimension[0]
        nodes.object.resize(len(points))
        for node, value in zip(nodes.object, points):
            node.geometry = [float(value)]
    grid.grid_subset.resize(1)
    grid.grid_subset[0].identifier.name = "nodes"
    grid.grid_subset[0].identifier.index = identifiers.ggd_subset_identifier["nodes"].index


def _coordinates(space, names) -> None:
    """The coordinates of a GGD space, e.g. ("r", "z"): identifiers in DD 4, their
    indices in DD 3."""
    from imas.ids_identifiers import identifiers

    members = [identifiers.coordinate_identifier[name] for name in names]
    if space.coordinates_type.metadata.data_type.name == "STRUCT_ARRAY":  # DD 4
        space.coordinates_type.resize(len(members))
        for kind, member in zip(space.coordinates_type, members):
            kind.name, kind.index = member.name, member.index
    else:
        space.coordinates_type = np.array([m.index for m in members], dtype=np.int32)


EDGE_TIMES = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
EDGE_GRID = (96, 12)
"""Cells of the edge grid of the demo: around the plasma, and across the edge."""


def unstructured_grid(grid, n_theta: int = EDGE_GRID[0], n_radial: int = EDGE_GRID[1]):
    """Write an unstructured GGD grid of quadrilateral cells in a ring around the
    plasma, from its boundary out to the first wall: one space in (R, Z), its nodes
    and its 2D cells (the nodes of each in order around it). The grid subset of the
    nodes is declared implicitly, that of the cells element by element.

    Returns the (R, Z) of the nodes, ``n_theta`` per ring from the inner ring out.
    """
    from imas.ids_identifiers import identifiers

    grid.identifier.name = "linear"
    grid.identifier.index = identifiers.ggd_identifier["linear"].index
    grid.space.resize(1)
    space = grid.space[0]
    _coordinates(space, ("r", "z"))
    space.objects_per_dimension.resize(3)  # nodes, (edges), cells
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)[:-1]
    # From the plasma boundary (a = 0.5 m, elongation 1.6, triangularity 0.3) to
    # the first wall
    rings = [(1.8 + (0.5 + 0.22 * f) * np.cos(theta + (0.3 + 0.05 * f) * np.sin(theta)),
              (0.8 + 0.35 * f) * np.sin(theta)) for f in np.linspace(0.0, 1.0, n_radial + 1)]
    points = np.array([(r, z) for ring in rings for r, z in zip(*ring)])
    nodes = space.objects_per_dimension[0].object
    nodes.resize(len(points))
    for node, point in zip(nodes, points):
        node.geometry = point
    cells = space.objects_per_dimension[2].object
    cells.resize(n_theta * n_radial)
    for k, cell in enumerate(cells):
        ring, j = divmod(k, n_theta)
        corners = [ring * n_theta + j, ring * n_theta + (j + 1) % n_theta]
        corners += [corner + n_theta for corner in corners[::-1]]
        cell.nodes = np.array(corners, dtype=np.int32) + 1  # Fortran indices
    grid.grid_subset.resize(2)
    grid.grid_subset[0].identifier.name = "nodes"
    grid.grid_subset[0].identifier.index = identifiers.ggd_subset_identifier["nodes"].index
    grid.grid_subset[0].dimension = 1
    cells_subset = grid.grid_subset[1]
    cells_subset.identifier.name = "cells"
    cells_subset.identifier.index = identifiers.ggd_subset_identifier["cells"].index
    cells_subset.dimension = 3  # 2D objects of a space in the plane
    cells_subset.element.resize(len(cells))
    for k, element in enumerate(cells_subset.element):
        element.object.resize(1)
        element.object[0].space, element.object[0].dimension = 1, 3
        element.object[0].index = k + 1
    return points


def edge_profiles(factory, scale: float = 1.0):
    """edge_profiles: the electron temperature on the cells of an unstructured grid
    around the plasma (:func:`unstructured_grid`), and the electron density on its
    nodes, falling off towards the wall and brighter on the outboard side."""
    from imas.ids_defs import IDS_TIME_MODE_HETEROGENEOUS
    from imas.ids_identifiers import identifiers

    edge = factory.edge_profiles()
    # One grid for all times: its time and those of the values are their own
    edge.ids_properties.homogeneous_time = IDS_TIME_MODE_HETEROGENEOUS
    edge.grid_ggd.resize(1)
    edge.grid_ggd[0].time = EDGE_TIMES[0]
    unstructured_grid(edge.grid_ggd[0])
    n_theta, n_radial = EDGE_GRID

    def falloff(depth, theta, t, length):
        """Down from the boundary (depth 0) to the wall (1), more on the outboard
        side (theta 0), and rising with the heating."""
        return (scale * (0.5 + 0.05 * t) * (1.0 + 0.4 * np.cos(theta))
                * np.exp(-depth / length))

    # the rings of nodes and of cells, and the angles around them
    node_depth, node_theta = np.meshgrid(np.linspace(0.0, 1.0, n_radial + 1),
                                         np.arange(n_theta) * 2 * np.pi / n_theta,
                                         indexing="ij")
    cell_depth, cell_theta = np.meshgrid((np.arange(n_radial) + 0.5) / n_radial,
                                         (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta,
                                         indexing="ij")
    edge.ggd.resize(len(EDGE_TIMES))
    for ggd, t in zip(edge.ggd, EDGE_TIMES):
        ggd.time = t
        ggd.electrons.temperature.resize(1)
        temperature = ggd.electrons.temperature[0]
        temperature.grid_index = 1
        temperature.grid_subset_index = identifiers.ggd_subset_identifier["cells"].index
        temperature.values = 100.0 * falloff(cell_depth, cell_theta, t, 0.3).ravel()
        ggd.electrons.density.resize(1)
        density = ggd.electrons.density[0]
        density.grid_index = 1
        density.grid_subset_index = identifiers.ggd_subset_identifier["nodes"].index
        density.values = 2.0e19 * falloff(node_depth, node_theta, t, 0.5).ravel()
    return edge


DISTRIBUTION_TIMES = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
NEUTRON_GRID = (np.linspace(1.2, 2.4, 25), np.linspace(-1.2, 1.2, 49))
"""R and Z of the grid of the neutron birth rates."""


def _neutron_births(t: np.ndarray, scale: float):
    """Birth rates of the neutrons of D-D fusion [m^-3 s^-1] on
    :data:`NEUTRON_GRID`, thermal and beam-thermal, at the times ``t``."""
    r, z = np.meshgrid(*NEUTRON_GRID, indexing="ij")
    inside = np.clip(1.0 - ((r - 1.8) / 0.55) ** 2 - (z / 0.95) ** 2, 0.0, None)
    ramp = np.minimum(t / 2.0, 1.0)
    beam = scale * _switch_on(t, 3.0)
    thermal = [scale**2 * 2.0e15 * a * inside**4 for a in ramp]
    beam_thermal = [scale * 8.0e15 * b * inside**2 * np.exp(-((z - 0.05) / 0.35) ** 2)
                    for b in beam]
    return {"thermal_thermal": thermal, "beam_thermal": beam_thermal}


def _total(rate: np.ndarray) -> float:
    """A birth rate on :data:`NEUTRON_GRID` summed over the torus [s^-1]."""
    r, z = NEUTRON_GRID
    volume = 2.0 * np.pi * r[:, None] * (r[1] - r[0]) * (z[1] - z[0])
    return float((rate * volume).sum())


def distribution_sources(factory, scale: float = 1.0, n_markers: int = 400):
    """distribution_sources, as a neutral beam code writes them (NuBDeC, for
    one): where the ions of the beam are born, as markers, and where the neutrons
    of D-D fusion are, as birth rates on a grid of R by Z, at
    :data:`DISTRIBUTION_TIMES`. The beam starts at 3 s."""
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS
    from imas.ids_identifiers import identifiers

    ds = factory.distribution_sources()
    ds.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    t = DISTRIBUTION_TIMES
    ds.time = t
    beam = scale * _switch_on(t, 3.0)
    rng = np.random.default_rng(7)
    births = _neutron_births(t, scale)
    ds.source.resize(1 + len(births))

    def process(source, name: str, energy: Optional[str] = None) -> None:
        source.process.resize(1)
        member = identifiers.distribution_source_identifier[name]
        kind = source.process[0].type
        kind.name, kind.index, kind.description = member.name, member.index, member.description
        if energy is not None:
            source.process[0].reactant_energy.name = energy
            source.process[0].reactant_energy.index = {"thermal_thermal": 1,
                                                       "beam_beam": 2,
                                                       "beam_thermal": 3}[energy]

    nbi = ds.source[0]
    process(nbi, "NBI")
    nbi.markers.resize(len(t))
    nbi.global_quantities.resize(len(t))
    columns = ("r", "z", "phi", "velocity_parallel", "velocity_perpendicular")
    for i, markers in enumerate(nbi.markers):
        markers.time = t[i]
        totals = nbi.global_quantities[i]
        totals.time = t[i]
        totals.particles = beam[i] * 3.0e20
        totals.power = beam[i] * 5.0e6
        if beam[i] < 1e-3:
            continue
        # Born along the beam, which comes in from the low field side
        r = np.clip(2.35 - rng.exponential(0.35, n_markers), 1.3, 2.35)
        z = rng.normal(0.05, 0.1, n_markers)
        markers.coordinate_identifier.resize(len(columns))
        for identifier, name in zip(markers.coordinate_identifier, columns):
            member = identifiers.coordinate_identifier[name]
            identifier.name, identifier.index = member.name, member.index
        markers.positions = np.column_stack([
            r, z, rng.uniform(0.0, 2.0 * np.pi, n_markers),
            rng.normal(1.5e6, 0.3e6, n_markers), rng.normal(0.8e6, 0.2e6, n_markers),
        ])
        markers.weights = np.full(n_markers, beam[i] * 3.0e20 / n_markers)

    for k, (energy, rates) in enumerate(births.items(), start=1):
        source = ds.source[k]
        process(source, "D_D_to_3He_n", energy)
        source.species.type.name = "neutron"
        source.species.type.index = 6
        source.ggd.resize(len(t))
        source.global_quantities.resize(len(t))
        for i, ggd in enumerate(source.ggd):
            ggd.time = t[i]
            structured_grid(ggd.grid, (("r", NEUTRON_GRID[0]), ("z", NEUTRON_GRID[1])))
            ggd.particles.resize(1)
            ggd.particles[0].grid_index = 1
            ggd.particles[0].grid_subset_index = 1  # the nodes
            ggd.particles[0].values = rates[i].ravel(order="F")  # R running fastest
            source.global_quantities[i].time = t[i]
            source.global_quantities[i].particles = _total(rates[i])
    return ds


def _d_shape(a: float, b: float, theta: np.ndarray):
    """(R, Z) of a D-shaped contour of the demo machine: half-width ``a``, half-height
    ``b``, centred on R = 1.8 m."""
    return 1.8 + a * np.cos(theta + 0.35 * np.sin(theta)), b * np.sin(theta)


def wall(factory):
    """wall: a D-shaped first wall around the plasma, as one limiter unit, and the
    vacuum vessel around it, as one annular unit (its inner and outer contours)."""
    from imas.ids_defs import IDS_TIME_MODE_INDEPENDENT

    w = factory.wall()
    w.ids_properties.homogeneous_time = IDS_TIME_MODE_INDEPENDENT
    w.description_2d.resize(1)
    w.description_2d[0].limiter.unit.resize(1)
    unit = w.description_2d[0].limiter.unit[0]
    unit.name = "first wall"
    theta = np.linspace(0.0, 2.0 * np.pi, 121)[:-1]  # closed: the first point is not repeated
    unit.outline.r, unit.outline.z = _d_shape(0.72, 1.15, theta)
    w.description_2d[0].vessel.unit.resize(1)
    vessel = w.description_2d[0].vessel.unit[0]
    vessel.name = "vacuum vessel"
    for contour, (a, b) in ((vessel.annular.outline_inner, (0.8, 1.25)),
                            (vessel.annular.outline_outer, (0.85, 1.32))):
        contour.r, contour.z = _d_shape(a, b, theta)
        if hasattr(contour, "closed"):  # Data Dictionary 3
            contour.closed = 1
    return w


def pf_passive(factory, scale: float = 1.0, n_time: int = N_TIME):
    """pf_passive: two stabilizing plates inside the vessel, above and below the
    midplane on the outboard side, with the currents induced in them: by the ramp-up
    of the plasma current, and, in opposite directions, by the plasma moving up and
    down."""
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    pf = factory.pf_passive()
    pf.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    t = np.linspace(0.0, 10.0, n_time)
    pf.time = t
    pf.loop.resize(2)
    for loop, name, sign in zip(pf.loop, ("upper plate", "lower plate"), (1.0, -1.0)):
        loop.name = name
        loop.element.resize(1)
        geometry = loop.element[0].geometry
        geometry.geometry_type = 1  # an outline: the plate follows the wall
        theta = sign * np.linspace(0.3, 0.8, 11)
        inner, outer = _d_shape(0.745, 1.19, theta), _d_shape(0.775, 1.22, theta)
        geometry.outline.r = np.concatenate([inner[0], outer[0][::-1]])
        geometry.outline.z = np.concatenate([inner[1], outer[1][::-1]])
        loop.element[0].turns_with_sign = 1.0
        loop.current = scale * (-25.0e3 * np.exp(-(((t - 1.0) / 0.6) ** 2))
                                + sign * 3.0e3 * np.sin(3.7 * t))
    return pf


TF_COILS = 16
_S = np.sqrt(0.5)
TF_CONDUCTOR = [
    # name, type (1: line, 2: arc), start, intermediate, end, centre (R, Z) [m]
    ("inner leg", 1, (0.9, -1.5), None, (0.9, 1.5), None),
    ("upper inner arc", 2, (0.9, 1.5), (1.33 - 0.43 * _S, 1.5 + 0.43 * _S), (1.33, 1.93),
     (1.33, 1.5)),
    ("top", 1, (1.33, 1.93), None, (2.3, 1.93), None),
    ("upper outer arc", 2, (2.3, 1.93), (2.3 + 0.83 * _S, 1.1 + 0.83 * _S), (3.13, 1.1),
     (2.3, 1.1)),
    ("outer leg", 1, (3.13, 1.1), None, (3.13, -1.1), None),
    ("lower outer arc", 2, (3.13, -1.1), (2.3 + 0.83 * _S, -1.1 - 0.83 * _S), (2.3, -1.93),
     (2.3, -1.1)),
    ("bottom", 1, (2.3, -1.93), None, (1.33, -1.93), None),
    ("lower inner arc", 2, (1.33, -1.93), (1.33 - 0.43 * _S, -1.5 - 0.43 * _S), (0.9, -1.5),
     (1.33, -1.5)),
]
"""The center line of the conductor of a toroidal field coil of the demo machine:
a D of straight legs and arcs, around the vessel and the PF coils."""


def tf(factory, scale: float = 1.0, n_time: int = N_TIME):
    """tf: the toroidal field coils (one standing for all, :data:`TF_COILS` of
    them), their current and the vacuum field they make."""
    from imas.ids_defs import EMPTY_FLOAT, IDS_TIME_MODE_HOMOGENEOUS

    coils = factory.tf()
    coils.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    t = np.linspace(0.0, 10.0, n_time)
    coils.time = t
    coils.r0 = 1.8
    coils.is_periodic = 1
    coils.coils_n = TF_COILS
    # R B_phi = mu0 N I / (2 pi), for N I of all the coils together
    current = scale * 11.25e3 * np.ones(len(t))  # per turn, of 100 turns
    field = 2.0e-7 * TF_COILS * 100 * current
    vacuum = (coils.b_field_phi_vacuum_r if hasattr(coils, "b_field_phi_vacuum_r")
              else coils.b_field_tor_vacuum_r)  # Data Dictionary 4, 3
    vacuum.data = field
    coils.coil.resize(1)
    coil = coils.coil[0]
    coil.name = "TF"
    coil.turns = 100.0
    coil.current.data = current
    coil.conductor.resize(1)
    elements = coil.conductor[0].elements
    if hasattr(elements, "names"):  # not in every version of the Data Dictionary
        elements.names = [name for name, *_ in TF_CONDUCTOR]
    elements.types = np.array([kind for _, kind, *_ in TF_CONDUCTOR], dtype=np.int32)

    def column(points, k):
        return np.array([EMPTY_FLOAT if p is None else p[k] for p in points])

    for name, index in (("start_points", 2), ("intermediate_points", 3),
                        ("end_points", 4), ("centres", 5)):
        points = [row[index] for row in TF_CONDUCTOR]
        node = getattr(elements, name)
        node.r, node.z = column(points, 0), column(points, 1)
        node.phi = np.array([EMPTY_FLOAT if p is None else 0.0 for p in points])
    return coils


def summary(factory, scale: float = 1.0, n_time: int = N_TIME,
            time: Optional[float] = None):
    """summary: the plasma current and the neutron rates (``time``: write this
    single time instead)."""
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    summ = factory.summary()
    summ.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    t = np.linspace(0.0, 10.0, n_time) if time is None else np.array([time])
    summ.time = t
    summ.global_quantities.ip.value = scale * 1.0e6 * np.minimum(t / 2.0, 1.0)
    rates = {energy: np.array([_total(rate) for rate in values])
             for energy, values in _neutron_births(t, scale).items()}
    dd = summ.fusion.neutron_rates.dd
    dd.thermal.value = rates["thermal_thermal"]
    dd.beam_thermal.value = rates["beam_thermal"]
    dd.total.value = rates["thermal_thermal"] + rates["beam_thermal"]
    # The tritons of D-D fusion burn with deuterons: about 1 % as many neutrons
    summ.fusion.neutron_rates.dt.total.value = 0.01 * dd.total.value
    summ.fusion.neutron_rates.total.value = 1.01 * dd.total.value
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
                    pf_active(factory, scale), wall(factory), pf_passive(factory, scale),
                    tf(factory, scale), core_sources(factory, scale),
                    distribution_sources(factory, scale), edge_profiles(factory, scale)):
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
