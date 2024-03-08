#! /usr/bin/env python
# coding: utf-8

# # Evolve Island World
#
# *(Greg Tucker, University of Colorado Boulder)*
#
# Demonstration of a Landlab-built simulation of the morphological evolution of
# a hypothetical island micro-continent.
#

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from landlab import ModelGrid, RasterModelGrid, create_grid, load_params
from landlab.components import (
    ErosionDeposition,
    Flexure,
    FlowAccumulator,
    ListricKinematicExtender,
    SimpleSubmarineDiffuser,
)
from landlab.io.native_landlab import load_grid, save_grid

from graphics import display_island
from igneous_intruder import IgneousIntruder


def merge_user_and_default_params(user_params, default_params):
    """Merge default parameters into the user-parameter dictionary, adding
    defaults where user values are absent.

    Examples
    --------
    >>> u = {"a": 1, "d": {"da": 4}, "e": 5, "grid": {"RasterModelGrid": []}}
    >>> d = {"a": 2, "b": 3, "d": {"db": 6}, "grid": {"HexModelGrid": []}}
    >>> merge_user_and_default_params(u, d)
    >>> u["a"]
    1
    >>> u["b"]
    3
    >>> u["d"]
    {'da': 4, 'db': 6}
    >>> u["grid"]
    {'RasterModelGrid': []}
    """
    for k in default_params.keys():
        if k in default_params:
            if k not in user_params.keys():
                user_params[k] = default_params[k]
            elif isinstance(user_params[k], dict) and k != "grid":
                merge_user_and_default_params(user_params[k], default_params[k])


def get_or_create_node_field(grid, name, dtype="float64"):
    """Get handle to a grid field if it exists, otherwise create it."""
    try:
        return grid.at_node[name]
    except KeyError:
        return grid.add_zeros(name, at="node", dtype=dtype, clobber=True)


class IslandSimulator:
    """Simulate geologic evolution of an island or micro-continent."""

    ALL_PROCESSES = (
        "flexure",
        "fluvial",
        "hillslope",
        "runoff",
        "sea_level",
        "submarine_diffusion",
        "tectonic_extension",
    )

    DEFAULT_GRID_PARAMS = {
        "source": "file",  # "create", "file", or "grid_object"
        "grid_file_name": "initial_island.grid",
        "grid_object": None,
        "grid": {
            "HexModelGrid": [
                {"shape": [51, 51]},
                {"spacing": 1000.0},
                {"xy_spacing": 1000.0},
                {"node_layout": "rect"},
                {"node_orientation": "horiz"},
                {"fields": None},
                {"boundary_conditions": None},
            ],
        },
    }

    DEFAULT_PROCESS_PARAMS = {
        "flexure": {
            "method": "flexure",
            "eet": 1.0e4,
            "rho_mantle": 3300.0,
        },
        "fluvial": {
            "K": 1.0e-5,
            "v_s": 1.0,
        },
        "hillslope:": {},
        "runoff": {},
        "sea_level": {
            "source": "random",
            "sea_level_file_name": None,
            "sea_level_delta": 0.4,
            "amplitude": 0.0,
            "period": 100000.0,
            "drift": 0.0,
        },
        "submarine_diffusion": {
            "sea_level": 0.0,
            "wave_base": 60.0,
            "shallow_water_diffusivity": 100.0,
            "tidal_range": 2.0,
        },
        "tectonic_extension": {
            "extension_rate_x": 0.01,
            "extension_rate_y": 0.0,
            "fault_dip": 60.0,
            "fault_x0": 4.0e4,
            "fault_y0": 0.0,
            "fault_strike": 60.0,
            "detachment_depth": 1.0e4,
        },
        "igneous_intrusion": {
            "episode_duration": 1.0e5,
            "episode_start_time": 0.0,
            "mean_waiting_time": 1.0e4,
            "thickness_min": 100.0,
            "thickness_max": 1000.0,
            "radius_min": 1000.0,
            "radius_max": 10000.0,
            "log_duration_min": 4.0,
            "log_duration_max": 6.0,
            "depth_min": 1000.0,
            "depth_max": 20000.0,
        },
        "other": {
            "crust_datum": -1.5e4,
            "unit_weight": 2650.0 * 9.8,
            "fault_jump": {
                "jump_interval_mean": 40000.0,
                "jump_interval_std": 4000.0,
                "jump_distance_mean": 5000.0,
                "jump_distance_std": 500.0,
            },
            "extension_duration": 3.0e6,
        },
    }

    DEFAULT_RUN_PARAMS = {
        "random_seed": 1,
        "dt": 100.0,  # time-step duration, y
        "run_duration": 250000.0,  # duration of run, y
        "start_time": 0.0,  # starting time of simulation, y
        "click_years_per_sec": 0.0,  # click rate in sim yrs per clock sec (0 = max)
    }

    DEFAULT_OUTPUT_PARAMS = {
        "plot_interval": 2000.0,  # time interval for plotting, y
        "save_interval": 25000.0,  # time interval for saving grid, y
        "save_name": "rift-island-save",
        "ndigits": 3,  # number of digits for output files
        "max_elev_for_color_scale": 1650.0,  # elevation for color scale in plotting, m
        "scale_fac_for_surface_water": 0.3,  # surface water color = -this x above scale
        "area_threshold": 5.0e7,  # minimum drainage area for displayed streams, m2
        "path_for_latest_image": ".",  # file path for location of latest output image
    }

    def __init__(
        self, grid_params={}, process_params={}, run_params={}, output_params={}
    ):
        """Initialize IslandSimulator."""

        merge_user_and_default_params(grid_params, self.DEFAULT_GRID_PARAMS)
        merge_user_and_default_params(process_params, self.DEFAULT_PROCESS_PARAMS)
        merge_user_and_default_params(run_params, self.DEFAULT_RUN_PARAMS)
        merge_user_and_default_params(output_params, self.DEFAULT_OUTPUT_PARAMS)
        np.random.seed(run_params["random_seed"])

        self.setup_grid(grid_params)
        if isinstance(self.grid, RasterModelGrid):
            self.flexure_grid = self.grid
        else:
            self.create_raster_grid_for_flexure()
        self.setup_fields()
        self.setup_for_output(output_params)
        self.setup_sea_level(process_params["sea_level"])
        self.instantiate_components(process_params)
        self.setup_run_control(run_params)
        self.setup_for_flexure_and_tectonics(process_params["other"])
        self.setup_for_igneous_intrusion(process_params["igneous_intrusion"])

        self.fa.run_one_step()  # update surface water for display
        procs = "Sea-level change"
        if self._extension_duration > 0.0:
            procs += ", tectonic extension"
        display_island(
            self.grid,
            self.sea_level_history,
            self.mean_elevation_history,
            self.dt,
            0,
            output_params,
            sim_time=0.0,
            procs=procs,
        )

    def setup_grid(self, grid_params):
        """Load or create the grid.

        Examples
        --------
        >>> p = {"source": "create"}
        >>> p["grid"] = {"RasterModelGrid": [(4, 5)]}
        >>> sim = IslandSimulator(grid_params=p)
        >>> sim.grid.shape
        (4, 5)
        >>> from landlab.io.native_landlab import load_grid, save_grid
        >>> _ = sim.grid.at_node.pop("cumulative_subsidence_depth")
        >>> save_grid(sim.grid, "test.grid", clobber=True)
        >>> p = {"source": "file"}
        >>> p["grid_file_name"] = "test.grid"
        >>> sim = IslandSimulator(grid_params=p)
        >>> sim.grid.shape
        (4, 5)
        >>> from landlab import RasterModelGrid
        >>> p = {"source": "grid_object"}
        >>> p["grid_object"] = RasterModelGrid((3, 3))
        >>> sim = IslandSimulator(grid_params=p)
        >>> sim.grid.shape
        (3, 3)
        >>> from numpy.testing import assert_raises
        >>> p["grid_object"] = "spam"
        >>> assert_raises(ValueError, IslandSimulator, p)
        grid_object must be a Landlab grid.
        """
        if grid_params["source"] == "create":
            self.grid = create_grid(grid_params, section="grid")
        elif grid_params["source"] == "file":
            self.grid = load_grid(grid_params["grid_file_name"])
        elif grid_params["source"] == "grid_object":
            if isinstance(grid_params["grid_object"], ModelGrid):
                self.grid = grid_params["grid_object"]
            else:
                print("grid_object must be a Landlab grid.")
                raise ValueError
        self.interior_nodes = self.grid.status_at_node == self.grid.BC_NODE_IS_CORE

    def setup_fields(self):
        """Get handles to various fields, creating them as needed."""
        self.elev = get_or_create_node_field(self.grid, "topographic__elevation")
        self.sed = get_or_create_node_field(self.grid, "soil__depth")
        self.wse = get_or_create_node_field(self.grid, "water_surface__elevation")
        self.is_submarine = get_or_create_node_field(
            self.grid, "is_submarine", dtype="bool"
        )
        self.cum_depo = get_or_create_node_field(
            self.grid, "cumulative_deposit_thickness"
        )
        self.thickness = get_or_create_node_field(self.grid, "upper_crust__thickness")
        self.fw_thickness = get_or_create_node_field(self.grid, "footwall__thickness")
        self.load = get_or_create_node_field(
            self.flexure_grid, "lithosphere__overlying_pressure_increment"
        )
        self.elev_before_ero_dep = np.zeros(self.grid.number_of_nodes)

    def create_raster_grid_for_flexure(self):
        """Create a raster grid for flexure, if the main grid isn't raster.

        Currently assumes main grid is hex with uniform spacing.

        Examples
        --------
        >>> p = {"source": "create"}
        >>> p["grid"] = {"HexModelGrid": [(3, 3), {"node_layout": "rect"}]}
        >>> sim = IslandSimulator(grid_params=p)
        >>> sim.flexure_grid.y_of_node.reshape((3, 3))
        array([[ 0.   ,  0.   ,  0.   ],
               [ 0.866,  0.866,  0.866],
               [ 1.732,  1.732,  1.732]])
        """
        self.flexure_grid = RasterModelGrid(
            (self.grid.number_of_node_rows, self.grid.number_of_node_columns),
            xy_spacing=(self.grid.spacing, 0.866 * self.grid.spacing),
        )

    def setup_sea_level(self, params):
        """Setup variables related to varying sea level."""
        self.current_sea_level = 0.0
        self.sea_level_history = [self.current_sea_level]
        self.mean_elevation_history = [np.mean(self.elev[self.elev > 0.0])]
        self._sea_level_option = params["source"][0].lower()
        self._sea_level_delta = params["sea_level_delta"]
        if self._sea_level_option == "r":  # "random" option
            self._sea_level_amplitude = params["amplitude"]
            self._sea_level_period = params["period"]
            self._prior_base_sl = 0.0
        elif self._sea_level_option == "f":  # file option
            self._sl_curve = np.loadtxt(params["sea_level_file_name"], delimiter=",")
        self._sea_level_drift = params["drift"]

    def setup_for_output(self, params):
        """Setup variables for control of plotting and saving."""
        self.plot_interval = params["plot_interval"]
        self.next_plot = self.plot_interval
        self.save_interval = params["save_interval"]
        self.next_save = self.save_interval
        self.ndigits = params["ndigits"]
        self.frame_num = 0  # current output image frame number
        self.save_num = 0  # current save file frame number
        self.save_name = params["save_name"]
        self.display_params = params

    def instantiate_components(self, params):
        """Instantiate and initialize process components."""

        self.fa = FlowAccumulator(
            self.grid,
            depression_finder="LakeMapperBarnes",
            fill_surface=self.wse,
            redirect_flow_steepest_descent=True,
            reaccumulate_flow=True,
        )

        self.sp = ErosionDeposition(self.grid, **params["fluvial"])

        self.sd = SimpleSubmarineDiffuser(
            self.grid,
            **params["submarine_diffusion"],
        )

        self.ke = ListricKinematicExtender(self.grid, **params["tectonic_extension"])

        # temporary hack: keep tectonic params
        self._tec_params = params["tectonic_extension"].copy()

        self.fl = Flexure(self.flexure_grid, **params["flexure"])

    def setup_for_flexure_and_tectonics(self, params):
        """Initialize variables for flexure and calculate initial deflection."""
        self.crust_datum = params["crust_datum"]
        self.unit_weight = params["unit_weight"]
        self.thickness[:] = self.elev - self.crust_datum
        self.load[:] = self.unit_weight * self.thickness
        self.fl.update()
        self.deflection = self.flexure_grid.at_node[
            "lithosphere_surface__elevation_increment"
        ]
        self._extension_duration = params["extension_duration"]
        self.init_deflection = self.deflection.copy()

        # tectonics needs footwall and hangingwall thickness.
        # hangingwall should already be set by the ListricKinematicExtender.
        self.fault_plane_elev = self.grid.at_node["fault_plane__elevation"]
        self.hw_thickness = self.grid.at_node["hangingwall__thickness"]
        self.fw_thickness = self.thickness - self.hw_thickness

        # fault jumping
        fj_params = params["fault_jump"]
        if fj_params["jump_interval_mean"] > 0.0:
            self._fault_jump_interval_mean = fj_params["jump_interval_mean"]
            self._fault_jump_interval_std = fj_params["jump_interval_std"]
            self._fault_jump_distance_mean = fj_params["jump_distance_mean"]
            self._fault_jump_distance_std = fj_params["jump_distance_std"]
            self._next_fault_jump = (
                self.current_time
                + self._fault_jump_interval_mean
                + self._fault_jump_interval_std * np.random.randn()
            )
        else:
            self._next_fault_jump = 0.0  # indicator that fault is not jumping

        # for tracking purposes
        self.init_thickness = self.thickness.copy()

    def setup_run_control(self, params):
        """Initialize variables related to control of run timing."""
        self.run_duration = params["run_duration"]
        self.dt = params["dt"]
        self.current_time = params["start_time"]
        self.click_years_per_sec = params["click_years_per_sec"]
        self.tec_counter = 0
        self.tec_interval = 4

    def setup_for_igneous_intrusion(self, params):
        self.intruder = IgneousIntruder(
            self.grid,
            contacts=[self.elev, self.fault_plane_elev],
            layers=[self.hw_thickness, self.fw_thickness],
            **params,
        )

    def set_boundaries_for_subaerial(self):
        """Identify subaerial vs. marine nodes, and set marine to open-boundary
        status.

        This causes subaerial flow routing and fluvial erosion/deposition to act
        only on nodes above the current sea level.
        """
        self.is_submarine[:] = self.elev <= self.current_sea_level
        self.grid.status_at_node[self.is_submarine] = self.grid.BC_NODE_IS_FIXED_VALUE
        self.grid.status_at_node[
            np.invert(self.is_submarine)
        ] = self.grid.BC_NODE_IS_CORE

    def set_boundaries_for_full_domain(self):
        """Set all interior nodes to core-node status.

        This undoes set_boundaries_for_subaerial, in preparation for updating
        submarine erosion, transport, and deposition.
        """
        self.grid.status_at_node[self.interior_nodes] = self.grid.BC_NODE_IS_CORE

    def _handle_fault_jump(self):
        """Implement a jump in the position of the fault plane."""
        jump_distance = (
            self._fault_jump_distance_mean
            + self._fault_jump_distance_std * np.random.randn()
        )
        new_strike = self._tec_params["fault_strike"] + 5 * np.random.randn()

        self._tec_params["fault_x0"] += jump_distance
        self.ke.update_fault_plane_elevation_and_hangingwall_thickness(
            self.grid,
            self._tec_params["fault_x0"],
            self._tec_params["fault_y0"],
            new_strike,
            self._tec_params["fault_dip"],
            self._tec_params["detachment_depth"],
        )
        self.hw_thickness[:] = (
            self.elev
            + (self.deflection - self.init_deflection)  # remove deflection
            - self.fault_plane_elev
        )
        self.fw_thickness[:] = self.fault_plane_elev - self.crust_datum
        self.thickness[:] = self.fw_thickness + self.hw_thickness
        self._next_fault_jump += (
            self._fault_jump_interval_mean
            + self._fault_jump_interval_std * np.random.randn()
        )
        print("FAULT JUMP BY", jump_distance, "km; strike", new_strike)

    def update_tectonics_and_flexure(self, dt):
        """Update tectonics and flexure.

        Run the tectonic component to implement tectonic motion for one time
        interval of duration dt. Update the crustal load, then use the updated
        load to calculate isostatic deflection (flexure). Recalculate elevation
        by summing the original crustal datum, isostatic deflection (calculated
        as the difference between current deflection and the deflection from the
        initial configuration, so the latter is effectively zero), the crustal
        thickness field, and cumulative tectonic subsidence calculated by the
        tectonic component.
        """
        if self.current_time < self._extension_duration:
            if (
                self._next_fault_jump > 0.0
                and self.current_time >= self._next_fault_jump
            ):
                self._handle_fault_jump()
            self.ke.run_one_step(dt)  # update extensional subsidence

        self.thickness[:] = self.fw_thickness + self.hw_thickness
        self.load[self.grid.core_nodes] = self.unit_weight * (
            self.thickness[self.grid.core_nodes]
        )
        self.fl.update()  # update flexure
        self.elev[:] = (
            self.crust_datum + self.thickness - (self.deflection - self.init_deflection)
        )

    def update_sea_level(self, current_time):
        """Update sea level by adding a random increment."""
        if self._sea_level_option == "r":  # "random"
            base_sl = self._sea_level_amplitude * np.sin(
                2.0 * np.pi * current_time / self._sea_level_period
            )
            dsl = base_sl - self._prior_base_sl
            self.current_sea_level += (
                self._sea_level_delta * np.random.randn() + dsl + self._sea_level_drift
            )
            self._prior_base_sl = base_sl
        elif self._sea_level_option == "f":  # from file
            self.current_sea_level = (
                np.interp(current_time, self._sl_curve[:, 0], self._sl_curve[:, 1])
                + self._sea_level_delta * np.random.randn()
                - self._sea_level_drift * current_time
            )
            print(current_time, self.current_sea_level)
        self.sea_level_history.append(self.current_sea_level)
        mean_rel_elev = self.elev - self.current_sea_level
        self.mean_elevation_history.append(np.mean(mean_rel_elev[mean_rel_elev > 0.0]))

    def update_subaerial_processes(self, dt):
        """Run subaerial flow routing and fluvial processes."""
        self.fa.run_one_step()
        self.sp.run_one_step(dt)

    def deposit_river_sediment_at_coast(self, dt):
        """Calculate deposition of river sediment along the coasts.

        Assumes that a side effect of the fluvial component will be a calculation
        of fluvial volumetric sediment influx to each node, including open-boundary
        nodes that receive flow and sediment from adjacent core nodes. The
        resulting deposit thickness at each submarine "coastal" node is the
        volume sediment inflow rate times time-step duration divided by cell area.

        TODO: might also be good to use
        area of all cells not just one representative (which assumes uniform cell
        area)
        """
        # algorithm to spread sediment offshore of river mouths
        qs_in = self.grid.at_node["sediment__influx"]
        outlet_nodes = np.where(np.logical_and(self.is_submarine, qs_in > 0.0))[0]
        depo = (
            dt * qs_in[outlet_nodes] / self.grid.area_of_cell[0]
        )  # depth deposited this time step
        outlets_sorted = np.flip(
            np.argsort(qs_in[outlet_nodes])
        )  # do the biggest first
        for i in outlets_sorted:
            node = outlet_nodes[i]
            wdepth = self.current_sea_level - self.elev[node]
            local_dep = min(depo[i], wdepth)
            remaining_dep = depo[i] - local_dep
            self.elev[node] += local_dep  # drop it here
            counter = 0
            while (
                remaining_dep > 0.0
            ):  # as long as there's any left, move it to the deepest adjacent node
                nbrs = self.grid.adjacent_nodes_at_node[node]
                target = nbrs[np.argmin(self.elev[nbrs])]
                if self.elev[target] >= self.elev[node]:
                    local_dep = remaining_dep
                else:
                    local_dep = min(
                        remaining_dep, self.current_sea_level - self.elev[target]
                    )
                if local_dep == 0.0:
                    local_dep = remaining_dep
                self.elev[target] += local_dep
                remaining_dep -= local_dep
                node = target
                counter += 1
                if counter > 100:
                    print("INF LOOP!")
                    break

    def deposit_river_sediment_at_coast2(self, dt):
        """Calculate deposition of river sediment along the coasts."""
        # algorithm to spread sediment offshore of river mouths
        qs_in = self.grid.at_node["sediment__influx"]
        water_depth = self.current_sea_level - self.elev
        self.is_submarine[:] = water_depth > 0.0
        marine_nodes_in_order = self.grid.at_node["flow__upstream_node_order"][
            self.is_submarine[self.grid.at_node["flow__upstream_node_order"]]
        ]
        rcvr = self.grid.at_node["flow__receiver_node"]
        min_depth_increment = 0.1
        min_depth = min_depth_increment + np.zeros(self.grid.number_of_nodes)
        cellarea = self.grid.area_of_cell[0]  # assumes all cells same area
        counter = 0
        for node in marine_nodes_in_order:
            if (
                qs_in[node] > 0.0
                and self.grid.status_at_node[node] == self.grid.BC_NODE_IS_CORE
            ):
                counter += 1
                sed_depth = qs_in[node] * dt / cellarea
                max_depo_depth = max(water_depth[node] - min_depth[node], 0.0)
                depo_depth = min(sed_depth, max_depo_depth)
                self.elev[node] += depo_depth
                leftover_depth = sed_depth - depo_depth
                if leftover_depth > 0.0:
                    # print("passing", leftover_depth, "from", node, "to", rcvr[node])
                    qs_in[rcvr[node]] += leftover_depth * cellarea / dt
                    min_depth[rcvr[node]] += min_depth_increment
        qs_in[self.is_submarine] = 0.0

    def update_submarine_processes(self, dt):
        """Run the submarine diffusion component for one time step dt."""
        self.sd.sea_level = self.current_sea_level
        self.sd.run_one_step(dt)

    def update_deposit_and_crust_thickness(self, dt, elev_before):
        """Update the crustal thickness and the cumulative deposit thickness.

        (TODO: Note that cum depo might no longer be needed if a fluvial
        component that tracks sediment is used)
        """
        dz = (
            self.elev[self.grid.core_nodes]
            - self.elev_before_ero_dep[self.grid.core_nodes]
        )
        self.cum_depo[self.grid.core_nodes] += dz
        self.thickness[self.grid.core_nodes] += dz
        self.hw_thickness[:] = np.maximum(
            self.crust_datum + self.thickness - self.fault_plane_elev, 0.0
        )
        self.fw_thickness[:] = self.thickness - self.hw_thickness

    def update_igneous_intrusions(self, dt):
        if (
            self.current_time >= self.intruder.episode_start_time
            and self.current_time < self.intruder.last_stop_time
        ):
            self.intruder.update(dt, self.current_time)
            self.thickness[:] = self.fw_thickness + self.hw_thickness

    def update(self, dt):
        """Run all components for one time step of duration dt."""
        # print("before mean hwt, fwt, tt, elev, fpe",
        #      np.mean(self.hw_thickness),
        #      np.mean(self.fw_thickness),
        #      np.mean(self.thickness),
        #      np.mean(self.elev),
        #      np.mean(self.fault_plane_elev))
        self.update_igneous_intrusions(dt)
        # print("after mean hwt, fwt, tt",
        #      np.mean(self.hw_thickness),
        #      np.mean(self.fw_thickness),
        #      np.mean(self.thickness),
        #      np.mean(self.elev),
        #      np.mean(self.fault_plane_elev))
        self.tec_counter += 1
        if self.tec_counter == self.tec_interval:
            self.update_tectonics_and_flexure(dt * self.tec_interval)
            self.tec_counter = 0
            # print("TEC update mean hwt, fwt, tt",
            #      np.mean(self.hw_thickness),
            #      np.mean(self.fw_thickness),
            #      np.mean(self.thickness),
            #      np.mean(self.elev),
            #      np.mean(self.fault_plane_elev))
        self.update_sea_level(self.current_time)
        self.elev_before_ero_dep[:] = self.elev.copy()
        self.set_boundaries_for_subaerial()
        for _ in range(4):
            self.update_subaerial_processes(dt / 4)
            self.deposit_river_sediment_at_coast(dt / 4)
        self.set_boundaries_for_full_domain()
        self.update_submarine_processes(dt)
        self.update_deposit_and_crust_thickness(dt, self.elev_before_ero_dep)
        self.current_time += dt

    def update_until(self, update_to_time, dt, click_yps=0.0):
        """Iterate up to given time, using time-step duration dt."""
        remaining_time = update_to_time - self.current_time
        while remaining_time > 0.0:
            dt = min(dt, remaining_time)
            start_time = time.time()
            self.update(dt)
            end_time = time.time()
            if click_yps > 0.0:
                click_dur = dt / click_yps
                time.sleep(max(0.0, click_dur - (end_time - start_time)))
            remaining_time -= dt

    def run(self, run_duration=None, dt=None, click_years_per_sec=None):
        """Run the model for given duration, or self.run_duration if none given.

        Includes file output of images and model state at user-specified
        intervals.
        """
        start_time = time.time()
        if run_duration is None:
            run_duration = self.run_duration
        if dt is None:
            dt = self.dt
        if click_years_per_sec is None:
            click_years_per_sec = self.click_years_per_sec

        stop_time = run_duration + self.current_time
        while self.current_time < stop_time:
            next_pause = min(self.next_plot, self.next_save)
            self.update_until(next_pause, dt, click_yps=click_years_per_sec)
            if self.current_time >= self.next_plot:
                self.frame_num += 1
                self.fa.run_one_step()  # re-run router to update water-surface height
                procs = "Sea-level change"
                if self.current_time < self._extension_duration:
                    procs += ", tectonic extension"
                if (
                    self.current_time >= self.intruder.episode_start_time
                    and self.current_time < self.intruder.last_stop_time
                ):
                    procs += ", igneous intrusions"
                display_island(
                    self.grid,
                    self.sea_level_history,
                    self.mean_elevation_history,
                    self.dt,
                    self.frame_num,
                    self.display_params,
                    sim_time=self.current_time,
                    procs=procs,
                )
                self.next_plot += self.plot_interval
            if self.current_time >= self.next_save:
                self.save_num += 1
                this_save_name = (
                    self.save_name + str(self.save_num).zfill(self.ndigits) + ".grid"
                )
                save_grid(self.grid, this_save_name, clobber=True)
                np.savetxt("oes_sea_level_history.txt", self.sea_level_history)
                np.savetxt(
                    "oes_mean_elevation_history.txt", self.mean_elevation_history
                )
                self.next_save += self.save_interval
        rtime = time.time() - start_time
        print("Run loop time", rtime)
        print("Rate", self.run_duration / rtime, "years per sec")


if __name__ == "__main__":
    """Launch a run.

    Optional command-line argument is the name of a yaml-format text file with
    parameters. File should include sections for "grid_setup", "process",
    "run_control", and "output". Each of these should have the format shown in
    the defaults defined above in the class header.
    """
    if len(sys.argv) > 1:
        params = load_params(sys.argv[1])
        sim = IslandSimulator(
            params["grid_setup"],
            params["process"],
            params["run_control"],
            params["output"],
        )
    else:
        sim = IslandSimulator()  # use default params
    sim.run()
