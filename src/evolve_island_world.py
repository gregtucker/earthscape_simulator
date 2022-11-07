#! /usr/bin/env python
# coding: utf-8

# # Evolve Island World
#
# *(Greg Tucker, University of Colorado Boulder)*
#
# Demonstration of a Landlab-built simulation of the morphological evolution of
# a hypothetical island micro-continent.
#

from landlab.io.native_landlab import load_grid, save_grid
from landlab import ModelGrid, imshow_grid, RasterModelGrid, create_grid, load_params
from landlab.components import (
    FlowAccumulator,
    ErosionDeposition,
    SimpleSubmarineDiffuser,
    ListricKinematicExtender,
    Flexure,
)
import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import cmocean
import datetime


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
    except:
        return grid.add_zeros(name, at="node", dtype=dtype, clobber=True)


def display_island(grid, current_sea_level, frame_num, params):
    """Display the island and save the image to a png file.

    Parameters
    ----------
    grid : Landlab ModelGrid
        Reference to the simulation grid object.
    current_sea_level : float, m
        Current sea level.
    frame_num : int
        Current frame number, for the file name.
    params : dict
        Dictionary containing:
            max_elev_for_color_scale : float, m
                Maximum elevation for color scale (higher than this gets max color)
            scale_fac_for_surface_water : float
                Factor that controls the color for rivers.
            area_threshold : float
                Rivers with drainage area >= this get a special color
    """
    z = grid.at_node["topographic__elevation"]
    area = grid.at_node["drainage_area"]
    wse = grid.at_node["water_surface__elevation"]
    max_elev_for_color_scale = params["max_elev_for_color_scale"]
    fresh_water_elev_scale = -(
        params["scale_fac_for_surface_water"] * max_elev_for_color_scale
    )
    earth_sea = z - current_sea_level
    is_channel_or_flooded = np.logical_or(area > params["area_threshold"], wse > z)
    is_fresh_water = np.logical_and(is_channel_or_flooded, earth_sea > 0.0)
    earth_sea[is_fresh_water] = fresh_water_elev_scale
    plt.clf()
    imshow_grid(
        grid,
        earth_sea,
        cmap=cmocean.cm.topo,
        vmin=-max_elev_for_color_scale,
        vmax=max_elev_for_color_scale,
    )
    plt.axis(False)
    plt.savefig("island" + str(frame_num).zfill(params["ndigits"]) + ".png")


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
            "sea_level_delta": 0.4,
        },
        "submarine_diffusion": {
            "sea_level": 0.0,
            "wave_base": 60.0,
            "shallow_water_diffusivity": 100.0,
            "tidal_range": 2.0,
        },
        "tectonic_extension": {
            "extension_rate": 0.01,
            "fault_dip": 60.0,
            "fault_location": 4.0e4,
            "detachment_depth": 1.0e4,
            "track_crustal_thickness": True,
        },
        "other": {"crust_datum": -1.5e4, "unit_weight": 2650.0 * 9.8},
    }

    DEFAULT_RUN_PARAMS = {
        "random_seed": 1,
        "dt": 100.0,  # time-step duration, y
        "run_duration": 250000.0,  # duration of run, y
        "start_time": 0.0,  # starting time of simulation, y
    }

    DEFAULT_OUTPUT_PARAMS = {
        "plot_interval": 2000.0,  # time interval for plotting, y
        "save_interval": 25000.0,  # time interval for saving grid, y
        "save_name": "rift-island-save",
        "ndigits": 3,  # number of digits for output files
        "max_elev_for_color_scale": 1650.0,  # elevation for color scale in plotting, m
        "scale_fac_for_surface_water": 0.3,  # surface water gets color equiv to -this times above scale, -
        "area_threshold": 5.0e7,  # minimum drainage area for displayed streams, m2
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
        self.setup_for_flexure(process_params["other"])
        self.setup_run_control(run_params)

        self.fa.run_one_step()  # update surface water for display
        display_island(self.grid, 0.0, 0, output_params)

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
        self.thickness = get_or_create_node_field(self.grid, "upper_crust_thickness")
        self.load = get_or_create_node_field(
            self.flexure_grid, "lithosphere__overlying_pressure_increment"
        )

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
        self.sea_level_history = []
        self.sea_level_delta = params["sea_level_delta"]

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

        self.fl = Flexure(self.flexure_grid, **params["flexure"])

    def setup_for_flexure(self, params):
        """Initialize variables for flexure and calculate initial deflection."""
        self.crust_datum = params["crust_datum"]
        self.unit_weight = params["unit_weight"]
        self.thickness[:] = self.elev - self.crust_datum
        self.load[:] = self.unit_weight * self.thickness
        self.fl.update()
        self.deflection = self.flexure_grid.at_node[
            "lithosphere_surface__elevation_increment"
        ]
        self.init_deflection = self.deflection.copy()
        self.cum_subs = self.grid.at_node["cumulative_subsidence_depth"]

        # for tracking purposes
        self.init_thickness = self.thickness.copy()

    def setup_run_control(self, params):
        """Initialize variables related to control of run timing."""
        self.run_duration = params["run_duration"]
        self.dt = params["dt"]
        self.current_time = params["start_time"]

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
        self.ke.run_one_step(dt)  # update extensional subsidence
        self.load[self.grid.core_nodes] = self.unit_weight * (
            self.thickness[self.grid.core_nodes] - self.cum_subs[self.grid.core_nodes]
        )
        self.fl.update()  # update flexure
        self.elev[:] = (
            self.crust_datum
            + self.thickness
            - (self.cum_subs + (self.deflection - self.init_deflection))
        )

    def update_sea_level(self):
        """Update sea level by adding a random increment."""
        self.current_sea_level += self.sea_level_delta * np.random.randn()
        self.sea_level_history.append(self.current_sea_level)

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
        depo_rate = self.grid.at_node["sediment__influx"] / self.grid.area_of_cell[0]
        self.elev[self.is_submarine] += depo_rate[self.is_submarine] * dt

    def update_submarine_processes(self, dt):
        """Run the submarine diffusion component for one time step dt."""
        self.sd.sea_level = self.current_sea_level
        self.sd.run_one_step(dt)

    def update_deposit_and_crust_thickness(self, dt, elev_before):
        """Update the crustal thickness and the cumulative deposit thickness.

        (TODO: Note that cum depo might no longer be needed if a fluvial
        component that tracks sediment is used)
        """
        dz = self.elev[self.grid.core_nodes] - elev_before[self.grid.core_nodes]
        self.cum_depo[self.grid.core_nodes] += dz
        self.thickness[self.grid.core_nodes] += dz

    def update(self, dt):
        """Run all components for one time step of duration dt."""
        self.update_tectonics_and_flexure(dt)
        self.update_sea_level()
        self.elev_before_ero_dep = self.elev.copy()
        self.set_boundaries_for_subaerial()
        self.update_subaerial_processes(dt)
        self.deposit_river_sediment_at_coast(dt)
        self.set_boundaries_for_full_domain()
        self.update_submarine_processes(dt)
        self.update_deposit_and_crust_thickness(dt, self.elev_before_ero_dep)
        self.current_time += dt

    def update_until(self, update_to_time, dt):
        """Iterate up to given time, using time-step duration dt."""
        remaining_time = update_to_time - self.current_time
        while remaining_time > 0.0:
            dt = min(dt, remaining_time)
            self.update(dt)
            remaining_time -= dt

    def run(self, run_duration=None, dt=None):
        """Run the model for given duration, or self.run_duration if none given.

        Includes file output of images and model state at user-specified
        intervals.
        """
        if run_duration is None:
            run_duration = self.run_duration
        if dt is None:
            dt = self.dt

        stop_time = run_duration + self.current_time
        while self.current_time < stop_time:
            next_pause = min(self.next_plot, self.next_save)
            self.update_until(next_pause, dt)
            if self.current_time >= self.next_plot:
                self.frame_num += 1
                self.fa.run_one_step()  # re-run flow router to update the water-surface height
                display_island(
                    self.grid,
                    self.current_sea_level,
                    self.frame_num,
                    self.display_params,
                )
                self.next_plot += self.plot_interval
            if self.current_time >= self.next_save:
                self.save_num += 1
                this_save_name = (
                    self.save_name + str(self.save_num).zfill(self.ndigits) + ".grid"
                )
                save_grid(self.grid, this_save_name, clobber=True)
                self.next_save += self.save_interval


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
