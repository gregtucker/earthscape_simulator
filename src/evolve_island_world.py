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
from landlab import ModelGrid, imshow_grid, RasterModelGrid, create_grid
from landlab.components import (
    FlowAccumulator,
    SpaceLargeScaleEroder,
    SimpleSubmarineDiffuser,
    ListricKinematicExtender,
    Flexure,
)
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
            "K_br": 1.0e-5,
            "K_sed": 1.0e-2,
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
        "num_iter": 2500,  # number of iterations
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
        self.is_subaerial = get_or_create_node_field(
            self.grid, "is_subaerial", dtype="bool"
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
        self.current_sea_level = 0.0
        self.sea_level_history = []
        self.sea_level_delta = params["sea_level_delta"]

    def setup_for_output(self, params):
        self.plot_interval = params["plot_interval"]
        self.next_plot = self.plot_interval
        self.save_interval = params["save_interval"]
        self.next_save = self.save_interval
        self.ndigits = params["ndigits"]
        self.frame_num = 0  # current output image frame number
        self.save_num = 0  # current save file frame number
        self.save_name = params["save_name"]

    def instantiate_components(self, params):
        """Instantiate and initialize process components."""

        self.fa = FlowAccumulator(
            self.grid,
            depression_finder="LakeMapperBarnes",
            fill_surface=self.wse,
            redirect_flow_steepest_descent=True,
            reaccumulate_flow=True,
        )

        self.sp = SpaceLargeScaleEroder(self.grid, **params["fluvial"])

        self.sd = SimpleSubmarineDiffuser(
            self.grid,
            **params["submarine_diffusion"],
        )

        self.ke = ListricKinematicExtender(self.grid, **params["tectonic_extension"])

        self.fl = Flexure(self.flexure_grid, **params["flexure"])

    def setup_for_flexure(self, params):
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
        self.num_iter = params["num_iter"]
        self.dt = params["dt"]

    def set_boundaries_for_subaerial(self):
        self.subaerial[:] = self.elev > self.current_sea_level
        self.grid.status_at_node[
            np.invert(self.subaerial)
        ] = grid.BC_NODE_IS_FIXED_VALUE
        self.grid.status_at_node[subaerial] = grid.BC_NODE_IS_CORE

    def set_boundaries_for_full_domain(self):
        self.grid.status_at_node[self.interior_nodes] = grid.BC_NODE_IS_CORE

    def update_tectonics_and_flexure(self, dt):

        self.ke.run_one_step(dt)  # update extensional subsidence
        self.load[self.grid.core_nodes] = self.unit_wt * (
            self.thickness[self.grid.core_nodes] - self.cum_subs[self.grid.core_nodes]
        )
        self.fl.update()  # update flexure
        self.elev[:] = (
            self.crust_datum
            + self.thickness
            - (self.cum_subs + (self.deflection - self.init_deflection))
        )

    def update_sea_level(self):
        self.current_sea_level += self.sea_level_delta * np.random.randn()
        # print("Sea level = " + str(current_sea_level) + " m")
        self.sea_level_history.append(self.current_sea_level)

    def update_subaerial_processes(self, dt):
        self.fa.run_one_step()
        self.ed.run_one_step(dt)

    def deposit_river_sediment_at_coast(self, dt):
        depo_rate = self.ed._qs_in / self.grid.area_of_cell[0]
        self.elev[self.submarine] += depo_rate[self.submarine] * dt

    def update_submarine_processes(self, dt):
        self.sd.sea_level = self.current_sea_level
        self.sd.run_one_step(dt)

    def update_deposit_and_crust_thickness(self, dt, elev_before):
        dz = self.elev[self.grid.core_nodes] - elev_before[self.grid.core_nodes]
        self.cum_depo[self.grid.core_nodes] += dz
        self.thickness[grid.core_nodes] += dz

    def update(self, dt):
        self.update_tectonics_and_flexure(dt)
        self.update_sea_level(dt)
        self.elev_before_ero_dep = self.elev.copy()
        self.set_boundaries_for_subaerial()
        self.update_subaerial_processes(dt)
        self.deposit_river_sediment_at_coast(dt)
        self.set_boundaries_for_full_domain()
        self.update_submarine_processes(dt)
        self.update_deposit_and_crust_thickness(dt, elev_before_ero_dep)

    def update_until():
        pass

    def run(self, num_iter=None, dt=None):

        if num_iter is None:
            num_iter = self.num_iter
        if dt is None:
            dt = self.dt

        # TODO: use update_until to run to display or output step
        for i in range(1, num_iter + 1):
            self.update(dt)
