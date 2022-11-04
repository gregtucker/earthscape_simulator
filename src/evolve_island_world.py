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
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy

# import cmocean
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
        "flexure": {"method": "flexure", "rho_mantle": 3300.0, "isostasytime": 0},
        "fluvial": {
            "K_br": 1.0e-5,
            "v_s": 1.0,
        },
        "hillslope:": {},
        "runoff": {},
        "sea_level": {
            "sea_level_delta": 0.4,
        },
        "submarine_diffusion": {
            "wave_base": 50.0,
            "marine_diff": 100.0,
        },
        "tectonic_extension": {
            "extension_rate": 0.01,
            "fault_dip": 60.0,
            "fault_location": 4.0e4,
            "detachment_depth": 1.0e4,
            "crust_datum": -1.5e4,
        },
    }

    DEFAULT_RUN_PARAMS = {
        "random_seed": 0,
    }

    DEFAULT_OUTPUT_PARAMS = {}

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
        # self.setup_fields()

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

    def set_fluvial_parameters(K_br, v_s):
        self.K_br = K_br

    def update():
        pass

    def update_until():
        pass

    def run():
        pass
