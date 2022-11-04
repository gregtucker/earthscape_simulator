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
from landlab import imshow_grid, RasterModelGrid
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy

# import cmocean
import datetime


class IslandSimulator:
    """Simulate geologic evolution of an island or micro-continent.

    Examples
    --------
    >>> sim = IslandSimulator()
    """

    ALL_PROCESSES = (
        "flexure",
        "fluvial",
        "hillslope",
        "runoff",
        "sea_level",
        "submarine_diffusion",
        "tectonic_extension",
    )

    DEFAULT_PARAMS = {
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

    def __init__(
        K_br=1.0e-5,  # fluvial erosion coefficient, 1/y
        v_s=1.0,  # fluvial deposition parameter, -
        sea_level_delta=0.4,  # scale factor for random SL variation, m
        wave_base=50.0,  # depth to wave base, m
        marine_diff=100.0,  # marine sediment diffusivity, m2/y
        extension_rate=0.01,  # horizontal extension rate, m/y
        fault_dip=60.0,  # surface fault dip, degrees
        fault_location=4.0e4,  # location parameter for fault, m
        detachment_depth=1.0e4,  # depth to decollement, m
        effective_elastic_thickness=1.0e4,  # elastic thickness, m
        crust_datum=-1.5e4,  # depth to datum in crust, m
        unit_wt=2650.0 * 9.8,  # unit weight of load, kg / m s2
        dt=100.0,  # time-step duration, y
        num_iter=2500,  # number of iterations
        plot_interval=2000.0,  # time interval for plotting, y
        save_interval=25000.0,  # time interval for saving grid, y
        ndigits=3,  # number of digits for output files
        seed=1,  # random seed
        max_elev_for_color_scale=1650.0,  # elevation for color scale in plotting, m
        scale_fac_for_surface_water=0.3,  # surface water gets color equiv to -this times above scale, -
        area_threshold=5.0e7,  # minimum drainage area for displayed streams, m2
    ):
        pass

    def set_fluvial_parameters(K_br, v_s):
        self.K_br = K_br

    def update():
        pass

    def update_until():
        pass

    def run():
        pass
