#!/usr/bin/env python
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
import cmocean
import datetime


class IslandSimulator:
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


# Parameters: submarine sediment transport

# Derived or initial parameters
current_sea_level = 0.0
next_plot = plot_interval  # next time to plot
next_save = save_interval  # next time to save grid
frame_num = 0  # current output image frame number
save_num = 0  # current save file frame number
save_name = "rift-island-save"

# Other initialization
np.random.seed(seed)
sea_level = []  # list of sea-level values over time


# ### Load grid and topography
#
# We start with a previously generated hex grid. This grid includes a topography field that represents a quasi-circular oceanic plateau. We also want to record the perimeter node IDs so we can work with them later.

# In[3]:


grid = load_grid("initial_island.grid")
z = grid.at_node["topographic__elevation"]
perimeter_nodes = grid.status_at_node != grid.BC_NODE_IS_CORE


# #### Display initial topography

# In[4]:


cmap = copy.copy(mpl.cm.get_cmap("seismic"))
scale = np.amax(np.abs(z))
imshow_grid(grid, z, vmin=-scale, vmax=scale, cmap=cmap)


# ### Create a raster grid for flexure
#
# The 2D elastic lithosphere flexure component `Flexure` requires a raster grid (not hex). We will therefore define a separate raster grid for this operation. The grid has the same number of rows and columns as the hex grid, and the same spacing on the two axes. The
# only difference is that the hex grid has alternate rows offset by half a grid width. (Because we assume that the flexural wavelength is much longer than this, we don't bother interpolating between the grids.)

# In[5]:


flex_rast_grid = RasterModelGrid(
    (grid.number_of_node_rows, grid.number_of_node_columns),
    xy_spacing=(grid.spacing, 0.866 * grid.spacing),
)


# ### Create grid fields
#
# In addition to the `topographic__elevation` field, and the output fields created by the various Components, we need the following fields:
#
# - *Water surface elevation:* the "filled topography" field used by the flow routing and depression-filling algorithms (using a separate field allows us to fill depressions with water rather than raising the topographic elevations).
#
# - *Subaerial flag:* boolean field indicating whether a given node is above current relative sea level.
#
# - *Cumulative deposit thickness:* used to track the thickness of sediment and (where negative) cumulative exhumation.
#
# - *Upper crust thickness:* used in flexural isostasy calculations to keep track of the time- and  space-varying load.
#
# - *Load:* the weight per unit area of rock/sediment (note: in this version we do not track water loading, though ultimately one should).

# In[6]:


# Add field(s)
wse = grid.add_zeros("water_surface__elevation", at="node", clobber=True)
subaerial = grid.add_zeros("is_subaerial", at="node", dtype=bool, clobber=True)
cum_depo = grid.add_zeros("cumulative_deposit_thickness", at="node")
thickness = grid.add_zeros("upper_crust_thickness", at="node")
load = flex_rast_grid.add_zeros("lithosphere__overlying_pressure_increment", at="node")


# ### Import Components
#
# Here we import the Components needed for this model:
#
# - FlowAccumulator: handles subaerial routing of surface-water flow. Also creates a FlowDirectorSteepest and a LakeMapperBarnes.
#
# - ErosionDeposition: handles erosion and deposition by fluvial processes, using the Davy & Lague (2009) equations.
#
# - SimpleSubmarineDiffuser: transports sediment under water using diffusion with a coefficient that varies with local water depth.
#
# - ListricKinematicExtender: calculates tectonic extension on an idealized listric normal fault, with periodic horizontal shift of topography in the hangingwall.
#
# - Flexure: handles 2D elastic lithosphere flexure.

# In[7]:


from landlab.components import (
    FlowAccumulator,
    ErosionDeposition,
    SimpleSubmarineDiffuser,
    ListricKinematicExtender,
    Flexure,
)


# ### Instantiate Components
#
# Note that Flexure gets its own grid.

# In[8]:


fa = FlowAccumulator(
    grid,
    depression_finder="LakeMapperBarnes",
    fill_surface=wse,
    redirect_flow_steepest_descent=True,
    reaccumulate_flow=True,
)

ed = ErosionDeposition(grid, K=K_br, v_s=v_s, solver="adaptive")

sd = SimpleSubmarineDiffuser(
    grid, sea_level=0.0, wave_base=wave_base, shallow_water_diffusivity=marine_diff
)

ke = ListricKinematicExtender(
    grid,
    extension_rate=extension_rate,
    fault_dip=fault_dip,
    fault_location=fault_location,
    detachment_depth=detachment_depth,
    track_crustal_thickness=True,
)

fl = Flexure(flex_rast_grid, eet=effective_elastic_thickness, method="flexure")


# ### Define sea level function
#
# This function adds or subtracts a random amount to the current sea level.

# In[9]:


def sea_level_random(current_sea_level, delta):
    return current_sea_level + delta * np.random.randn()


# ### Set up flexure and tectonic subsidence
#
# To initialize calculation of flexural isostasy and rift-related subsidence, we need to calculate:
#
# - the starting crustal thickness (above the datum, which is arbitrary)
# - the load created by this thickness
# - the initial lithospheric deflection (calculated via a call to Flexure.update())
#
# We save this initial deflection, so that for each time step we can calculate the net deflection over time (in other words, the initial deflection is assumed to be "already accounted for" in the initial topography).
#
# We also create a shorthand variable, *cum_subs*, to access the cumulative subsidence field.

# In[10]:


# Prepare flexure and tectonic subsidence
thickness[:] = z - crust_datum
load[:] = unit_wt * thickness
fl.update()
deflection = flex_rast_grid.at_node["lithosphere_surface__elevation_increment"]
init_deflection = deflection.copy()
cum_subs = grid.at_node["cumulative_subsidence_depth"]

# for tracking purposes
init_thickness = thickness.copy()


# ### Create a display function
#
# This function displays the current topography, and saves a plot to file.

# In[11]:


def display_island(grid, current_sea_level, frame_num, ndigits):
    z = grid.at_node["topographic__elevation"]
    fa.run_one_step()  # re-run flow router to update the water-surface height
    wse = grid.at_node["water_surface__elevation"]
    fresh_water_elev_scale = -(scale_fac_for_surface_water * max_elev_for_color_scale)
    earth_sea = z - current_sea_level
    area = grid.at_node["drainage_area"]
    is_channel_or_flooded = np.logical_or(area > area_threshold, wse > z)
    is_fresh_water = np.logical_and(is_channel_or_flooded, earth_sea > 0.0)
    earth_sea[is_fresh_water] = fresh_water_elev_scale
    imshow_grid(
        grid,
        earth_sea,
        cmap=cmocean.cm.topo,
        vmin=-max_elev_for_color_scale,
        vmax=max_elev_for_color_scale,
    )
    plt.axis(False)
    plt.savefig("island" + str(frame_num).zfill(ndigits) + ".png")


# ### Display the starting topography
#
# Create an image of the starting condition.

# In[12]:


display_island(grid, 0.0, 0, ndigits)


# ## Run
#
# ### Tectonics and flexure
#
# The kinematic extender updates the cumulative subsidence created by the fact that the hangingwall is sliding down a listric ramp. The load is then calculated based on the current thickness minus what has been lost to subsidence (because subsidence comes from local thinning of the crust as the hangingwall slides by, in general replacing a thicker slice with a thinner one). The isostatic deflection is calculated based on the updated load. The topography is then updated by adding the thickness field to the crustal datum elevation, and subtracting the cumulative subsidence plus the isostatic subsidence (which in many places will be negative, i.e., isostatic uplift in response to tectonic and erosional thinning).
#
# ### Sea level
#
# Current sea level is updated, and appended to the list to keep track of sea-level history. Subaerial and submarine nodes are identified based on the new sea level.
#
# ### Copying present topography
#
# We make a copy of the topography at this point in order to later calculate the *change* in topography due to erosion and sedimentation.
#
# ### Subaerial erosion and deposition
#
# In order to restrict subaerial flow routing and fluvial erosion/deposition to land only, we change the boundary status such that all submarine nodes are flagged as boundary (fixed-value) nodes. We then run the flow-routing algorithms, followed by running the ErosionDeposition (fluvial) Component for one time step.
#
# ### Submarine erosion and deposition
#
# In order to keep track of sediment delivered to the shoreline by rivers, we take the fluvial sediment-influx field, which is in m3/y, and convert it to a deposition rate by dividing by cell area. For submarine nodes, which were previously treated as boundaries and so were not updated for deposition, we now deposit this material by adding one time step's worth of deposition.
#
# We now apply submarine water-depth-dependent diffusion. This calculation will be applied to the entire grid, with an arbitrarily small diffusion coefficient applied to subaerial nodes. To enable this, we switch the boundary status of submarine nodes back to CORE, while keeping the perimeter nodes as open (fixed-value) boundaries.
#
# ### Cumulative erosion and deposition
#
# We update the cumulative erosion/deposition by differencing the topography before and after this latest time step (because we copied the topography *after* doing tectonics and flexure, we include here only the effects of erosion and deposition).
#
# ### Updating crustal thickness
#
# We need to keep track of crustal thickness for the flexure calculations. Here we modify crustal thickness by adding/subtracting and deposition/erosion during this time step.
#
# ### Plotting and saving
#
# We periodically pause to plot an image of the model to a file, and/or to save the run to a Landlab .grid file.

# In[13]:


for i in range(1, num_iter + 1):

    print(i)

    # Tectonic extension & flexure
    ke.run_one_step(dt)  # update extensional subsidence
    load[grid.core_nodes] = unit_wt * (
        thickness[grid.core_nodes] - cum_subs[grid.core_nodes]
    )
    fl.update()  # update flexure
    z[:] = crust_datum + thickness - (cum_subs + (deflection - init_deflection))

    # Adjust sea level
    current_sea_level = sea_level_random(current_sea_level, sea_level_delta)
    print("Sea level = " + str(current_sea_level) + " m")
    sea_level.append(current_sea_level)
    subaerial[:] = z > current_sea_level
    submarine = np.invert(subaerial)

    # Remember previous topo
    z0 = z.copy()

    # Subaerial erosion
    # a. make the submarine nodes open boundaries
    grid.status_at_node[submarine] = grid.BC_NODE_IS_FIXED_VALUE
    grid.status_at_node[subaerial] = grid.BC_NODE_IS_CORE
    # b. route flow
    fa.run_one_step()
    # c. do some erosion
    ed.run_one_step(dt)

    # Submarine deposition
    depo_rate = ed._qs_in / grid.area_of_cell[0]
    z[submarine] += depo_rate[submarine] * dt

    # Submarine diffusion
    # a. make the submarine nodes core
    grid.status_at_node[submarine] = grid.BC_NODE_IS_CORE
    grid.status_at_node[perimeter_nodes] = grid.BC_NODE_IS_FIXED_VALUE
    # b. diffuse
    sd.sea_level = current_sea_level
    sd.run_one_step(dt)

    # Cumulative depo
    cum_depo[grid.core_nodes] += z[grid.core_nodes] - z0[grid.core_nodes]

    # Update crustal thickness
    thickness[grid.core_nodes] += z[grid.core_nodes] - z0[grid.core_nodes]

    # Plot
    if i * dt >= next_plot:
        frame_num += 1
        plt.clf()
        display_island(grid, current_sea_level, frame_num, ndigits)
        next_plot += plot_interval

    # Save
    if i * dt >= next_save:
        save_num += 1
        this_save_name = save_name + str(save_num).zfill(ndigits) + ".grid"
        save_grid(grid, this_save_name, clobber=True)
        next_save += save_interval


# ## Finalize
#
# Here we do some plotting of the model's state at the end of the run.

# ### Topography & bathymetry
#
# Note that bathymetry is cut off; colors indicating the deepest should be take as that deep OR DEEPER.

# In[14]:


import cmocean
import datetime

area_threshold = 5e7
za = grid.at_node["topographic__elevation"] - current_sea_level
cscale = 1500.0
deep_water_scale = -cscale
river_scale = -0.5 * cscale

river = np.logical_and(grid.at_node["drainage_area"] > area_threshold, za > 0.0)
za[river] = river_scale

za[za < deep_water_scale] = deep_water_scale

fa.run_one_step()
lake = np.logical_and(wse > z, za > 0.0)
za[lake] = river_scale

imshow_grid(grid, za, cmap=cmocean.cm.topo, vmin=-cscale, vmax=cscale)
plt.axis(False)
figname = (
    "rift-island-t"
    + str(int(num_iter * dt))
    + "-"
    + datetime.date.today().strftime("%y%m%d")
    + ".pdf"
)
plt.savefig(figname)


# ### Cumulative deposition/erosion

# In[15]:


cdep = cum_depo.copy()
cdep[perimeter_nodes] = 0.0
dmax = np.amax(np.abs(cdep))
imshow_grid(grid, cdep, cmap="Spectral", vmin=-dmax, vmax=dmax)
plt.axis(False)
plt.savefig("cum_depo.png")


# ### Sea-level history

# In[16]:


plt.plot(0.001 * dt * np.arange(len(sea_level)), sea_level)
plt.xlabel("Time since start of run (ky)")
plt.ylabel("Sea level (m)")
plt.title("Sea level history")
plt.grid(True)


# ### Cross-sectional profile

# In[17]:


startnode = (grid.number_of_node_rows // 2) * grid.number_of_node_columns
endnode = startnode + grid.number_of_node_columns
midrow = np.arange(startnode, endnode, dtype=int)

x = 0.001 * grid.spacing * np.arange(0.0, len(midrow))

plt.figure()
plt.plot(x, z[midrow] - np.maximum(cdep[midrow], 0.0), "k:", label="Basement")
plt.plot(x, z[midrow], "g", label="Surface")
plt.plot([0, max(x)], [current_sea_level, current_sea_level], label="Sea level")
plt.xlabel("Distance (km)")
plt.ylabel("Elevation (m)")
plt.legend()
plt.grid(True)


# ### Flexure

# In[18]:


net_flex = init_deflection - deflection
imshow_grid(flex_rast_grid, net_flex)


# End of notebook.
