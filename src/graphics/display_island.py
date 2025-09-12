"""
display_island: module in EarthscapeSimulator that displays various information
about the current island and its configuration.

Greg Tucker, Jan 2024
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cmocean
from landlab import imshow_grid

_HALFRANGE_FOR_ERODEP = (
    5.0  # plot color range for image of cumulative erosion/deposition
)
_NFRAMES = 60
_MIN_FRAMES = 1


def display_island(
    grid,
    sea_level_history,
    mean_elev_history,
    time_interval,
    frame_num,
    params,
    sim_time=-1.0,
    procs="",
):
    """
    Display the island and save the image to a png file.

    Parameters
    ----------
    grid : Landlab ModelGrid
        Reference to the simulation grid object.
    sea_level_history : list or array of float
        History of sea level values up to now (m)
    mean_elev_history : list or array of float
        History of mean subaerial elevation up to now (m)
    time_interval : float
        Time between successive values of sea level or mean elevation (y)
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
            sim_time : float (optional)
                Current time in simulation, yr (default -1 means do not display time)
    sim_time : float, optional
        Current time in simulation (y)
    procs : str, optional
        List of current processes (to be displayed as part of title)
    """
    plt.rcParams["figure.figsize"] = (7, 24)
    plt.rcParams['figure.constrained_layout.use'] = True

    # Get axis objects
    _, ax = plt.subplots(5, 1)

    display_topo_and_rivers(
        grid,
        ax[0],
        params,
        current_sea_level=sea_level_history[-1],
        sim_time=sim_time,
        procs=procs,
    )

    plot_sea_level_history(sea_level_history, time_interval, ax[1])

    plot_sed_flux(grid, ax[2])

    plot_mean_elev_history(mean_elev_history, sea_level_history, time_interval, ax[3])

    plot_cum_ero_dep(grid, ax[4])

    # Set filename and save to file, then copy file as "latest"
    filename = "island" + str(frame_num).zfill(params["ndigits"]) + ".png"
    plt.savefig(filename, bbox_inches="tight")
    shutil.copy(filename, params["path_for_latest_image"] + "/" + "current_island.png")

    plt.close()

    make_anim(
        "island",
        frame_num,
        _NFRAMES,
        params["ndigits"],
        params["path_for_latest_image"],
    )


def add_scale_bar(ax):
    ax.plot([10000, 60000], [10000, 10000], "w")
    ax.text(20000, 15000, "50 km", color="white")


def setup_for_time_axis(num_vals, dt):
    t = dt * np.arange(num_vals)

    # for time axis extent
    tmax = max(np.amax(t), dt)
    order_of_mag = 10 ** np.floor(np.log10(tmax))
    time_scale_max = order_of_mag * np.ceil(tmax / order_of_mag)

    # for time axis units
    if tmax < 3000:
        time_unit = "(years)"
        time_factor = 1.0
    elif tmax < 1.0e6:
        time_unit = "(thousand years)"
        time_factor = 1.0e3
    else:
        time_unit = "(million years)"
        time_factor = 1.0e6

    return t, time_unit, time_factor, time_scale_max


def plot_sea_level_history(sea_level, dt, ax):
    t, time_unit, time_factor, time_scale_max = setup_for_time_axis(len(sea_level), dt)
    ax.plot(t / time_factor, sea_level, "b")
    ax.plot(max(t) / time_factor, sea_level[-1], "r.")
    ax.set_title(
        "Sea Level History\ncurrent level = " + str(round(sea_level[-1], 1)) + " m"
    )
    ax.set_xlabel("Time " + time_unit)
    ax.set_ylabel("Relative sea level (m)")
    if time_scale_max > 0.0:
        ax.set_xlim([0, time_scale_max / time_factor])


def plot_mean_elev_history(mean_elev, sea_level, dt, ax):
    t, time_unit, time_factor, time_scale_max = setup_for_time_axis(len(mean_elev), dt)

    ax.plot(t / time_factor, mean_elev, "b", label="Relative to sea level")
    ax.plot(max(t) / time_factor, mean_elev[-1], "r.")
    moho_relative = np.array(mean_elev) + np.array(sea_level)
    ax.plot(t / time_factor, moho_relative, "k--", label="Relative to moho")
    ax.plot(max(t) / time_factor, moho_relative[-1], "r.")
    ax.set_title(
        "Mean Elevation History\ncurrent elevation = "
        + str(round(mean_elev[-1], 1))
        + " m ASL"
    )
    ax.set_xlabel("Time " + time_unit)
    ax.set_ylabel("Mean subaerial elevation (m)")
    ax.set_xlim([0, time_scale_max / time_factor])
    ax.legend()


def display_topo_and_rivers(
    grid, ax, params, current_sea_level=0, frame_num=0, sim_time=-1.0, procs=""
):
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
    plt.axes(ax)
    ax.cla()
    imshow_grid(
        grid,
        earth_sea,
        cmap=cmocean.cm.topo,
        vmin=-max_elev_for_color_scale,
        vmax=max_elev_for_color_scale,
        colorbar_label="Elevation above current sea level (m)",
    )
    ax.axis(False)
    add_scale_bar(ax)
    if sim_time != -1:
        if sim_time > 1.0e6:
            time_unit = " million"
            sim_time /= 1.0e6
        else:
            time_unit = " thousand"
            sim_time /= 1.0e3
        ttl = "Time = " + str(round(sim_time, 3)) + time_unit + " years"
        if len(procs) > 0:
            ttl += "\nProcesses: " + procs
        plt.title(ttl)


def plot_sed_flux(grid, ax):
    sedflux = np.zeros(grid.number_of_nodes)
    sedflux[:] = grid.at_node["sediment__outflux"]  # rivers

    vals_at_link = np.zeros(grid.number_of_links + 1)
    vals_at_link[:-1] = np.abs(grid.at_link["hillslope_sediment__unit_volume_flux"])

    vals_at_node = np.mean(vals_at_link[grid.links_at_node], axis=1)
    sedflux[grid.at_node["is_submarine"]] = (
        vals_at_node[grid.at_node["is_submarine"]] * grid.spacing
    )
    # sedflux[:] = vals_at_node[:] * grid.spacing

    plt.axes(ax)
    plt.axis(False)
    imshow_grid(
        grid, np.log10(sedflux + 1), cmap="YlGn", colorbar_label="log(flux, m3/y)"
    )
    plt.title("Sediment Flux")


def plot_cum_ero_dep(grid, ax):
    plt.axes(ax)
    plt.axis(False)
    plt.axis("tight")
    depero = grid.at_node["cumulative_deposit_thickness"] / 1000.0
    half_range = _HALFRANGE_FOR_ERODEP
    imshow_grid(
        grid,
        depero,
        cmap="RdYlBu",
        colorbar_label="Thickness (km)",
        vmin=-half_range,
        vmax=half_range,
    )
    plt.title("Cumulative Sedimentation / Erosion")


def set_size(w, h, ax=None):
    """w, h: width, height in inches"""
    if not ax:
        ax = plt.gca()
    lf = ax.figure.subplotpars.left
    rt = ax.figure.subplotpars.right
    tp = ax.figure.subplotpars.top
    bt = ax.figure.subplotpars.bottom
    figw = float(w) / (rt - lf)
    figh = float(h) / (tp - bt)
    ax.figure.set_size_inches(figw, figh)


def make_anim(basename, endframe, nframes, ndigits, latest_image_path):
    startframe = max(endframe - nframes, 0)

    if (endframe - startframe) + 1 >= _MIN_FRAMES:
        images = []
        for i in range(startframe, endframe + 1):
            img_name = basename + str(i).zfill(ndigits) + ".png"
            try:
                images.append(iio.imread(img_name))
            except FileNotFoundError:
                print("Cannot find file", img_name)
                raise FileNotFoundError
        iio.imwrite(basename + "_movie.gif", images, duration=60, loop=0)
        #imageio.mimsave(basename + "_movie.gif", images)
        shutil.copy(
            basename + "_movie.gif",
            latest_image_path + "/" + "current_island_movie.gif",
        )
