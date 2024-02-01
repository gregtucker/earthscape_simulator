#! /usr/bin/env python
# coding: utf-8

# # Igneous Intruder
#
# *(Greg Tucker, University of Colorado Boulder)*
#

import numpy as np


class Intrusion:
    def __init__(
        self,
        thickness,
        radius,
        duration,
        max_rate,
        start_time,
        center_x,
        center_y,
        depth,
        x_of_node,
        y_of_node,
    ):
        self.thickness = thickness
        self.radius = radius
        self.duration = duration
        self.start_time = start_time
        self.stop_time = start_time + duration
        self.depth = depth
        self.nodes, distances = self.identify_intrusion_nodes(
            x_of_node, y_of_node, center_x, center_y, radius
        )
        self.rates = max_rate * (1.0 - (distances / radius) ** 2)

    def identify_intrusion_nodes(self, x_of_node, y_of_node, intru_x, intru_y, radius):
        """
        Identify and return array of node IDs affected by an intrusion
        with a circular footprint of radius `radius` and center coordinates
        `intru_x` and `intru_y`.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> grid = RasterModelGrid((6, 6))
        >>> IgneousIntruder.identify_intrusion_nodes(
        ...     None, grid.x_of_node, grid.y_of_node, 3.0, 3.0, 2.5)
        array([ 7,  8,  9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28])
        """
        dist2 = (x_of_node - intru_x) ** 2 + (y_of_node - intru_y) ** 2
        nodes = np.where(dist2 <= radius**2)[0]
        distances = np.sqrt(dist2[nodes])
        return nodes, distances


class IgneousIntruder:
    def __init__(
        self,
        grid,
        contacts,
        layers,
        episode_duration,
        episode_start_time,
        mean_waiting_time,
        thickness_min,
        thickness_max,
        radius_min,
        radius_max,
        log_duration_min,
        log_duration_max,
        depth_min,
        depth_max,
        seed=None,
    ):
        """
        Initialize IgneousIntruder.
        """
        self.intrusions = []
        self.contacts = contacts
        self.layers = layers
        self.episode_start_time = episode_start_time
        thickness_range = thickness_max - thickness_min
        radius_range = radius_max - radius_min
        log_duration_range = log_duration_max - log_duration_min
        depth_range = depth_max - depth_min
        if seed is not None:
            np.random.seed(seed)
        self.last_stop_time = 0.0
        ctime = 0.0
        while ctime < episode_duration:
            this_thickness = thickness_min + thickness_range * np.random.rand()
            this_radius = radius_min + radius_range * np.random.rand()
            this_duration = 10.0 ** (
                log_duration_min + log_duration_range * np.random.rand()
            )
            this_maxrate = this_thickness / this_duration
            this_x = np.amax(grid.x_of_node[grid.core_nodes]) * np.random.rand()
            this_y = np.amax(grid.y_of_node[grid.core_nodes]) * np.random.rand()
            this_depth = depth_min + depth_range * np.random.rand()
            this_intrusion = Intrusion(
                this_thickness,
                this_radius,
                this_duration,
                this_maxrate,
                ctime + episode_start_time,
                this_x,
                this_y,
                this_depth,
                grid.x_of_node,
                grid.y_of_node,
            )
            self.intrusions.append(this_intrusion)
            if this_intrusion.stop_time > self.last_stop_time:
                self.last_stop_time = this_intrusion.stop_time
            # print(ctime + episode_start_time, vars(this_intrusion))
            ctime += mean_waiting_time * np.random.exponential()

    def inject_intrusion(self, nodes, rates, depth, dt, contacts, layers):
        """

        Notes
        -----
        Takes a list of *contacts* and *layers*. A *contact* is an array that
        contains the height either of the surface (the contact between the
        uppermost layer and the atmosphere or ocean) or the contact between
        two layers. A *layer* is an array of the same length as each *contact*
        that contains the thickness of the layer between the corresponding
        contact and the contact below it (or an arbitrary datum, if it's the
        last one). Any *layer* can be `None`, in which case it is ignored.

        The inflation of a layer affects the thickness of that particular
        layer, but it affects the height not just of its own upper contact but
        also that of each contact lying above. The arrays:
            `layer_rates` and `max_layer_rates`
        are used to handle this fact: `layer_rates` contains the rate
        for each relevant node and each layer; `max_layer_rates` is used to
        raise the height of overlying layers.

        Examples
        --------
        >>> grid = RasterModelGrid((3, 5))
        >>> elev = grid.add_zeros("topographic__elevation", at="node")
        >>> mid_upr_cont = grid.add_zeros("mid_upr_contact", at="node")
        >>> mid_upr_cont[:] = -grid.x_of_node - 1.0
        >>> lwr_mid_cont = grid.add_zeros("lwr_mid_contact", at="node")
        >>> lwr_mid_cont[:] = mid_upr_cont - 1.0
        >>> upper_layer = grid.add_zeros("upper_layer", at="node")
        >>> upper_layer[:] = elev - mid_upr_cont
        >>> mid_layer = grid.add_zeros("mid_layer", at="node")
        >>> mid_layer[:] = mid_upr_cont - lwr_mid_cont
        >>> lower_layer = grid.add_zeros("lower_layer", at="node")
        >>> lower_layer[:] = mid_upr_cont + 10.0
        >>> nodes = np.array([6, 7, 8])
        >>> rates = np.array([1.0, 1.0, 1.0])
        >>> contacts = [elev, mid_upr_cont, lwr_mid_cont]
        >>> layers = [upper_layer, mid_layer, lower_layer]
        >>> IgneousIntruder.inject_intrusion(
        ...     None, nodes, rates, depth=-3.5,
        ...     dt=1.0, contacts=contacts, layers=layers)
        >>> elev[grid.core_nodes]
        array([ 1.,  1.,  1.])
        >>> mid_upr_cont[grid.core_nodes]
        array([-1., -2., -4.])
        >>> lwr_mid_cont[grid.core_nodes]
        array([-2., -4., -5.])
        >>> upper_layer[grid.core_nodes]
        array([ 2.,  3.,  5.])
        >>> mid_layer[grid.core_nodes]
        array([ 1.,  2.,  1.])
        >>> lower_layer[grid.core_nodes]
        array([ 9.,  7.,  6.])
        """
        print("injecting at", nodes[0], depth)
        nnodes = len(nodes)
        ncontacts = len(contacts)
        layer_rates = np.zeros((nnodes, ncontacts))
        max_layer_rates = np.zeros((nnodes, ncontacts))
        layer_rates[:, 0] = rates
        max_layer_rates[:, 0] = rates
        for i in range(1, ncontacts):
            c = contacts[i][nodes]
            n = np.where(depth < c)[0]
            layer_rates[n, i] = layer_rates[n, i - 1]
            max_layer_rates[n, i] = max_layer_rates[n, i - 1]
            layer_rates[n, i - 1] = 0.0
            contacts[i - 1][nodes] += max_layer_rates[:, i - 1] * dt
            if layers[i - 1] is not None:
                layers[i - 1][nodes] += layer_rates[:, i - 1] * dt
        contacts[-1][nodes] += max_layer_rates[:, -1] * dt
        if layers[-1] is not None:
            layers[-1][nodes] += layer_rates[:, -1] * dt

    def update(self, dt, current_time=0.0):
        """Update intrusions."""
        for intrusion in self.intrusions:
            if (
                current_time >= intrusion.start_time
                and current_time < intrusion.stop_time
            ):
                self.inject_intrusion(
                    intrusion.nodes,
                    intrusion.rates,
                    -intrusion.depth,
                    dt,
                    self.contacts,
                    self.layers,
                )
