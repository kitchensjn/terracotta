import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import random
import numpy as np
from scipy import linalg
import matplotlib as mpl
import networkx as nx
import tskit
from numba import njit, prange
from collections import Counter
import math
from scipy.optimize import shgo
from scipy.spatial import Voronoi, voronoi_plot_2d
from glob import glob
import sys
import time


def voronoi_finite_polygons_2d(vor, radius=None):
    """

    https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram

    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def _calc_optimal_organization_of_suplots(num_plots):
    ncols = math.ceil(math.sqrt(num_plots))
    nrows = math.ceil(num_plots/ncols)
    return nrows, ncols

@njit()
def logsumexp_custom(x, axis):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c), axis=axis))

class WorldMap:
    """Stores a map of the demes

    Attributes
    ----------
    demes : pandas.DataFrame
    deme_suitabilities : numpy.ndarray
    samples : pandas.DataFrame
    epochs : numpy.ndarray
    connections : list of pandas.DataFrames
    connection_types : list of tuples
        All possible connection types, even if not observed in the map
    """

    def __init__(self, demes, connections, samples=None):
        """Initializes the WorldMap object

        Parameters
        ----------
        demes : pandas.DataFrame
            Five columns: "id", "xcoord", "ycoord", "type", "neighbours" (note the "u" in neighbours).
            See README.md for more details about `demes.tsv` file structure
        samples : pandas.DataFrame
        """

        self.samples = None
        self.sample_locations_array = None
        if samples is not None:
            self.samples = samples.copy()
            self.sample_locations_array = self._build_sample_locations_array()

        self.demes = demes.copy()  
        self.demes["suitability"] = self.demes["suitability"].astype(str)
        formatted_suitability_strings = []
        for suitability in self.demes["suitability"]:
            formatted_suitability_strings.append(self._format_parameter_string(suitability))
        self.demes["suitability"] = formatted_suitability_strings

        self.connections = connections.copy()
        self.connections["migration_modifier"] = self.connections["migration_modifier"].astype(str)
        formatted_modifier_strings = []
        for modifier in self.connections["migration_modifier"]:
            formatted_modifier_strings.append(self._format_parameter_string(modifier))
        self.connections["migration_modifier"] = formatted_modifier_strings
    
        epochs = [0]
        for suitability_string in self.demes["suitability"]:
            epoch_formatter = suitability_string.replace(",", ":").split(":")
            for epoch_assignment in epoch_formatter[::2]:
                epoch_assignment = float(epoch_assignment)
                if epoch_assignment not in epochs:
                    epochs.append(epoch_assignment)
        for modifier_string in self.connections["migration_modifier"]:
            epoch_formatter = modifier_string.replace(",", ":").split(":")
            for epoch_assignment in epoch_formatter[::2]:
                epoch_assignment = float(epoch_assignment)
                if epoch_assignment not in epochs:
                    epochs.append(epoch_assignment)
        self.epochs = np.sort(epochs)

        self.suitability_ratios = [[] for t in range(len(epochs))]
        srs = []
        for index,row in self.connections.iterrows():
            suitability_ratio_string = ""
            for t in range(len(epochs)):
                time_period = epochs[t]
                deme_0_suitability = max(1e-9, self.get_deme_suitability_at_time(row["deme_0"], time_period))
                deme_1_suitability = max(1e-9, self.get_deme_suitability_at_time(row["deme_1"], time_period))
                sr = deme_1_suitability/deme_0_suitability
                self.suitability_ratios[t].append(sr)
                suitability_ratio_string += f"{time_period}:{sr},"
            srs.append(suitability_ratio_string[:-1])
        self.connections["suitability_ratio"] = srs

    def get_deme_suitabilities_by_epoch(self):
        deme_suitabilities = [[] for epoch in self.epochs]
        for suitability_string in self.demes["suitability"]:
            epoch_formatter = suitability_string.replace(",", ":").split(":")
            for t in range(0, len(epoch_formatter), 2):
                deme_suitabilities[np.where(self.epochs==float(epoch_formatter[t]))[0][0]].append(float(epoch_formatter[t+1]))
        for i,dsl in enumerate(deme_suitabilities):
            deme_suitabilities[i] = np.sort(dsl)
        return deme_suitabilities

    def get_unique_deme_suitabilities(self):
        deme_suitabilities = []
        for suitability_string in self.demes["suitability"]:
            epoch_formatter = suitability_string.replace(",", ":").split(":")
            for t in range(0, len(epoch_formatter), 2):
                deme_suitabilities.append(float(epoch_formatter[t+1]))
        deme_suitabilties = np.unique(deme_suitabilties)
        return deme_suitabilities

    def get_range_of_deme_suitabilities(self):
        ds = self.get_unique_deme_suitabilities()
        return min(ds), max(ds)

    def _format_parameter_string(self, s):
        if ":" in s:
            split_string = s.replace(",", ":").split(":")
            epochs = np.array(split_string[::2]).astype(float)
            parameters = np.array(split_string[1::2]).astype(float)
            sorted_epochs = np.argsort(epochs)
            formatted = ""
            if epochs[sorted_epochs[0]] != 0:
                formatted += "0:0,"
            for e in sorted_epochs:
                formatted += f"{epochs[e]}:{parameters[e]},"
            return formatted[:-1]
        elif s == "nan":
            raise RuntimeError(f"String `{s}` cannot be formatted.")
        return f"0:{s}"

    def get_coordinates_of_deme(self, id):
        """Gets x and y coordinates for specified deme

        Parameters
        ----------
        id : int
            ID of deme

        Returns
        -------
        coords : tuple
            (x, y) where x and y are both ints or floats
        """

        row = self.demes.loc[self.demes["id"]==id,:].iloc[0]
        coords = (row["xcoord"], row["ycoord"])
        return coords
    
    def get_deme_suitability_string(self, id):
        """Gets the type for specified deme

        Wrapper of pandas.DataFrame function. Assumes that deme IDs are unique.

        Parameters
        ----------
        id : int
            ID of deme

        Returns
        -------
        deme_type : string
            Formatted string of deme type(s)
        """

        deme_type = self.demes.loc[self.demes["id"]==id,"type"].iloc[0]
        return deme_type
    
    def get_deme_suitability_at_time(self, id, time):
        """Gets the type for specified deme

        Wrapper of pandas.DataFrame function. Assumes that deme IDs are unique.

        Parameters
        ----------
        id : int
            ID of deme
        time : int or float
            Specified time of map

        Returns
        -------
        deme_type : float
            Type of deme
        """

        deme_suitability = self.demes.loc[self.demes["id"]==id,"suitability"].iloc[0]
        epochs = deme_suitability.split(",")
        for t in range(len(epochs)-1):
            details = epochs[t].split(":")
            start = float(details[0])
            end = float(epochs[t+1].split(":")[0])
            if (time >= start) and (time < end):
                return float(details[1])
        return float(epochs[-1].split(":")[1])
    
    def get_all_deme_suitabilities_at_time(self, time):
        """

        Parameters
        ----------
        time : int or float
            Specified time of world map

        Returns
        -------
        suitabilities_at_time : pd.Series
            Deme types ordered by the demes dataframe

        """

        suitabilities_at_time = []
        for id in self.demes["id"]:
            suitabilities_at_time.append(self.get_deme_suitability_at_time(id, time))
        return pd.Series(suitabilities_at_time)
    
    def get_connection_parameter_at_time(self, parameter, id, time):
        connection_parameter = self.connections.loc[self.connections["id"]==id,parameter].iloc[0]
        epochs = connection_parameter.split(",")
        for t in range(len(epochs)-1):
            details = epochs[t].split(":")
            start = float(details[0])
            end = float(epochs[t+1].split(":")[0])
            if (time >= start) and (time < end):
                return details[1]
        return epochs[-1].split(":")[1]

    def plot_suitability_ratio_hist(self):
        

        plt.hist(self.suitability_ratios, histtype="barstacked")
        plt.legend(self.epochs, title="Epoch")
        plt.xlabel("Suitability Ratio")
        plt.ylabel("Count")
        plt.show()

    def draw_migration_surface_voronoi(
            self,
            figsize,
            migration_rate_parameters
        ):

        connections = self.connections.copy()
        times = self.epochs
        nrows, ncols = _calc_optimal_organization_of_suplots(num_plots=len(times))

        max_emr = None
        min_emr = None
        for t in times:
            emr = []
            for _,row in connections.iterrows():
                emr.append((migration_rate_parameters[0] * self.get_connection_parameter_at_time("suitability_ratio", row["id"], t)**migration_rate_parameters[1]) * self.get_connection_parameter_at_time("migration_modifier", row["id"], t))
            connections[f"emr{t}"] = emr
            if max_emr is None:
                max_emr = max(emr)
                min_emr = min(emr)
            else:
                max_emr = max(max_emr, max(emr))
                min_emr = min(min_emr, min(emr))

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        if len(times) == 1:
            axs = np.array([[axs]])
        elif nrows == 1:
            axs = np.array([axs])

        for e in range(nrows*ncols):
            if e < len(times):  #condition where empty subplots still need to have their axes turned off
                epoch = times[e]
                points = []
                zs = []
                for _,row in connections.iterrows():
                    if row["deme_0"] < row["deme_1"]:
                        deme_0 = self.get_coordinates_of_deme(row["deme_0"])
                        deme_1 = self.get_coordinates_of_deme(row["deme_1"])
                        x = (deme_0[0]+deme_1[0])/2
                        y = (deme_0[1]+deme_1[1])/2
                        dx = deme_1[0]-deme_0[0]
                        dy = deme_1[1]-deme_0[1]
                        alt_direction_migration_rate = connections.loc[(connections["deme_1"]==row["deme_0"])&(connections["deme_0"]==row["deme_1"]), f"emr{epoch}"].iloc[0]
                        z = max(row[f"emr{epoch}"], alt_direction_migration_rate)
                        rat = row[f"emr{epoch}"]/alt_direction_migration_rate
                        axs[e//ncols, e%ncols].plot([deme_0[0], deme_1[0]], [deme_0[1], deme_1[1]], color="white", alpha=0.2)
                        if rat > 1.1:
                            axs[e//ncols, e%ncols].arrow(x-(dx*rat/5)/5, y-(dy*rat/5)/2, dx*rat/5, dy*rat/5, length_includes_head=True, color="black", head_width=0.1, zorder=3)
                        elif rat < 0.9:
                            axs[e//ncols, e%ncols].arrow(x+(dx*rat/5)/5, y+(dy*rat/5)/2, -dx*rat/5, -dy*rat/5, length_includes_head=True, color="black", head_width=0.1, zorder=3)
                        points.append([x, y])
                        zs.append(z)

                max_emr = max(zs)
                min_emr = min(zs)
        
                vor = Voronoi(points)
                regions, vertices = voronoi_finite_polygons_2d(vor)

                for i,region in enumerate(regions):
                    polygon = vertices[region]
                    if max_emr > min_emr:
                        color = colorFader("blue","red",mix=(zs[i]-min_emr)/(max_emr-min_emr))
                    else:
                        color = "purple"
                    axs[e//ncols, e%ncols].fill(*zip(*polygon), color)

                axs[e//ncols, e%ncols].scatter(self.demes["xcoord"], self.demes["ycoord"], color="white", zorder=2)
            axs[e//ncols, e%ncols].set_xlim(-6.5, 6.5)
            axs[e//ncols, e%ncols].set_ylim(-6.5, 6.5)
            axs[e//ncols, e%ncols].set_aspect("equal", 'box')
            axs[e//ncols, e%ncols].axis("off")

        plt.show()
        



    def draw_migration_surface(
            self,
            figsize,
            migration_rate_parameters
        ):

        connections = self.connections.copy()
        times = self.epochs
        nrows, ncols = _calc_optimal_organization_of_suplots(num_plots=len(times))

        max_emr = None
        min_emr = None
        for t in times:
            emr = []
            for _,row in connections.iterrows():
                emr.append((migration_rate_parameters[0] * self.get_connection_parameter_at_time("suitability_ratio", row["id"], t)**migration_rate_parameters[1]) * self.get_connection_parameter_at_time("migration_modifier", row["id"], t))
            connections[f"emr{t}"] = emr
            if max_emr is None:
                max_emr = max(emr)
                min_emr = min(emr)
            else:
                max_emr = max(max_emr, max(emr))
                min_emr = min(min_emr, min(emr))    

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        if len(times) == 1:
            axs = np.array([[axs]])
        elif nrows == 1:
            axs = np.array([axs])

        for e in range(nrows*ncols):
            if e < len(times):  #condition where empty subplots still need to have their axes turned off
                epoch = times[e]
                for _,row in connections.iterrows():
                    if row["deme_0"] < row["deme_1"]:
                        deme_0 = self.get_coordinates_of_deme(row["deme_0"])
                        deme_1 = self.get_coordinates_of_deme(row["deme_1"])
                        x = (deme_0[0]+deme_1[0])/2
                        y = (deme_0[1]+deme_1[1])/2
                        dx = deme_1[0]-deme_0[0]
                        dy = deme_1[1]-deme_0[1]
                        alt_direction_migration_rate = connections.loc[(connections["deme_1"]==row["deme_0"])&(connections["deme_0"]==row["deme_1"]), f"emr{epoch}"].iloc[0]
                        which_mr = max(row[f"emr{epoch}"], alt_direction_migration_rate)
                        diff = row[f"emr{epoch}"] - alt_direction_migration_rate
                        if max_emr > min_emr:
                            axs[e//ncols, e%ncols].scatter(x, y, color=colorFader("blue","red",mix=(which_mr-min_emr)/(max_emr-min_emr)), s=500, alpha=0.5)
                        else:
                            axs[e//ncols, e%ncols].scatter(x, y, color="purple")
                        if diff > 0:
                            axs[e//ncols, e%ncols].arrow(x-(dx*abs(diff)*2)/2, y-(dy*abs(diff)*2)/2, dx*abs(diff)*2, dy*abs(diff)*2, length_includes_head=True, color="black", head_width=0.1)
                        elif diff < 0:
                            axs[e//ncols, e%ncols].arrow(x+(dx*abs(diff)*2)/2, y+(dy*abs(diff)*2)/2, -dx*abs(diff)*2, -dy*abs(diff)*2, length_includes_head=True, color="black", head_width=0.1)
                axs[e//ncols, e%ncols].set_title(times[e], fontname="Georgia")
            axs[e//ncols, e%ncols].set_aspect("equal", 'box')
            axs[e//ncols, e%ncols].axis("off")
        plt.show()


    def draw(
            self,
            figsize,
            migration_rate_parameters=None,
            color_connections=False,
            color_demes=False,
            save_to=None,
            location_vector=None,
            show_samples=False,
            title=None,
            show=True
        ):

        """Draws the world map

        Uses matplotlib.pyplot

        Parameters
        ----------
        figsize : tuple
            (width, height) of output figure
        color_demes : bool
            Whether to color the demes based on type. Default is False.
        color_connections : bool
            Whether to color the connections based on type. Default is False.
        save_to : str
            Path to save the map to. Default is None.
        """

        connections = self.connections.copy()
        
        if location_vector is None:
            times = self.epochs
            nrows, ncols = _calc_optimal_organization_of_suplots(num_plots=len(times))
        else:
            times = [0]
            nrows, ncols = 1, 1

        if migration_rate_parameters is not None:
            max_emr = None
            min_emr = None
            for t in times:
                emr = []
                for _,row in connections.iterrows():
                    emr.append((migration_rate_parameters[0] * self.get_connection_parameter_at_time("suitability_ratio", row["id"], t)**migration_rate_parameters[1]) * self.get_connection_parameter_at_time("migration_modifier", row["id"], t))
                connections[f"emr{t}"] = emr
                if max_emr is None:
                    max_emr = max(emr)
                    min_emr = min(emr)
                else:
                    max_emr = max(max_emr, max(emr))
                    min_emr = min(min_emr, min(emr))

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        if len(times) == 1:
            axs = np.array([[axs]])
        elif nrows == 1:
            axs = np.array([axs])

        range_of_deme_suitabilities = self.get_range_of_deme_suitabilities()

        for e in range(nrows*ncols):
            if e < len(times):  #condition where empty subplots still need to have their axes turned off
                epoch = times[e]
                for _,row in connections.iterrows():
                    deme_0 = self.get_coordinates_of_deme(row["deme_0"])
                    deme_1 = self.get_coordinates_of_deme(row["deme_1"])

                    dx = deme_1[0]-deme_0[0]
                    dy = deme_1[1]-deme_0[1]
                    length_of_line = math.sqrt(dx**2 + dy**2)
                    angle_rad = math.atan2(dy, dx)
                    shift_along_x = math.cos(angle_rad) * (length_of_line*0.2)
                    shift_along_y = math.sin(angle_rad) * (length_of_line*0.2)
                    angle_deg = math.degrees(angle_rad)
                    perp_angle_rad = math.atan2(-dx, dy)
                    shift_perp_x = math.cos(perp_angle_rad) * 0.1
                    shift_perp_y = math.sin(perp_angle_rad) * 0.1
                    if (angle_deg >= 0) and (angle_deg < 180):
                        x = deme_0[0] + shift_along_x + shift_perp_x
                        y = deme_0[1] + shift_along_y + shift_perp_y
                    else:
                        x = deme_0[0] + shift_along_x + shift_perp_x
                        y = deme_0[1] + shift_along_y + shift_perp_y
                    if migration_rate_parameters is not None:
                        axs[e//ncols, e%ncols].arrow(x, y, dx*0.6, dy*0.6, length_includes_head=True, color=colorFader("blue","red",mix=(row[f"emr{epoch}"]-min_emr)/(max_emr-min_emr)), head_width=0.1)
                    else:
                        axs[e//ncols, e%ncols].arrow(x, y, dx*0.6, dy*0.6, length_includes_head=True, color="grey", head_width=0.1)
                        
                if location_vector is not None:
                    plt.scatter(self.demes["xcoord"], self.demes["ycoord"], zorder=2, c=location_vector[self.demes.index], vmin=0, cmap="Oranges", s=25)
                if (e == 0) and isinstance(self.samples, pd.DataFrame) and show_samples:
                    counts = self.samples["deme"].value_counts().reset_index()
                    counts = counts.merge(self.demes, how="left", left_on="deme", right_on="id").loc[:,["id", "xcoord", "ycoord", "count"]]
                    axs[e//ncols, e%ncols].scatter(counts["xcoord"], counts["ycoord"], color="orange", s=counts["count"]*10, zorder=3)
                if color_demes:
                    deme_suitabilities_at_time = self.get_all_deme_suitabilities_at_time(times[e])
                    axs[e//ncols, e%ncols].scatter(self.demes["xcoord"], self.demes["ycoord"], c=deme_suitabilities_at_time, vmin=range_of_deme_suitabilities[0], vmax=range_of_deme_suitabilities[1], zorder=2)
                if title is None:
                    axs[e//ncols, e%ncols].set_title(times[e], fontname="Georgia")
                else:
                    axs[e//ncols, e%ncols].set_title(title, fontname="Georgia")
            axs[e//ncols, e%ncols].set_aspect("equal", 'box')
            axs[e//ncols, e%ncols].axis("off")

        if save_to != None:
            plt.savefig(save_to, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def build_transition_matrices(self, m, a=1):
        """Builds the transition matrix based on the world map and migration rates

        row is the starting location, column is the next location

        Parameters
        ----------
        migration_rates : np.array or list
            Order list of instantaneous migration rate along that connection type

            OLD:
            Keys are the connection type and values are the instantaneous migration
            rate along that connection
        
        Returns
        -------
        transition_matrix : np.array
        """
        transition_matrix = np.zeros((len(self.epochs), len(self.demes),len(self.demes)))
        e = 0
        for time in self.epochs:
            for _,connection in self.connections.iterrows():
                i_0 = self.demes.loc[self.demes["id"]==connection["deme_0"]].index[0]
                i_1 = self.demes.loc[self.demes["id"]==connection["deme_1"]].index[0]
                transition_matrix[e, i_0, i_1] = (m * self.get_connection_parameter_at_time("suitability_ratio", connection["id"], time)**a) * self.get_connection_parameter_at_time("migration_modifier", connection["id"], time)
            diag = -np.sum(transition_matrix[e], axis=1)
            np.fill_diagonal(transition_matrix[e], diag)
            e += 1
        return transition_matrix
    
    def _build_sample_locations_array(self):
        """

        Returns
        -------
        sample_locations_array
        sample_ids
        """

        if (self.samples is None):
            raise RuntimeError("No samples provided. Check that you've added samples to your WorldMap.")
        sample_locations_array = np.zeros((len(self.samples), len(self.demes)), dtype="float64")
        sample_ids = np.array(self.samples["id"])
        for i, sample in self.samples.iterrows():
            sample_locations_array[i,self.demes.loc[self.demes["id"]==sample["deme"]].index[0]] = 1
        return sample_locations_array, sample_ids
    
    def convert_to_networkx_graph(self):
        """Converts world map to undirected networkx graph
        """

        G = nx.Graph()
        G.add_nodes_from(self.demes.id)
        connections = []
        for _,row in self.connections.iterrows():
            connections.append((row["deme_0"], row["deme_1"]))
        G.add_edges_from(connections)
        return G
    
    def check_if_fully_connected(self, verbose=False):
        """Checks that all demes are connected and accessible

        This is important if there are coalescence events between nodes in
        inaccessible demes. Should raise warning for users (or fail).

        Parameters
        ----------
        verbose : bool
            If True and more than one component in graph, prints the nodes in
            each component
        """

        nx_graph = self.convert_to_networkx_graph()
        components = nx.connected_components(nx_graph)
        if len(components) > 1:
            if verbose:
                print("Not all demes are accessible.\nSeparated world map components:")
                for c in components:
                    print(c)
            return False
        else:
            return True

def precalculate_transitions(branch_lengths, transition_matrices, fast=True):
    """Calculates the transition probabilities between demes for each branch length

    Parameters
    ----------
    branch_lengths : np.ndarray
        Array of branch lengths of increasing size
    transition_matrix : np.ndarray
        Instantaneous migration rate matrix, output of WorldMap.build_transition_matrices()
    fast : bool
        Whether to use the faster but less numerically stable algorithm (default is True)
    
    Returns
    -------
    transitions : np.ndarray
        3D array with transitions probabilities for each branch length
    """

    all_transitions = []
    for e in range(len(branch_lengths)):
        exponentiated = linalg.expm(transition_matrices[e])
        num_demes = exponentiated.shape[0]
        transitions = np.zeros((len(branch_lengths[e]), num_demes, num_demes), dtype="float64")
        if fast:
            transitions[0] = np.linalg.matrix_power(exponentiated, branch_lengths[e][0])
            for i in range(1, len(branch_lengths[e])):
                for j in range(i-1, -1, -1):
                    if branch_lengths[e][j] > 0:
                        power = branch_lengths[e][i] / branch_lengths[e][j]
                        if power % 1 == 0:
                            transitions[i] = np.linalg.matrix_power(transitions[j], int(power))
                            break
                    else:
                        transitions[i] = np.linalg.matrix_power(exponentiated, branch_lengths[e][i])
                        break
                    if j == 0:
                        transitions[i] = np.linalg.matrix_power(exponentiated, branch_lengths[e][i])
        else:
            for i in range(len(branch_lengths[e])):
                transitions[i] = np.linalg.matrix_power(exponentiated, branch_lengths[e][i])
        all_transitions.append(transitions)
    return all_transitions

def ts_to_nx(ts):
    """Covert tskit.TreeSequence to networkx graph
    
    Parameters
    ----------
    ts : tskit.TreeSequence
        Input tree sequence

    Returns
    -------
    networkx.DiGraph
        Output networkx graph object
    """
    
    edges = []
    for edge in ts.tables.edges:
        edges.append((edge.child, edge.parent))
    return nx.from_edgelist(edges, create_using=nx.DiGraph)

def nx_bin_ts(ts, bins):
    """Uses networkx library to simplify a tree by grouping nodes into discrete time bins

    Parameters
    ----------
    ts : tskit.TreeSequence
        Input tree sequence
    bins : list
        List of time bins to group nodes into

    Returns
    -------
    ts_out : tskit.TreeSequence
        Output tree sequence
    """

    nx_ts = ts_to_nx(ts=ts)
    node_time_bins = np.digitize(ts.tables.nodes.time, bins, right=True)
    previously_removed = []
    for edge in ts.edges():
        if node_time_bins[edge.child] != node_time_bins[edge.parent]:
            if (edge.child, edge.parent) not in previously_removed:
                nx_ts.remove_edge(edge.child, edge.parent)
                previously_removed.append((edge.child, edge.parent))
    ccs = list(nx.connected_components(nx_ts.to_undirected()))
    collapsed_node_list = [-1 for i in range(ts.num_nodes)]
    for group,cc in enumerate(ccs):
        for node in cc:
            collapsed_node_list[node] = group

    tables = tskit.TableCollection(sequence_length=ts.sequence_length)
    new_node_table = tables.nodes
    new_edge_table = tables.edges
    
    node_id_map = []
    for node in ts.nodes():
        if collapsed_node_list[node.id] not in node_id_map:
            if node.flags == 1:
                new_node_table.add_row(
                    flags=1,
                    time=bins[node_time_bins[node.id]],
                    population=-1,
                    individual=-1,
                    metadata=node.metadata
                )
            else:
                new_node_table.add_row(
                    flags=0,
                    time=bins[node_time_bins[node.id]],
                    population=-1,
                    individual=-1,
                    metadata=node.metadata
                )
            node_id_map.append(collapsed_node_list[node.id])

    for edge in ts.edges():
        new_child = node_id_map.index(collapsed_node_list[edge.child])
        new_parent = node_id_map.index(collapsed_node_list[edge.parent])
        if new_child != new_parent:
            if new_node_table.time[new_parent] <= new_node_table.time[new_child]:
                raise RuntimeError(new_child, new_parent, new_node_table.time[new_child], new_node_table.time[new_parent], edge.child, edge.parent, ts.node(edge.child).time, ts.node(edge.parent).time)
            new_edge_table.add_row(
                left=edge.left,
                right=edge.right,
                parent=new_parent,
                child=new_child,
                metadata=edge.metadata
            )
    tables.sort()
    ts_out = tables.tree_sequence()
    return ts_out

def _deconstruct_tree(tree, epochs, time_bins=None):
    num_nodes = len(tree.postorder())
    parents = np.full(num_nodes, -1, dtype="int64")
    branch_above = np.zeros((len(epochs), num_nodes), dtype="int64")
    time_bin_widths = np.full(num_nodes, -1, dtype="int64")
    ids_asc_time = np.full(num_nodes, -1, dtype="int64")
    for i,node in enumerate(tree.nodes(order="timeasc")):
        node_time = tree.time(node)
        parent = tree.parent(node)
        if parent != -1:
            parent_time = tree.time(parent)
            starting_epoch = np.digitize(node_time, epochs)-1
            ending_epoch = np.digitize(parent_time, epochs)-1
            if starting_epoch == ending_epoch:
                branch_above[starting_epoch, node] = math.ceil(parent_time - node_time)
            else:
                branch_above[starting_epoch, node] = math.ceil(epochs[starting_epoch+1] - node_time)
                for e in range(starting_epoch+1, ending_epoch):
                    branch_above[e, node] = epochs[e+1] - epochs[e]
                branch_above[ending_epoch, node] = math.ceil(parent_time - epochs[ending_epoch])
        ids_asc_time[i] = node
        parents[node] = parent
        if time_bins is not None:
            i = next(j for j, e in enumerate(time_bins) if e >= node_time)
            width = max(1, time_bins[i] - time_bins[i - 1])
            time_bin_widths[node] = width
        else:
            time_bin_widths[node] = 1
    return parents, branch_above, time_bin_widths, ids_asc_time

def _deconstruct_trees(trees, epochs, time_bins=None):
    """

    Note: It would be great if pl and bal were numpy.ndarray, but that would force
    the trees to have the same number of nodes, which is unrealistic.
    """
    
    pl = []
    bal = []
    tbw = []
    iat = []
    all_branch_lengths = [[] for e in epochs]
    for tree in trees:
        parents, branch_above, time_bin_widths, ids_asc_time = _deconstruct_tree(tree, epochs, time_bins=time_bins)
        pl.append(parents)
        bal.append(branch_above)
        tbw.append(time_bin_widths)
        iat.append(ids_asc_time)
        for e in range(len(epochs)):
            all_branch_lengths[e].extend(branch_above[e])
    unique_branch_lengths = []
    for e in range(len(epochs)):
        unique_branch_lengths.append(np.unique(all_branch_lengths[e]))
    return pl, bal, tbw, iat, unique_branch_lengths

@njit()
def _likelihood_of_tree_log_faster(
        parents,
        branch_above,
        unique_branch_lengths,
        time_bin_widths,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        precomputed_transitions,
        stationary_distribution_log
    ):

    num_demes = len(sample_locations_array[0])
    messages = np.zeros((len(parents), num_demes), dtype="float64")
    composite_across_roots = 0
    for id in ids_asc_time:
        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
            current_pos[current_pos <= 1e-99] = 1e-99
            current_pos = np.log(current_pos)
        else:
            children = np.where(parents==id)[0]
            current_pos = np.zeros(num_demes, dtype="float64")
            for child in children:
                current_pos += messages[child]
        parent = parents[id]
        if parent != -1:
            bl = branch_above[:,id]
            included_epochs = np.where(bl > 0)[0]
            P = np.eye(num_demes)
            if (len(included_epochs) > 0):
                for epoch in included_epochs:
                    bl_index = np.where(unique_branch_lengths[epoch]==bl[epoch])[0][0]
                    P = np.dot(P, precomputed_transitions[epoch][bl_index])
            P = np.log(np.maximum(P, 1e-99))
            messages[id] = logsumexp_custom(P + current_pos, axis=1)
        else:   # collect roots here
            composite_across_roots += logsumexp_custom(stationary_distribution_log + current_pos, axis=0)
    return composite_across_roots

@njit()
def _likelihood_of_tree_log(
        parents,
        branch_above,
        unique_branch_lengths,
        time_bin_widths,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        precomputed_transitions,
        stationary_distribution_log
    ):

    num_demes = len(sample_locations_array[0])
    messages = np.zeros((len(parents), num_demes), dtype="float64")
    composite_across_roots = 0
    for id in ids_asc_time:
        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
            current_pos[current_pos <= 1e-99] = 1e-99
            current_pos = np.log(current_pos)
        else:
            children = np.where(parents==id)[0]
            current_pos = np.zeros(num_demes, dtype="float64")
            for child in children:
                current_pos += messages[child]
        parent = parents[id]
        if parent != -1:
            bl = branch_above[:,id]
            included_epochs = np.where(bl > 0)[0]
            P = np.eye(num_demes)
            if (len(included_epochs) > 0):
                for epoch in included_epochs:
                    bl_index = np.where(unique_branch_lengths[epoch]==bl[epoch])[0][0]
                    P = np.dot(P, precomputed_transitions[epoch][bl_index])
            P = np.log(np.maximum(P, 1e-99))
            messages[id] = logsumexp_custom(P + current_pos, axis=1)
        else:   # collect roots here
            composite_across_roots += logsumexp_custom(stationary_distribution_log + current_pos, axis=0)
    return composite_across_roots

@njit(parallel=True)
def _process_trees(
        parents,
        branch_above,
        unique_branch_lengths,
        time_bin_widths,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        precomputed_transitions,
        stationary_distribution,
        stationary_distribution_log
    ):

    composite_likelihood = 0
    for i in prange(len(branch_above)):
        like = _likelihood_of_tree_log_faster(
            parents=parents[i],
            branch_above=branch_above[i],
            unique_branch_lengths=unique_branch_lengths,
            time_bin_widths=time_bin_widths[i],
            ids_asc_time=ids_asc_time[i],
            sample_locations_array=sample_locations_array,
            sample_ids=sample_ids,
            precomputed_transitions=precomputed_transitions,
            stationary_distribution_log=stationary_distribution_log
        )
        composite_likelihood += like
    return abs(composite_likelihood)

def _calc_composite_likelihood_for_parameters(
        migration_rates,
        world_map,
        parents,
        branch_above,
        unique_branch_lengths,
        time_bin_widths,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        output_file=None
    ):

    migration_rates[0] = np.exp(migration_rates[0])

    return _calc_composite_likelihood_for_rates(
        migration_rates=migration_rates,
        world_map=world_map,
        parents=parents,
        branch_above=branch_above,
        unique_branch_lengths=unique_branch_lengths,
        time_bin_widths=time_bin_widths,
        ids_asc_time=ids_asc_time,
        sample_locations_array=sample_locations_array,
        sample_ids=sample_ids,
        output_file=output_file
    )

def _calc_composite_likelihood_for_log_rates(
        migration_rates,
        world_map,
        parents,
        branch_above,
        unique_branch_lengths,
        time_bin_widths,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        output_file=None
    ):

    return _calc_composite_likelihood_for_rates(
        migration_rates=np.exp(migration_rates),
        world_map=world_map,
        parents=parents,
        branch_above=branch_above,
        unique_branch_lengths=unique_branch_lengths,
        time_bin_widths=time_bin_widths,
        ids_asc_time=ids_asc_time,
        sample_locations_array=sample_locations_array,
        sample_ids=sample_ids,
        output_file=output_file
    )

def _calc_composite_likelihood_for_rates(
        migration_rates,
        world_map,
        parents,
        branch_above,
        unique_branch_lengths,
        time_bin_widths,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        verbose=False,
        output_file=None
    ):

    #start = time.time()
    if len(migration_rates) > 1:
        transition_matrices = world_map.build_transition_matrices(m=migration_rates[0], a=migration_rates[1])
    else:
        transition_matrices = world_map.build_transition_matrices(m=migration_rates[0])
    #print(time.time() - start, flush=True)

    precomputed_transitions = precalculate_transitions(
        branch_lengths=unique_branch_lengths,
        transition_matrices=transition_matrices
    )

    # https://people.duke.edu/~ccc14/sta-663-2016/homework/Homework02_Solutions.html
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrices[-1].T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    stationary_distribution = w/np.sum(w)
    stationary_distribution[stationary_distribution <= 1e-99] = 1e-99
    stationary_distribution_log = np.log(stationary_distribution)

    composite_likelihood = _process_trees(
        parents=parents,
        branch_above=branch_above,
        unique_branch_lengths=unique_branch_lengths,
        time_bin_widths=time_bin_widths,
        ids_asc_time=ids_asc_time,
        sample_locations_array=sample_locations_array,
        sample_ids=sample_ids,
        precomputed_transitions=precomputed_transitions,
        stationary_distribution=stationary_distribution,
        stationary_distribution_log=stationary_distribution_log
    )
    if output_file is not None:
        with open(output_file, "a") as outfile:
            outfile.write(f"{migration_rates}\t{-composite_likelihood}\n")
    elif verbose:
        print(migration_rates, -composite_likelihood)
    return composite_likelihood

def run(
        demes_path,
        connections_path,
        samples_path,
        trees_dir_path,
        chop_time=None,
        time_bins=None,
        num_walkers=10,
        num_iters=1,
        output_file=None
    ):
    
    if output_file is not None:
        with open(output_file, "w") as outfile:
            outfile.write("rates\tloglikelihood\n")

    demes = pd.read_csv(demes_path, sep="\t")
    connections = pd.read_csv(connections_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes, connections, samples)
    
    unique_suitability_ratios = np.unique(world_map.suitability_ratios)

    if trees_dir_path[-1] != "/":
        trees_dir_path += "/"
    
    trees = []
    for ts in glob(trees_dir_path+"*"):
        tree = tskit.load(ts)
        if chop_time is not None:
            decap = tree.decapitate(chop_time)
            tree = decap.subset(nodes=np.where(decap.tables.nodes.time <= chop_time)[0])
        if time_bins is not None:
            tree = nx_bin_ts(tree, time_bins)
        trees.append(tree.first())

    sample_locations_array, sample_ids = world_map._build_sample_locations_array()
    parents, branch_above, time_bin_widths, ids_asc_time, unique_branch_lengths = _deconstruct_trees(trees=trees, epochs=world_map.epochs, time_bins=time_bins)

    if len(unique_suitability_ratios) > 1:
        bounds = [(-10, 3), (-5, 5)]
    else:
        bounds = [(-10, 3)]
    
    res = shgo(
        _calc_composite_likelihood_for_parameters,
        bounds=bounds,
        n=num_walkers,
        iters=num_iters,
        sampling_method="sobol",
        args=(
            world_map,
            parents,
            branch_above,
            unique_branch_lengths,
            time_bin_widths,
            ids_asc_time,
            sample_locations_array,
            sample_ids,
            output_file
        ),
        minimizer_kwargs={"ftol":0.001}
    )

    final = res.x.copy()
    final[0] = np.exp(final[0])
    return final, -res.fun

def run_for_parameters(
        parameters,
        demes_path,
        connections_path,
        samples_path,
        trees_dir_path,
        chop_time=None,
        time_bins=None
    ):

    demes = pd.read_csv(demes_path, sep="\t")
    connections = pd.read_csv(connections_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes, connections, samples)
    unique_suitability_ratios = np.unique(world_map.suitability_ratios)

    if trees_dir_path[-1] != "/":
        trees_dir_path += "/"
    
    trees = []
    for ts in glob(trees_dir_path+"*"):
        tree = tskit.load(ts)
        if chop_time is not None:
            decap = tree.decapitate(chop_time)
            tree = decap.subset(nodes=np.where(decap.tables.nodes.time <= chop_time)[0])
        if time_bins is not None:
            tree = nx_bin_ts(tree, time_bins)
        trees.append(tree.first())

    sample_locations_array, sample_ids = world_map._build_sample_locations_array()
    parents, branch_above, time_bin_widths, ids_asc_time, unique_branch_lengths = _deconstruct_trees(trees=trees, epochs=world_map.epochs, time_bins=time_bins)

    parameters[0] = np.log(parameters[0])

    likelihood = _calc_composite_likelihood_for_parameters(
        parameters,
        world_map,
        parents,
        branch_above,
        unique_branch_lengths,
        time_bin_widths,
        ids_asc_time,
        sample_locations_array,
        sample_ids
    )
    
    return -likelihood













#OLD
def run_for_rate_combo(
        migration_rates,
        demes_path,
        samples_path,
        trees_dir_path,
        time_bins=None
    ):
    
    demes = pd.read_csv(demes_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes, samples)

    if trees_dir_path[-1] != "/":
        trees_dir_path += "/"
    
    trees = []
    for ts in glob(trees_dir_path+"*"):
        tree = tskit.load(ts)
        if time_bins is not None:
            tree = nx_bin_ts(tree, time_bins)
        trees.append(tree.first())

    sample_locations_array, sample_ids = world_map._build_sample_locations_array()
    parents, branch_above, time_bin_widths, ids_asc_time, unique_branch_lengths = _deconstruct_trees(trees=trees, epochs=world_map.epochs, time_bins=time_bins)

    comp_like = _calc_composite_likelihood_for_rates(
        migration_rates=migration_rates,
        world_map=world_map,
        parents=parents,
        branch_above=branch_above,
        unique_branch_lengths=unique_branch_lengths,
        time_bin_widths=time_bin_widths,
        ids_asc_time=ids_asc_time,
        sample_locations_array=sample_locations_array,
        sample_ids=sample_ids
    )

    return comp_like

#OLD
def locate(
        demes_path,
        samples_path,
        tree_path,
        migration_rates,
        asymmetric=False
    ):

    demes = pd.read_csv(demes_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes, samples, asymmetric)
    tree = tskit.load(tree_path).first()

    # ADD A CHECK THAT THE NUMBER OF RATES PROVIDED MATCHES THOSE NEEDED

    locs = locate_nodes_in_tree(
        tree=tree,
        world_map=world_map,
        migration_rates=migration_rates
    )

    return locs


def _get_messages(
        parents,
        branch_above,
        time_bin_widths,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        transition_matrices
    ):

    num_demes = len(sample_locations_array[0])

    messages = {}

    for id in ids_asc_time:    
        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
        else:
            children = np.where(parents==id)[0]
            incoming_messages = np.zeros((len(children), num_demes))
            for j,child in enumerate(children):
                incoming_messages[j] = messages[(child, id)]
            current_pos = np.prod(incoming_messages, axis=0)
        parent = parents[id]
        if parent != -1:
            bl = branch_above[:,id]
            included_epochs = np.where(bl > 0)[0]
            if (len(included_epochs) > 0):
                P = np.eye(num_demes)
                for epoch in included_epochs:
                    P = np.dot(P, linalg.expm(transition_matrices[epoch]*bl[epoch]))
                messages[(id, parent)] = np.dot(P, current_pos)
            else:
                messages[(id, parent)] = current_pos
    
    for id in ids_asc_time[-1::-1]:
        incoming_keys = [key for key in messages.keys() if key[1] == id]
        children = np.where(parents==id)[0]
        for child in children:
            if child not in sample_ids:
                if id in sample_ids:
                    current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
                else:
                    incoming_keys_filtered = [key for key in incoming_keys if key[0] != child]
                    incoming_messages = np.zeros((len(incoming_keys_filtered), num_demes))
                    for j,key in enumerate(incoming_keys_filtered):
                        incoming_messages[j] = messages[key]
                    current_pos = np.prod(incoming_messages, axis=0)
                bl = branch_above[:,child]
                included_epochs = np.where(bl > 0)[0]
                if (len(included_epochs) > 0):
                    P = np.eye(num_demes)
                    for epoch in included_epochs:
                        P = np.dot(P, linalg.expm(transition_matrices[epoch]*bl[epoch]))
                    messages[(id, child)] = np.dot(current_pos, P)
                else:
                    messages[(id, child)] = current_pos
    return messages

#OLD
def locate_nodes_in_tree(
        tree,
        world_map,
        migration_rates
    ):

    transition_matrices = world_map.build_transition_matrices(m=migration_rates[0])

    parents, branch_above, time_bin_widths, ids_asc_time = _deconstruct_tree(tree, world_map.epochs)
    
    sample_locations_array, sample_ids = world_map._build_sample_locations_array()

    messages = _get_messages(
        parents=parents,
        branch_above=branch_above,
        time_bin_widths=time_bin_widths,
        ids_asc_time=ids_asc_time,
        sample_locations_array=sample_locations_array,
        sample_ids=sample_ids,
        transition_matrices=transition_matrices
    )
    
    L = np.zeros((len(parents), len(world_map.demes)))
    for id in ids_asc_time:
        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
        else:
            incoming_keys = [key for key in messages.keys() if key[1] == id]
            incoming_messages = np.zeros((len(incoming_keys), len(world_map.demes)))
            for j,key in enumerate(incoming_keys):
                incoming_messages[j] = messages[key]
            current_pos = np.prod(incoming_messages, axis=0)
        L[id] = current_pos

    L = L/L.sum(axis=1, keepdims=True)
    return L


def ancs(tree, u):
    """Find all of the ancestors above a node for a tree

    Taken directly from https://github.com/tskit-dev/tskit/issues/2706

    Parameters
    ----------
    tree : tskit.Tree
    u : int
        The ID for the node of interest

    Returns
    -------
    An iterator over the ancestors of u in this tree
    """

    u = tree.parent(u)
    while u != -1:
         yield u
         u = tree.parent(u)


def track_lineage_in_tree(
        node,
        times,
        tree,
        world_map,
        migration_rates
    ):

    transition_matrices = world_map.build_transition_matrices(m=migration_rates[0])

    parents, branch_above, time_bin_widths, ids_asc_time = _deconstruct_tree(tree, world_map.epochs)
    
    sample_locations_array, sample_ids = world_map._build_sample_locations_array()

    messages = _get_messages(
        parents=parents,
        branch_above=branch_above,
        time_bin_widths=time_bin_widths,
        ids_asc_time=ids_asc_time,
        sample_locations_array=sample_locations_array,
        sample_ids=sample_ids,
        transition_matrices=transition_matrices
    )

    ancestors = [node] + list(ancs(tree=tree, u=node))

    node_times = []
    for a in ancestors:
        node_times.append(int(tree.time(a)))

    pc_combos = []
    for t in times:
        for i,v in enumerate(node_times):
            if v > t:
                child = ancestors[i-1]
                parent = ancestors[i]
                break
            elif v == t:
                child = ancestors[i]
                parent = ancestors[i]
                break
        pc_combos.append((child, parent))

    L = np.zeros((len(pc_combos), len(world_map.demes)))
    for element,node_combo in enumerate(pc_combos):
        print(L)
        if node_combo[0] == node_combo[1]:
            if node_combo[0] in sample_ids:
                node_pos = sample_locations_array[np.where(sample_ids==node_combo[0])[0][0]]
            else:
                incoming_keys = [key for key in messages.keys() if key[1] == node_combo[0]]
                incoming_messages = np.zeros((len(incoming_keys), len(world_map.demes)))
                for j,key in enumerate(incoming_keys):
                    incoming_messages[j] = messages[key]
                node_pos = np.prod(incoming_messages, axis=0)
        else:
            if node_combo[0] in sample_ids:
                child_pos = sample_locations_array[np.where(sample_ids==node_combo[0])[0][0]]
            else:
                incoming_keys_child = [key for key in messages.keys() if key[1] == node_combo[0]]
                incoming_messages_child = np.array([messages[income] for income in incoming_keys_child if income[0] != node_combo[1]])
                if len(incoming_messages_child) > 0:
                    child_pos = np.prod(incoming_messages_child, axis=0)
                else:
                    print(f"Node {node_combo[0]} does not have incoming messages. Potential error?")
                    child_pos = np.ones((1,len(world_map.demes)))[0]
            if node_combo[1] in sample_ids:
                parent_pos = sample_locations_array[np.where(sample_ids==node_combo[1])[0][0]]
            else:
                incoming_keys_parent = [key for key in messages.keys() if key[1] == node_combo[1]]
                incoming_messages_parent = np.array([messages[income] for income in incoming_keys_parent if income[0] != node_combo[0]])
                if len(incoming_messages_parent) > 0:
                    parent_pos = np.prod(incoming_messages_parent, axis=0)
                else:
                    print(f"Node {node_combo[0]} does not have incoming messages. Potential error?")
                    parent_pos = np.ones((1,len(world_map.demes)))[0]
            
            branch_length_to_child = int(times[element] - tree.time(node_combo[0]))
            bl_child = branch_above[:, node_combo[0]].copy()
            bl_parent = bl_child.copy()
            whats_left = branch_length_to_child
            for e in range(len(world_map.epochs)):
                current = bl_parent[e].copy()
                if current >= whats_left:
                    bl_parent[e] -= whats_left
                else:
                    bl_parent[e] = 0
                whats_left -= current
            bl_child -= bl_parent

            included_epochs = np.where(bl_child > 0)[0]
            outgoing_child_message = child_pos
            if (len(included_epochs) > 0):
                P = np.eye(len(world_map.demes))
                for epoch in included_epochs:
                    P = np.dot(P, linalg.expm(transition_matrices[epoch]*bl_child[epoch]))
                outgoing_child_message = np.dot(P, child_pos)

            included_epochs = np.where(bl_parent > 0)[0]
            outgoing_parent_message = parent_pos
            if (len(included_epochs) > 0):
                P = np.eye(len(world_map.demes))
                for epoch in included_epochs:
                    P = np.dot(P, linalg.expm(transition_matrices[epoch]*bl_parent[epoch]))
                outgoing_parent_message = np.dot(parent_pos, P)
            
            node_pos = np.multiply(outgoing_child_message, outgoing_parent_message)

        L[element] = node_pos

    L = L/L.sum(axis=1, keepdims=True)
    
    return L