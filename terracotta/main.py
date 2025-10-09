import pandas as pd
import matplotlib.pyplot as plt
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
from glob import glob


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def _calc_optimal_organization_of_suplots(num_plots):
    ncols = math.ceil(math.sqrt(num_plots))
    nrows = math.ceil(num_plots/ncols)
    return nrows, ncols

class WorldMap:
    """Stores a map of the demes

    Attributes
    ----------
    demes : pandas.DataFrame
    deme_types : numpy.ndarray
    samples : pandas.DataFrame
    epochs : numpy.ndarray
    connections : list of pandas.DataFrames
    connection_types : list of tuples
        All possible connection types, even if not observed in the map
    """

    def __init__(self, demes, samples=None):
        """Initializes the WorldMap object

        Parameters
        ----------
        demes : pandas.DataFrame
            Five columns: "id", "xcoord", "ycoord", "type", "neighbours" (note the "u" in neighbours).
            See README.md for more details about `demes.tsv` file structure
        samples : pandas.DataFrame
        """

        self.demes = demes.copy()
        
        self.demes["type"] = self.demes["type"].astype(str)
        
        formatted_dt = []
        for dt in self.demes["type"]:
            if ":" in dt:
                formatted_dt.append(dt)
            else:
                formatted_dt.append(f"0:{dt}")
        self.demes["type"] = formatted_dt

        converted_neighbors = []
        for index,row in self.demes.iterrows():
            neighbors = str(row["neighbours"]).split(",")
            int_neighbors = []
            for neighbor in neighbors:
                neighbor = int(neighbor)
                int_neighbors.append(neighbor)
            converted_neighbors.append(int_neighbors)
        self.demes["neighbours"] = converted_neighbors

        self.sample_location_vectors = None
        if samples is not None:
            self.samples = samples.copy()
            self.sample_location_vectors = self._build_sample_location_vectors()
    
        epochs = [0]
        deme_types = []
        for dt in self.demes["type"]:
            epoch_formatter = dt.replace(",", ":").split(":")
            for epoch_assignment in epoch_formatter[::2]:
                epoch_assignment = int(epoch_assignment)
                if epoch_assignment not in epochs:
                    epochs.append(epoch_assignment)
            for type_assignment in epoch_formatter[1::2]:
                type_assignment = int(type_assignment)
                if type_assignment not in deme_types:
                    deme_types.append(type_assignment)
        self.epochs = np.sort(epochs)
        deme_types = np.sort(deme_types)

        self.deme_types = deme_types

        connection_types = []
        for type0 in deme_types:
            for type1 in deme_types:
                connection_types.append((type0, type1))
                #if type1 > type0:
                #    connection_types.append((type0, type1))
                #else:
                #    connection_types.append((type1, type0))

        self.connection_types = connection_types

        connections = []
        for time_period in self.epochs:
            epoch_connections = []
            for index,row in demes.iterrows():
                row_type = self.get_deme_type_at_time(row["id"], time_period)
                neighbors = str(row["neighbours"]).split(",")
                for neighbor in neighbors:
                    neighbor = int(neighbor)
                    neighbor_type = self.get_deme_type_at_time(neighbor, time_period)
                    ct = connection_types.index((row_type, neighbor_type))
                    epoch_connections.append({"deme_0":row["id"], "deme_1":neighbor, "type":ct})
                    #if row["id"] < neighbor:
                    #    neighbor_type = self.get_deme_type_at_time(neighbor, time_period)
                    #    if neighbor_type > row_type:
                    #        ct = connection_types.index((row_type, neighbor_type))   #deme_types[neighbor_type]
                    #    else:
                    #        ct = connection_types.index((neighbor_type, row_type))   #deme_types[row_type]
                    #    #print(row["id"], neighbor, row_type, neighbor_type, ct)
                    #    epoch_connections.append({"deme_0":row["id"], "deme_1":neighbor, "type":ct})
            epoch_connections = pd.DataFrame(epoch_connections)
            connections.append(epoch_connections)
        self.connections = connections

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
    
    def get_deme_type_string(self, id):
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
    
    def get_deme_type_at_time(self, id, time):
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
        deme_type : int
            Type of deme
        """

        deme_type = self.demes.loc[self.demes["id"]==id,"type"].iloc[0]
        epochs = deme_type.split(",")
        for t in range(len(epochs)-1):
            details = epochs[t].split(":")
            start = int(details[0])
            end = int(epochs[t+1].split(":")[0])
            if (time >= start) and (time < end):
                return int(details[1])
        return int(epochs[-1].split(":")[1])
    
    def get_all_deme_types_at_time(self, time):
        """

        Parameters
        ----------
        time : int or float
            Specified time of world map

        Returns
        -------
        types_at_time : pd.Series
            Deme types ordered by the demes dataframe

        """

        types_at_time = []
        for id in self.demes["id"]:
            types_at_time.append(self.get_deme_type_at_time(id, time))
        return pd.Series(types_at_time)
            
    def get_neighbors_of_deme(self, id):
        """Gets the neighbors of a deme

        Wrapper of pandas.DataFrame function. Assumes that deme IDs are unique.

        Parameters
        ----------
        id : int
            ID of deme

        Returns
        -------
        neighbors : pd.Series
            Contains all of the neighbors of specified deme
        """

        neighbors = self.demes.loc[self.demes["id"]==id,"neighbours"].iloc[0]
        return neighbors
    
    get_neighbours_of_deme = get_neighbors_of_deme

    def draw(
            self,
            figsize,
            times=None,
            color_connections=False,
            color_demes=False,
            save_to=None
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
        
        if times is None:
            times = self.epochs
        
        nrows, ncols = _calc_optimal_organization_of_suplots(num_plots=len(times))

        if color_connections:
            num_connection_types = len(self.connection_types)
            color_rnd = random.Random()
            color_rnd.seed(1)
            connection_colors = ["#"+''.join([color_rnd.choice("0123456789ABCDEF") for j in range(6)]) for i in range(num_connection_types)]

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        if len(times) == 1:
            axs = np.array([[axs]])
        elif nrows == 1:
            axs = np.array([axs])

        for e in range(nrows*ncols):
            if e < len(times):
                for _,row in self.connections[e].iterrows():
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

                    if color_connections:
                        color = connection_colors[row["type"]]
                    else:
                        color="grey"
                    axs[e//ncols, e%ncols].arrow(x, y, dx*0.6, dy*0.6, length_includes_head=True, color=color, head_width=0.1)
                deme_types_at_time = self.get_all_deme_types_at_time(times[e])
                if color_demes:
                    axs[e//ncols, e%ncols].scatter(self.demes["xcoord"], self.demes["ycoord"], c=deme_types_at_time, vmin=min(self.deme_types), vmax=max(self.deme_types), zorder=2)
                else:
                    axs[e//ncols, e%ncols].scatter(self.demes["xcoord"], self.demes["ycoord"], color="grey", zorder=2)
                axs[e//ncols, e%ncols].set_title(times[e], fontname="Georgia")
            axs[e//ncols, e%ncols].set_aspect("equal", 'box')
            axs[e//ncols, e%ncols].axis("off")

        if save_to != None:
            plt.savefig(save_to)
        plt.show()

    def draw_estimated_location(
            self,
            location_vector,
            figsize,
            title=None,
            show_samples=False,
            add_points=None,
            save_to=None
        ):
        """Draws the world map

        Uses matplotlib.pyplot

        Parameters
        ----------
        title : str
            Title is added to the top of the map
        show_samples : bool
            Whether to add samples to map
        add_points : list
            List of demes IDs to add a point to. Useful for no highlighting a deme
        save_to : str
            Path to save the map to
        """

        plt.figure(figsize=figsize)
        
        for index,row in self.connections.iterrows():
            deme_0 = self.get_coordinates_of_deme(row["deme_0"])
            deme_1 = self.get_coordinates_of_deme(row["deme_1"])
            plt.plot([deme_0[0], deme_1[0]], [deme_0[1], deme_1[1]], color="grey")
        plt.scatter(self.demes["xcoord"], self.demes["ycoord"], zorder=2, c=location_vector[0][self.demes.index], vmin=0, cmap="Oranges")
        if isinstance(self.samples, pd.DataFrame) and show_samples != False:
            if isinstance(show_samples, list):
                counts = self.samples[self.samples["id"].isin(show_samples)]["deme"].value_counts().reset_index()
                counts = counts.merge(self.demes, how="left", left_on="deme", right_on="id").loc[:,["id", "xcoord", "ycoord", "count"]]
                plt.scatter(counts["xcoord"], counts["ycoord"], color="orange", s=counts["count"]*50, zorder=3)
            else:
                counts = self.samples["deme"].value_counts().reset_index()
                counts = counts.merge(self.demes, how="left", left_on="deme", right_on="id").loc[:,["id", "xcoord", "ycoord", "count"]]
                plt.scatter(counts["xcoord"], counts["ycoord"], color="orange", s=counts["count"]*50, zorder=3)
        if isinstance(add_points, list):
            counts = Counter(add_points)
            counts = pd.DataFrame({"deme":counts.keys(), "count":counts.values()})
            counts = counts.merge(self.demes, how="left", left_on="deme", right_on="id").loc[:,["id", "xcoord", "ycoord", "count"]]
            plt.scatter(counts["xcoord"], counts["ycoord"], color="orange", s=counts["count"]*10, zorder=3)
        plt.gca().set_aspect("equal")
        plt.axis("off")
        if title != None:
            plt.title(title, fontname="Georgia", fontsize=50)
        if save_to != None:
            plt.savefig(save_to)
        plt.show()

    def build_transition_matrices(self, migration_rates):
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
        for e in range(len(self.epochs)):
            for _,connection in self.connections[e].iterrows():
                i_0 = self.demes.loc[self.demes["id"]==connection["deme_0"]].index[0]
                i_1 = self.demes.loc[self.demes["id"]==connection["deme_1"]].index[0]
                rate = migration_rates[connection["type"]]
                transition_matrix[e, i_0, i_1] = rate
            diag = -np.sum(transition_matrix[e], axis=1)
            np.fill_diagonal(transition_matrix[e], diag)
        return transition_matrix

    def _build_sample_location_vectors(self):
        """Builds sample location vectors based on the ordering of demes in the world map

        Returns
        -------
        sample_location_vectors : dict
        """

        zeros = np.zeros((1,len(self.demes)))
        sample_location_vectors = {}
        for index,row in self.samples.iterrows():
            sample_loc = zeros.copy()
            sample_loc[0,self.demes.loc[self.demes["id"]==row["deme"]].index[0]] = 1
            sample_location_vectors[row["id"]] = sample_loc
        return sample_location_vectors
    
    def _build_sample_locations_array(self):
        """

        Returns
        -------
        sample_locations_array
        sample_ids
        """

        if (self.samples is None) or (len(self.sample_location_vectors) == 0):
            raise RuntimeError("No samples provided. Check that you've added samples to your WorldMap.")
        sample_locations_array = np.zeros((len(self.samples), len(self.demes)), dtype="float64")
        sample_ids = np.zeros(len(self.samples), dtype="int64")
        counter = 0
        for sample in self.sample_location_vectors:
            sample_locations_array[counter] = self.sample_location_vectors[sample]
            sample_ids[counter] = sample
            counter += 1
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

@njit()
def _calc_tree_log_likelihood(
        parent_list,
        branch_above_list,
        roots,
        sample_ids,
        sample_location_vectors,
        branch_lengths,
        precomputed_transitions,
    ):
    """Calculates the log-likelihood of a tree

    Parameters
    ----------
    parent_list
    branch_above_list
    roots
    sample_ids
    sample_location_vectors
    branch_lengths
    precomputed_transitions
    precomputed_log

    Returns
    -------
    tree_likelihood
    root_log_likes
    """
    
    num_nodes = len(branch_above_list[0])
    num_demes = len(sample_location_vectors[0])

    log_messages = np.zeros((num_nodes, num_demes), dtype="float64")

    for counter in range(len(sample_ids)):
        i = sample_ids[counter]
        bl = branch_above_list[:, i]
        included_epochs = np.where(bl > 0)[0]
        if (len(included_epochs) > 0):
            transition_prob = np.eye(num_demes)
            for epoch in included_epochs:
                bl_index = np.where(branch_lengths[epoch]==bl[epoch])[0][0]
                transition_prob = np.dot(transition_prob, precomputed_transitions[epoch][bl_index])
            log_messages[i] = np.log(np.dot(sample_location_vectors[counter], transition_prob))
    
    for i in range(num_nodes):
        children_of_i = np.where(parent_list==i)[0]
        if len(children_of_i) > 0:
            incoming_log_messages = np.zeros((len(children_of_i), num_demes), dtype="float64")
            counter = 0
            for child in children_of_i:
                incoming_log_messages[counter] = log_messages[child]
                counter += 1
            summed_log_messages = np.sum(incoming_log_messages, axis=0)
            bl = branch_above_list[:, i]
            included_epochs = np.where(bl > 0)[0]
            if (len(included_epochs) > 0):
                transition_prob = np.eye(num_demes)
                for epoch in included_epochs:
                    bl_index = np.where(branch_lengths[epoch]==bl[epoch])[0][0]
                    transition_prob = np.dot(transition_prob, precomputed_transitions[epoch][bl_index])
                transition_prob_log = np.log(transition_prob)
                combined = summed_log_messages + transition_prob_log
                c = np.max(combined)
                log_sum_exp = c + np.log(np.sum(np.exp(combined - c), axis=1))
                outgoing_log_message = log_sum_exp
            else:
                outgoing_log_message = summed_log_messages
            log_messages[i] = outgoing_log_message
    root_log_likes = np.zeros((len(roots)), dtype="float64")
    counter = 0
    for r in roots:
        if r not in sample_ids:
            c = np.max(log_messages[r])
            root_log_likes[counter] = c + np.log(np.sum(np.exp(log_messages[r] - c), axis=0))
        counter += 1
    tree_likelihood = sum(root_log_likes)
    return tree_likelihood, root_log_likes

@njit(parallel=True)
def _parallel_process_trees(
        parents,
        branch_above,
        roots,
        sample_ids,
        sample_location_vectors,
        branch_lengths,
        precomputed_transitions
    ):
    """Calculates and combines the log-likelihood of every tree

    Parameters
    ----------
    children
    branch_above
    roots
    sample_ids
    sample_location_vectors
    branch_lengths
    precomputed_transitions
    precomputed_log

    Returns
    -------
    mr_log_like
    log_likelihoods
    """

    log_likelihoods = np.zeros(len(branch_above), dtype="float64")
    for i in prange(len(branch_above)):
        log_likelihoods[i] = _calc_tree_log_likelihood(
            parent_list=parents[i],
            branch_above_list=branch_above[i],
            roots=roots[i],
            sample_ids=sample_ids,
            sample_location_vectors=sample_location_vectors,
            branch_lengths=branch_lengths,
            precomputed_transitions=precomputed_transitions
        )[0]
    mr_log_like = sum(log_likelihoods)
    return mr_log_like, log_likelihoods

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

def calc_log_migration_rate_log_likelihood(log_migration_rates, world_map, parents, branch_above, roots, branch_lengths):
    migration_rates = np.exp(log_migration_rates)
    print("sf", flush=True)
    return calc_migration_rate_log_likelihood(migration_rates, world_map, parents, branch_above, roots, branch_lengths)

def calc_migration_rate_log_likelihood(migration_rates, world_map, parents, branch_above, roots, branch_lengths):
    """Calculates the composite log-likelihood of the specified migration rates across trees
    
    Loops through all trees and calculates the log-likelihood for each, before summing together.

    Parameters
    ----------
    migration_rates
    world_map
    parents
    branch_above
    roots
    branch_lengths

    Returns
    -------
    mr_log_like : float
        Log-likelihood of the specified migration rates
    """
    

    print("ye", flush=True)

    transition_matrices = world_map.build_transition_matrices(migration_rates=migration_rates)

    print("yes", flush=True)

    precomputed_transitions = precalculate_transitions(
        branch_lengths=branch_lengths,
        transition_matrices=transition_matrices
    )

    print("no", flush=True)
    
    sample_locations_array, sample_ids = world_map._build_sample_locations_array()
    like, like_list = _parallel_process_trees(
        parents=parents,
        branch_above=branch_above,
        roots=roots,
        sample_ids=sample_ids,
        sample_location_vectors=sample_locations_array,
        branch_lengths=branch_lengths,
        precomputed_transitions=precomputed_transitions
    )
    print(migration_rates, abs(like), flush=True)
    return abs(like)

def _deconstruct_trees(trees, epochs):
    """

    Note: It would be great if pl and bal were numpy.ndarray, but that would force
    the trees to have the same number of nodes, which is unrealistic.
    """
    
    pl = []
    bal = []
    roots = []
    all_branch_lengths = [[] for e in epochs]
    for tree in trees:
        num_nodes = len(tree.postorder())
        parents = np.full(num_nodes, -1, dtype="int64")
        branch_above = np.zeros((len(epochs), num_nodes), dtype="int64")
        for node in tree.nodes(order="timeasc"):
            parent = tree.parent(node)
            if parent != -1:
                node_time = tree.time(node)
                parent_time = tree.time(parent)
                starting_epoch = np.digitize(node_time, epochs)-1
                ending_epoch = np.digitize(parent_time, epochs)-1
                if starting_epoch == ending_epoch:
                    branch_above[starting_epoch, node] = parent_time - node_time
                else:
                    branch_above[starting_epoch, node] = epochs[starting_epoch+1] - node_time
                    for e in range(starting_epoch+1, ending_epoch):
                        branch_above[e, node] = epochs[e+1] - epochs[e]
                    branch_above[ending_epoch, node] = parent_time - epochs[ending_epoch]
            parents[node] = parent
        pl.append(parents)
        bal.append(branch_above)
        roots.append(np.where(parents==-1)[0])
        for e in range(len(epochs)):
            all_branch_lengths[e].extend(branch_above[e])
    unique_branch_lengths = []
    for e in range(len(epochs)):
        unique_branch_lengths.append(np.unique(all_branch_lengths[e]))
    return pl, bal, roots, unique_branch_lengths

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

def run(
        demes_path,
        samples_path,
        trees_dir_path,
        time_bins=None,
        output_file=None
    ):
    
    demes = pd.read_csv(demes_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes, samples)

    if trees_dir_path[-1] != "/":
        trees_dir_path += "/"
    
    trees = []
    for ts in glob(trees_dir_path+"*"):
        tree = tskit.load(ts).simplify()
        if time_bins is not None:
            tree = nx_bin_ts(tree, time_bins)
        trees.append(tree.first())
    pl, bal, r, ubl = _deconstruct_trees(trees=trees, epochs=world_map.epochs)  # needed to use numba

    print(pl)

    bounds = [(-20, 7) for rate in world_map.connection_types]

    res = shgo(
        calc_log_migration_rate_log_likelihood,
        bounds=bounds,
        n=100,
        iters=5,
        sampling_method="sobol",
        args=(world_map, pl, bal, r, ubl)
    )

    return res








def locate_nodes_in_tree(tree, world_map, migration_rates):
    transition_matrix = world_map.build_transition_matrices(migration_rates=migration_rates)

    print("Precalculating transitions...")
    precomputed_transitions = {}
    for node in tree.nodes(order="timeasc"):
        bl = tree.branch_length(node)
        if bl not in precomputed_transitions:
            where_next = linalg.expm(transition_matrix*bl)
            where_next[where_next < 0] = 0
            precomputed_transitions[bl] = where_next
    
    messages = {}
    for node in tree.nodes(order="timeasc"):
        bl = tree.branch_length(node)
        if bl > 0:
            if tree.is_sample(node):
                messages[(node, tree.parent(node))] = np.matmul(world_map.sample_location_vectors[node], precomputed_transitions[bl])
            else:
                children = tree.children(node)
                incoming_messages = [messages[(child, node)] for child in children]
                for i in range(1,len(incoming_messages)):
                    incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                    incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
                messages[(node, tree.parent(node))] = np.matmul(incoming_messages[0], precomputed_transitions[bl])
    for node in tree.nodes(order="timedesc"):
        children = tree.children(node)
        incoming_keys = [key for key in messages.keys() if key[1] == node]
        for child in children:
            incoming_messages = []
            for income in incoming_keys:
                if income[0] != child:
                    incoming_messages.append(messages[income])
            for i in range(1,len(incoming_messages)):
                incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
            bl = tree.branch_length(child)
            messages[(node, child)] = np.matmul(incoming_messages[0], precomputed_transitions[bl])
    node_locations = {}
    for node in tree.nodes(order="timeasc"):
        if tree.is_sample(node):
            node_locations[node] = world_map.sample_location_vectors[node]
        else:
            incoming_messages = [messages[key] for key in messages.keys() if key[1] == node]
            for i in range(1,len(incoming_messages)):
                incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
            node_locations[node] = incoming_messages[0]
    return node_locations

def _get_messages(tree, world_map, migration_rates):
    """
    """
    
    transition_matrix = world_map.build_transition_matrices(migration_rates=migration_rates)
    
    total_number_of_edges = tree.num_edges+1
    branch_lengths = np.zeros(total_number_of_edges, dtype="int64")
    edge_counter = 0
    for node in tree.nodes(order="timeasc"):
        branch_lengths[edge_counter] = int(tree.branch_length(node))
        edge_counter += 1
    branch_lengths = np.unique(np.array(branch_lengths))

    precomputed_transitions = precalculate_transitions(
        branch_lengths=branch_lengths,
        transition_matrix=transition_matrix
    )
    precomputed_transitions[precomputed_transitions <= 1e-99] = 1e-99   # ensures that this is important there aren't negatives from numerical instability
    precomputed_log = np.log(precomputed_transitions)

    messages = {}
    for node in tree.nodes(order="timeasc"):
        if tree.parent(node) != -1:
            bl = int(tree.branch_length(node))
            bl_index = np.where(branch_lengths==bl)[0][0]
            if tree.is_sample(node):
                messages[(node, tree.parent(node))] = np.matmul(world_map.sample_location_vectors[node], precomputed_transitions[bl_index])
            else:
                children = tree.children(node)
                incoming_messages = [messages[(child, node)] for child in children]
                if len(incoming_messages) > 0:
                    for i in range(1,len(incoming_messages)):
                        incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                        incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0]) 
                    messages[(node, tree.parent(node))] = np.matmul(incoming_messages[0], precomputed_transitions[bl_index])
                else:
                    messages[(node, tree.parent(node))] = np.ones((1,len(world_map.demes)))
    for node in tree.nodes(order="timedesc"):
        children = tree.children(node)
        incoming_keys = [key for key in messages.keys() if key[1] == node]
        for child in children:
            incoming_messages = []
            for income in incoming_keys:
                if income[0] != child:
                    incoming_messages.append(messages[income])
            if len(incoming_messages) > 0:
                for i in range(1,len(incoming_messages)):
                    incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                    incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
                bl = int(tree.branch_length(child))
                bl_index = np.where(branch_lengths==bl)[0][0]
                messages[(node, child)] = np.matmul(incoming_messages[0], precomputed_transitions[bl_index])
            else:
                messages[(node, child)] = np.ones((1,len(world_map.demes)))
    return messages

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

def track_lineage_over_time(
        sample,
        times,
        tree,
        world_map,
        migration_rates
    ):
    """Estimates the location probability distribution for an ancestral lineage

    Note: this function is very slow and could benefit from revisiting.

    Parameters
    ----------
        sample : int
            sample ID 
        times : list
            List of times (in generations before past)
        tree : tskit.Tree
            Input tree for a specified location
        world_map : WorldMap
            Map including the demes and sample locations
        migration_rates : np.array
            Rates of different connection types

    Returns
    -------
    positions : dict
        All of the positions for 
    """

    ancestors = [sample] + list(ancs(tree=tree, u=sample))
    node_times = []
    for a in ancestors:
        node_times.append(tree.time(a))
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
    messages = _get_messages(tree=tree, world_map=world_map, migration_rates=migration_rates)
    transition_matrix = world_map.build_transition_matrices(migration_rates=migration_rates)
    positions = {}
    for element,node_combo in enumerate(pc_combos):
        if node_combo[0] == node_combo[1]:
            if node_combo[0] in world_map.sample_location_vectors:
                node_pos = world_map.sample_location_vectors[node_combo[0]]
            else:
                incoming_keys = [key for key in messages.keys() if key[1] == node_combo[0]]
                incoming_messages = []
                for income in incoming_keys:
                    incoming_messages.append(messages[income])
                for i in range(1,len(incoming_messages)):
                    incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                    incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
                node_pos = incoming_messages[0]
        else:
            if node_combo[0] in world_map.sample_location_vectors:
                child_pos = world_map.sample_location_vectors[node_combo[0]]
            else:
                incoming_keys_child = [key for key in messages.keys() if key[1] == node_combo[0]]
                incoming_messages_child = []
                for income in incoming_keys_child:
                    if income[0] != node_combo[1]:
                        incoming_messages_child.append(messages[income])
                if len(incoming_keys_child) > 0:
                    for i in range(1,len(incoming_messages_child)):
                        incoming_messages_child[0] = np.multiply(incoming_messages_child[0], incoming_messages_child[i])
                        incoming_messages_child[0] = incoming_messages_child[0] / np.sum(incoming_messages_child[0])
                    child_pos = incoming_messages_child[0]
                else:
                    child_pos = np.ones((1,len(world_map.demes)))
            branch_length_to_child = int(times[element] - tree.time(node_combo[0]))
            outgoing_child_message = np.matmul(child_pos, np.linalg.matrix_power(linalg.expm(transition_matrix), branch_length_to_child))
            if node_combo[1] in world_map.sample_location_vectors:
                parent_pos = world_map.sample_location_vectors[node_combo[1]]
            else:
                incoming_keys_parent = [key for key in messages.keys() if key[1] == node_combo[1]]
                incoming_messages_parent = []
                for income in incoming_keys_parent:
                    if income[0] != node_combo[0]:
                        incoming_messages_parent.append(messages[income])
                if len(incoming_messages_parent) > 0:
                    for i in range(1,len(incoming_messages_parent)):
                        incoming_messages_parent[0] = np.multiply(incoming_messages_parent[0], incoming_messages_parent[i])
                        incoming_messages_parent[0] = incoming_messages_parent[0] / np.sum(incoming_messages_parent[0])
                    parent_pos = incoming_messages_parent[0]
                else:
                    parent_pos = np.ones((1,len(world_map.demes)))
            branch_length_to_parent = int(tree.time(node_combo[1]) - times[element])
            outgoing_parent_message = np.matmul(parent_pos, np.linalg.matrix_power(linalg.expm(transition_matrix), branch_length_to_parent))
            node_pos = np.multiply(outgoing_child_message, outgoing_parent_message)
            node_pos = node_pos / np.sum(node_pos)
        positions[times[element]] = node_pos
    return positions