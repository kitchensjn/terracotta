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
import sys


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

    def __init__(self, demes, samples=None, asymmetric=False):
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

        self.sample_location_array = None
        if samples is not None:
            self.samples = samples.copy()
            self.sample_location_array = self._build_sample_locations_array()
    
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

        self.asymmetric = asymmetric
        
        connection_types = []
        if self.asymmetric:
            for type0 in deme_types:
                for type1 in deme_types:
                    connection_types.append((type0, type1))
        else:
            for i,type0 in enumerate(deme_types):
                for type1 in deme_types[i:]:
                    connection_types.append((type0, type1))

        self.all_connection_types = np.array(connection_types)

        existing_connections = []
        connections = []
        for time_period in self.epochs:
            epoch_connections = []
            for index,row in demes.iterrows():
                row_type = self.get_deme_type_at_time(row["id"], time_period)
                neighbors = str(row["neighbours"]).split(",")
                for neighbor in neighbors:
                    neighbor = int(neighbor)
                    neighbor_type = self.get_deme_type_at_time(neighbor, time_period)
                    if self.asymmetric:
                        ct = connection_types.index((row_type, neighbor_type))
                    else:
                        if (row_type > neighbor_type):
                            ct = connection_types.index((neighbor_type, row_type))
                        else:
                            ct = connection_types.index((row_type, neighbor_type))
                    existing_connections.append(ct)
                    epoch_connections.append({"deme_0":row["id"], "deme_1":neighbor, "type":ct})
            epoch_connections = pd.DataFrame(epoch_connections)
            connections.append(epoch_connections)
        self.connections = connections
        self.existing_connection_types = np.unique(np.array(existing_connections))

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
            color_connections=False,
            color_demes=False,
            save_to=None,
            location_vector=None,
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

        if location_vector is None:
            times = self.epochs
            nrows, ncols = _calc_optimal_organization_of_suplots(num_plots=len(times))
        else:
            times = [0]
            nrows, ncols = 1, 1
            
        if color_connections:
            num_connection_types = len(self.existing_connection_types)
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

                    if self.asymmetric:
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
                            color = connection_colors[np.where(self.existing_connection_types==row["type"])[0][0]]
                        else:
                            color="grey"
                        axs[e//ncols, e%ncols].arrow(x, y, dx*0.6, dy*0.6, length_includes_head=True, color=color, head_width=0.1)
                    else:
                        axs[e//ncols, e%ncols].plot([deme_0[0], deme_1[0]], [deme_0[1], deme_1[1]], color="grey")
                if location_vector is not None:
                    plt.scatter(self.demes["xcoord"], self.demes["ycoord"], zorder=2, c=location_vector[self.demes.index], vmin=0, cmap="Oranges", s=500, marker="s")
                #if color_demes:
                #    deme_types_at_time = self.get_all_deme_types_at_time(times[e])
                #    axs[e//ncols, e%ncols].scatter(self.demes["xcoord"], self.demes["ycoord"], c=deme_types_at_time, vmin=min(self.deme_types), vmax=max(self.deme_types), zorder=2)
                #else:
                #    axs[e//ncols, e%ncols].scatter(self.demes["xcoord"], self.demes["ycoord"], color="grey", zorder=2)
                axs[e//ncols, e%ncols].set_title(times[e], fontname="Georgia")
            axs[e//ncols, e%ncols].set_aspect("equal", 'box')
            axs[e//ncols, e%ncols].axis("off")

        if save_to != None:
            plt.savefig(save_to)
        if show:
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
                rate = migration_rates[np.where(self.existing_connection_types==connection["type"])[0][0]]
                transition_matrix[e, i_0, i_1] = rate
            diag = -np.sum(transition_matrix[e], axis=1)
            np.fill_diagonal(transition_matrix[e], diag)
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

@njit()
def _calc_tree_log_likelihood(
        parent_list,
        branch_above_list,
        time_bin_widths,
        roots,
        sample_ids,
        sample_locations_array,
        branch_lengths,
        precomputed_transitions
    ):
    """Calculates the log-likelihood of a tree

    Parameters
    ----------
    parent_list
    branch_above_list
    roots
    sample_ids
    sample_locations_array
    branch_lengths
    precomputed_transitions
    precomputed_log

    Returns
    -------
    tree_likelihood
    root_log_likes
    """
    
    num_nodes = len(branch_above_list[0])
    num_demes = len(sample_locations_array[0])

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
            log_messages[i] = np.log(np.dot(sample_locations_array[counter], transition_prob))
    
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
                outgoing_log_message = log_sum_exp + np.log(time_bin_widths[i])
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
        time_bin_widths,
        roots,
        sample_ids,
        sample_locations_array,
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
    sample_locations_array
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
            time_bin_widths=time_bin_widths[i],
            roots=roots[i],
            sample_ids=sample_ids,
            sample_locations_array=sample_locations_array,
            branch_lengths=branch_lengths,
            precomputed_transitions=precomputed_transitions,
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

def calc_log_migration_rate_log_likelihood(log_migration_rates, world_map, parents, branch_above, time_bin_widths, roots, branch_lengths, output_file=None):
    migration_rates = np.exp(log_migration_rates)
    return calc_migration_rate_log_likelihood(migration_rates, world_map, parents, branch_above, time_bin_widths, roots, branch_lengths, output_file=None)

def calc_migration_rate_log_likelihood(migration_rates, world_map, parents, branch_above, time_bin_widths, roots, branch_lengths, output_file=None):
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

    transition_matrices = world_map.build_transition_matrices(migration_rates=migration_rates)

    precomputed_transitions = precalculate_transitions(
        branch_lengths=branch_lengths,
        transition_matrices=transition_matrices
    )

    sample_locations_array, sample_ids = world_map._build_sample_locations_array()
    like, like_list = _parallel_process_trees(
        parents=parents,
        branch_above=branch_above,
        time_bin_widths=time_bin_widths,
        roots=roots,
        sample_ids=sample_ids,
        sample_locations_array=sample_locations_array,
        branch_lengths=branch_lengths,
        precomputed_transitions=precomputed_transitions
    )
    if output_file is not None:
        print(migration_rates, abs(like), file=f)
    else:
        print(migration_rates, abs(like), flush=True)
    return abs(like)

def _deconstruct_tree(tree, epochs, time_bins=None):
    num_nodes = len(tree.postorder())
    parents = np.full(num_nodes, -1, dtype="int64")
    branch_above = np.zeros((len(epochs), num_nodes), dtype="int64")
    time_bin_widths = np.full(num_nodes, -1, dtype="int64")
    for node in tree.nodes(order="timeasc"):
        node_time = tree.time(node)
        parent = tree.parent(node)
        if parent != -1:
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
        if time_bins is not None:
            i = next(j for j, e in enumerate(time_bins) if e >= node_time)
            width = max(1, time_bins[i] - time_bins[i - 1])
            time_bin_widths[node] = width
        else:
            time_bin_widths[node] = 1
    return parents, branch_above, time_bin_widths

def _deconstruct_trees(trees, epochs, time_bins=None):
    """

    Note: It would be great if pl and bal were numpy.ndarray, but that would force
    the trees to have the same number of nodes, which is unrealistic.
    """
    
    pl = []
    bal = []
    tbw = []
    roots = []
    all_branch_lengths = [[] for e in epochs]
    for tree in trees:
        parents, branch_above, time_bin_widths = _deconstruct_tree(tree, epochs, time_bins=time_bins)
        pl.append(parents)
        bal.append(branch_above)
        tbw.append(time_bin_widths)
        roots.append(np.where(parents==-1)[0])
        for e in range(len(epochs)):
            all_branch_lengths[e].extend(branch_above[e])
    unique_branch_lengths = []
    for e in range(len(epochs)):
        unique_branch_lengths.append(np.unique(all_branch_lengths[e]))
    return pl, bal, tbw, roots, unique_branch_lengths

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
        asymmetric=False,
        output_file=None
    ):

    if output_file is not None:
        sys.stdout = open(output_file, "w")
    
    demes = pd.read_csv(demes_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes, samples, asymmetric)

    if trees_dir_path[-1] != "/":
        trees_dir_path += "/"
    
    trees = []
    for ts in glob(trees_dir_path+"*"):
        tree = tskit.load(ts).simplify()
        if time_bins is not None:
            tree = nx_bin_ts(tree, time_bins)
        trees.append(tree.first())

    pl, bal, tbw, r, ubl = _deconstruct_trees(trees=trees, epochs=world_map.epochs, time_bins=time_bins)  # needed to use numba

    bounds = [(-10, 5) for rate in world_map.existing_connection_types]

    res = shgo(
        calc_log_migration_rate_log_likelihood,
        bounds=bounds,
        n=100,
        iters=max(5, len(bounds)),
        sampling_method="sobol",
        args=(world_map, pl, bal, tbw, r, ubl, output_file)
    )

    if output_file is not None:
        sys.stdout.close()

    return res

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

def _get_messages(
        parents,
        branch_above,
        branch_lengths,
        backwards,
        forwards,
        sample_locations_array,
        sample_ids
    ):

    
    num_demes = len(sample_locations_array[0])

    messages = {}
    for child in range(len(parents)):
        parent = int(parents[child])
        bl = branch_above[:, child]
        included_epochs = np.where(bl > 0)[0]
        if (len(included_epochs) > 0):
            transition_prob = np.eye(num_demes)
            for epoch in included_epochs:
                bl_index = np.where(branch_lengths[epoch]==bl[epoch])[0][0]
                transition_prob = np.dot(transition_prob, backwards[epoch][bl_index])
            if child in sample_ids:
                loc_vec = sample_locations_array[np.where(sample_ids==child)[0][0]]
            else:
                incoming_keys_child = [key for key in messages.keys() if key[1] == child]
                incoming_messages = [messages[income] for income in incoming_keys_child]
                for i in range(1,len(incoming_messages)):
                    incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                    incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
                loc_vec = incoming_messages[0]
            messages[(child, parent)] = np.dot(loc_vec, transition_prob)
    for child in range(len(parents)-1, -1, -1):
        if child not in sample_ids:
            parent = int(parents[child])
            bl = branch_above[:, child]
            included_epochs = np.where(bl > 0)[0]
            if (len(included_epochs) > 0):
                transition_prob = np.eye(num_demes)
                for epoch in included_epochs:
                    bl_index = np.where(branch_lengths[epoch]==bl[epoch])[0][0]
                    transition_prob = np.dot(transition_prob, forwards[epoch][bl_index])
                incoming_keys_parent = [key for key in messages.keys() if key[1] == parent]
                incoming_messages = [messages[income] for income in incoming_keys_parent if income[0] != child]
                for i in range(1,len(incoming_messages)):
                    incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                    incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
                loc_vec = incoming_messages[0]
                messages[(parent, child)] = np.dot(loc_vec, transition_prob)
    return messages
        

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

    parents, branch_above = _deconstruct_tree(tree, world_map.epochs)
    
    all_branch_lengths = [[] for e in world_map.epochs]
    for e in range(len(world_map.epochs)):
        all_branch_lengths[e].extend(branch_above[e])
    unique_branch_lengths = []
    for e in range(len(world_map.epochs)):
        unique_branch_lengths.append(np.unique(all_branch_lengths[e]))

    transition_matrices_backwards = world_map.build_transition_matrices(migration_rates=migration_rates)
    precomputed_transitions_backwards = precalculate_transitions(
        branch_lengths=unique_branch_lengths,
        transition_matrices=transition_matrices_backwards
    )
    transition_matrices_forwards = transition_matrices_backwards.copy()
    if world_map.asymmetric == True:
        for e in range(len(transition_matrices_forwards)):
            transition_matrices_forwards[e] = transition_matrices_forwards[e].T
        precomputed_transitions_forwards = precalculate_transitions(
            branch_lengths=unique_branch_lengths,
            transition_matrices=transition_matrices_forwards
        )
    else:
        precomputed_transitions_forwards = precomputed_transitions_backwards.copy()

    sample_locations_array, sample_ids = world_map._build_sample_locations_array()

    messages = _get_messages(
        parents,
        branch_above,
        unique_branch_lengths,
        precomputed_transitions_backwards,
        precomputed_transitions_forwards,
        sample_locations_array,
        sample_ids
    )

    positions = {}
    for element,node_combo in enumerate(pc_combos):
        if node_combo[0] == node_combo[1]:
            if node_combo[0] in sample_ids:
                node_pos = sample_locations_array[np.where(sample_ids==node_combo[0])[0][0]]
            else:
                incoming_keys = [key for key in messages.keys() if key[1] == node_combo[0]]
                incoming_messages = [messages[income] for income in incoming_keys]
                for i in range(1,len(incoming_messages)):
                    incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                    incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0])
                node_pos = incoming_messages[0]
        else:
            if node_combo[0] in sample_ids:
                child_pos = sample_locations_array[np.where(sample_ids==node_combo[0])[0][0]]
            else:
                incoming_keys_child = [key for key in messages.keys() if key[1] == node_combo[0]]
                incoming_messages_child = [messages[income] for income in incoming_keys_child if income[0] != node_combo[1]]
                if len(incoming_keys_child) > 0:
                    for i in range(1,len(incoming_messages_child)):
                        incoming_messages_child[0] = np.multiply(incoming_messages_child[0], incoming_messages_child[i])
                        incoming_messages_child[0] = incoming_messages_child[0] / np.sum(incoming_messages_child[0])
                    child_pos = incoming_messages_child[0]
                else:
                    child_pos = np.ones((1,len(world_map.demes)))
            if node_combo[1] in sample_ids:
                parent_pos = sample_locations_array[np.where(sample_ids==node_combo[1])[0][0]]
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

            branch_length_to_child = int(times[element] - tree.time(node_combo[0]))
            bl_child = branch_above[:, node_combo[0]].copy()
            bl_parent = bl_child.copy()
            whats_left = branch_length_to_child
            for e in world_map.epochs:
                current = bl_parent[e].copy()
                if current >= whats_left:
                    bl_parent[e] -= whats_left
                else:
                    bl_parent[e] = 0
                whats_left -= current
            bl_child -= bl_parent

            included_epochs = np.where(bl_child > 0)[0]
            transition_prob_backwards = np.eye(len(world_map.demes))
            for e in included_epochs:
                transition_prob_backwards = np.dot(transition_prob_backwards, np.linalg.matrix_power(linalg.expm(transition_matrices_backwards[e]), bl_child[e]))

            included_epochs = np.where(bl_parent > 0)[0]
            transition_prob_forwards = np.eye(len(world_map.demes))
            for e in included_epochs:
                transition_prob_forwards = np.dot(transition_prob_forwards, np.linalg.matrix_power(linalg.expm(transition_matrices_forwards[e]), bl_parent[e]))
            
            outgoing_child_message = np.dot(child_pos, transition_prob_backwards)
            outgoing_parent_message = np.dot(parent_pos, transition_prob_forwards)
            node_pos = np.multiply(outgoing_child_message, outgoing_parent_message)
            node_pos = node_pos / np.sum(node_pos)
        positions[times[element]] = node_pos
    return positions