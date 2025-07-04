import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy import linalg
from scipy.special import logsumexp
import matplotlib as mpl
import networkx as nx
import tskit
import time
from numba import njit, prange


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

class WorldMap:
    """Stores a map of the demes

    Attributes
    ----------
    demes : pandas.DataFrame
    samples : pandas.DataFrame
    connections : pandas.DataFrame
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

        self.demes = demes
        self.samples = samples
        self.sample_location_vectors = None
        if samples is not None:
            self.sample_location_vectors = self._build_sample_location_vectors()
        deme_types = np.sort(self.demes["type"].unique())
        #connection_types = []
        #for i,type0 in enumerate(deme_types):
        #    for type1 in deme_types[i:]:
        #        if type1 > type0:
        #            connection_types.append((type0, type1))
        #        else:
        #            connection_types.append((type1, type0))
        connections = []
        converted_neighbors = []
        for index,row in demes.iterrows():
            neighbors = str(row["neighbours"]).split(",")
            int_neighbors = []
            for neighbor in neighbors:
                neighbor = int(neighbor)
                int_neighbors.append(neighbor)
                if row["id"] < neighbor:
                    neighbor_type = self.get_deme_type(neighbor)
                    if neighbor_type > row["type"]:
                        ct = deme_types[neighbor_type]  #connection_types.index((row["type"], neighbor_type))
                    else:
                        ct = deme_types[row["type"]]  #connection_types.index((neighbor_type, row["type"]))
                    connections.append({"deme_0":row["id"], "deme_1":neighbor, "type":ct})
            converted_neighbors.append(int_neighbors)
        self.demes["neighbours"] = converted_neighbors
        self.connections = pd.DataFrame(connections)

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
    
    def get_deme_type(self, id):
        """Gets the type for specified deme

        Wrapper of pandas.DataFrame function. Assumes that deme IDs are unique.

        Parameters
        ----------
        id : int
            ID of deme

        Returns
        -------
        deme_type : int
            Type of deme
        """

        deme_type = self.demes.loc[self.demes["id"]==id,"type"].iloc[0]
        return deme_type

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

    def draw(self, figsize, show_samples=False, color_demes=False, color_connections=False, migration_rates=None, save_to=None):
        """Draws the world map

        Uses matplotlib.pyplot

        Parameters
        ----------
        color_demes : bool
            Whether to color the demes based on type
        color_connections : bool
            Whether to color the connections based on type
        """

        plt.figure(figsize=figsize)
        
        if migration_rates != None:
            mr_values = migration_rates.values()
            max_mr = max(mr_values)
            min_mr = min(mr_values)
        if color_connections:
            num_connection_types = len(self.connections["type"].unique())
            color_rnd = random.Random()
            color_rnd.seed(1)
            connection_colors = ["#"+''.join([color_rnd.choice("0123456789ABCDEF") for j in range(6)]) for i in range(num_connection_types)]
        for index,row in self.connections.iterrows():
            deme_0 = self.get_coordinates_of_deme(row["deme_0"])
            deme_1 = self.get_coordinates_of_deme(row["deme_1"])
            if migration_rates != None:
                mr = ((migration_rates[row["type"]]-min_mr)/(max_mr-min_mr))
                plt.plot([deme_0[0], deme_1[0]], [deme_0[1], deme_1[1]], linewidth=5, color=colorFader("brown", "orange", mr))
            elif color_connections:
                plt.plot([deme_0[0], deme_1[0]], [deme_0[1], deme_1[1]], color=connection_colors[row["type"]], linewidth=5)
            else:
                plt.plot([deme_0[0], deme_1[0]], [deme_0[1], deme_1[1]], color="grey")
        if color_demes:
            plt.scatter(self.demes["xcoord"], self.demes["ycoord"], c=self.demes["type"], zorder=2, s=100)
        else:
            plt.scatter(self.demes["xcoord"], self.demes["ycoord"], zorder=2, color="grey")
        if isinstance(self.samples, pd.DataFrame) and show_samples:
            counts = self.samples["deme"].value_counts().reset_index()
            counts = counts.merge(self.demes, how="left", left_on="deme", right_on="id").loc[:,["id", "xcoord", "ycoord", "count"]]
            plt.scatter(counts["xcoord"], counts["ycoord"], color="orange", s=counts["count"]*10, zorder=3)
        plt.gca().set_aspect("equal")
        plt.axis("off")
        if save_to != None:
            plt.savefig(save_to)
        plt.show()

    def draw_estimated_location(self, location_vector, figsize, show_samples=False, save_to=None, title=None):
        """Draws the world map

        Uses matplotlib.pyplot

        Parameters
        ----------
        color_demes : bool
            Whether to color the demes based on type
        color_connections : bool
            Whether to color the connections based on type
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
        plt.gca().set_aspect("equal")
        plt.axis("off")
        if title != None:
            plt.title(title, fontname="Georgia", fontsize=50)
        if save_to != None:
            plt.savefig(save_to)
        plt.show()

    def build_transition_matrix(self, migration_rates):
        """Builds the transition matrix based on the world map and migration rates

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

        transition_matrix = np.zeros((len(self.demes),len(self.demes)))
        for _,connection in self.connections.iterrows():
            i_0 = self.demes.loc[self.demes["id"]==connection["deme_0"]].index[0]
            i_1 = self.demes.loc[self.demes["id"]==connection["deme_1"]].index[0]
            rate = migration_rates[connection["type"]]
            #rate = migration_rates.get(connection["type"], -1)
            #if rate == -1:
            #    raise RuntimeError(f"Rate for connection of type '{connection["type"]}' is not provided. Please specify and try again.")
            transition_matrix[i_0, i_1] = rate
            transition_matrix[i_1, i_0] = rate
        diag = -np.sum(transition_matrix, axis=1)
        np.fill_diagonal(transition_matrix, diag)
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

def convert_tree_to_tuple_list(tree):
    """Breaks a tskit.Tree into primitive lists of children, branch_lengths, and root IDs
    """

    num_nodes = len(tree.postorder())
    children = np.zeros((num_nodes, num_nodes), dtype="int64")
    branch_above = []
    for node in tree.nodes(order="timeasc"):
        for child in tree.children(node):
            children[node, child] = 1
        branch_above.append(tree.branch_length(node))
    return children, np.array(branch_above), np.array(tree.roots)

@njit(fastmath=True)
def _calc_tree_log_likelihood(
        child_list,
        branch_above_list,
        roots,
        sample_ids,
        sample_location_vectors,
        branch_lengths,
        precomputed_transitions,
        precomputed_log
    ):
    """
    """

    num_nodes = len(branch_above_list)
    num_demes = len(sample_location_vectors[0])

    log_messages = np.zeros((num_nodes, num_demes), dtype="float64")
    counter = 0
    for l in sample_ids:
        bl = branch_above_list[l]
        if bl > 0:
            bl_index = np.where(branch_lengths==bl)[0][0]
            log_messages[l] = np.log(np.dot(sample_location_vectors[counter], precomputed_transitions[bl_index]))
        counter += 1

    node_counter = 0
    for children in child_list:
        if sum(children) > 0:
            childs = np.where(children == 1)[0]
            incoming_log_messages = np.zeros((len(childs), num_demes), dtype="float64")
            counter = 0
            for child in childs:
                incoming_log_messages[counter] = log_messages[child]
                counter += 1
            summed_log_messages = np.sum(incoming_log_messages, axis=0)
            bl = branch_above_list[node_counter]
            if bl > 0:
                bl_index = np.where(branch_lengths==bl)[0][0]
                combined = precomputed_log[bl_index] + summed_log_messages
                c = np.max(combined)
                log_sum_exp = c + np.log(np.sum(np.exp(combined - c), axis=1))
                outgoing_log_message = log_sum_exp#.reshape(1, -1)
            else:
                outgoing_log_message = summed_log_messages
            log_messages[node_counter] = outgoing_log_message
        node_counter += 1

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
        children,
        branch_above,
        roots,
        sample_ids,
        sample_location_vectors,
        branch_lengths,
        precomputed_transitions,
        precomputed_log
    ):
    """
    """

    log_likelihoods = np.zeros(len(branch_above), dtype="float64")
    for i in prange(len(branch_above)):
        log_likelihoods[i] = _calc_tree_log_likelihood(
            child_list=children[i],
            branch_above_list=branch_above[i],
            roots=roots[i],
            sample_ids=sample_ids,
            sample_location_vectors=sample_location_vectors,
            branch_lengths=branch_lengths,
            precomputed_transitions=precomputed_transitions,
            precomputed_log=precomputed_log
        )[0]
    mr_log_like = sum(log_likelihoods)
    return mr_log_like, log_likelihoods

def precalculate_transitions(branch_lengths, transition_matrix, fast=True):
    """Calculates the transition probabilities between demes for each branch length

    Parameters
    ----------
    branch_lengths : np.ndarray
        Array of branch lengths of increasing size
    transition_matrix : np.ndarray
        Instantaneous migration rate matrix, output of WorldMap.build_transition_matrix()
    fast : bool
        Speeds up calculation but can aggregate floating point errors for longer times
    """

    num_demes = transition_matrix.shape[0]
    exponentiated = linalg.expm(transition_matrix)
    previous_length = -1
    precomputed_transitions = np.zeros((len(branch_lengths), num_demes, num_demes), dtype="float64")
    precomputed_log = np.zeros((len(branch_lengths), num_demes, num_demes), dtype="float64")
    counter = 0
    for bl in branch_lengths:
        if fast and previous_length != -1:
            diff = bl - previous_length
            where_next = np.dot(previous_mat, np.linalg.matrix_power(exponentiated, diff))
        else:
            where_next = np.linalg.matrix_power(exponentiated, bl)
        precomputed_transitions[counter] = where_next
        precomputed_transitions[counter][precomputed_transitions[counter] <= 1e-99] = 1e-99
        precomputed_log[counter] = np.log(precomputed_transitions[counter]).T
        previous_length = bl
        previous_mat = where_next
        counter += 1
    return precomputed_transitions, precomputed_log

def calc_migration_rate_log_likelihood(migration_rates, world_map, children, branch_above, roots, branch_lengths):
    """Calculates the composite log-likelihood of the specified migration rates across trees
    
    Loops through all trees and calculates the log-likelihood for each, before summing together.

    Parameters
    ----------
    world_map : terracotta.WorldMap
    trees : list
        List of tskit.Tree objects
    migration_rates : dict
        Keys are the connection type and values are the instantaneous migration
        rate along that connection
    branch_lengths : np.array

    Returns
    -------
    mr_log_like : float
        Log-likelihood of the specified migration rates
    """

    transition_matrix = world_map.build_transition_matrix(migration_rates=migration_rates)
    precomputed_transitions, precomputed_log = precalculate_transitions(
        branch_lengths=branch_lengths,
        transition_matrix=transition_matrix
    )
    sample_locations_array, sample_ids = world_map._build_sample_locations_array()
    like, like_list = _parallel_process_trees(
        children=children,
        branch_above=branch_above,
        roots=roots,
        sample_ids=sample_ids,
        sample_location_vectors=sample_locations_array,
        branch_lengths=branch_lengths,
        precomputed_transitions=precomputed_transitions,
        precomputed_log=precomputed_log
    )
    print(migration_rates, abs(like), flush=True)
    return abs(like)

def locate_nodes_in_tree(tree, world_map, migration_rates):
    transition_matrix = world_map.build_transition_matrix(migration_rates=migration_rates)

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
    
    transition_matrix = world_map.build_transition_matrix(migration_rates=migration_rates)
    
    total_number_of_edges = tree.num_edges+1
    branch_lengths = np.zeros(total_number_of_edges, dtype="int64")
    edge_counter = 0
    for node in tree.nodes(order="timeasc"):
        branch_lengths[edge_counter] = int(tree.branch_length(node))
        edge_counter += 1
    branch_lengths = np.unique(np.array(branch_lengths))

    precomputed_transitions, precomputed_log = precalculate_transitions(
        branch_lengths=branch_lengths,
        transition_matrix=transition_matrix
    )

    messages = {}
    for node in tree.nodes(order="timeasc"):
        bl = int(tree.branch_length(node))
        bl_index = np.where(branch_lengths==bl)[0][0]
        if tree.is_sample(node):
            messages[(node, tree.parent(node))] = np.matmul(world_map.sample_location_vectors[node], precomputed_transitions[bl_index])
        else:
            children = tree.children(node)
            incoming_messages = [messages[(child, node)] for child in children]
            for i in range(1,len(incoming_messages)):
                incoming_messages[0] = np.multiply(incoming_messages[0], incoming_messages[i])
                incoming_messages[0] = incoming_messages[0] / np.sum(incoming_messages[0]) 
            messages[(node, tree.parent(node))] = np.matmul(incoming_messages[0], precomputed_transitions[bl_index])
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
            bl = int(tree.branch_length(child))
            bl_index = np.where(branch_lengths==bl)[0][0]
            messages[(node, child)] = np.matmul(incoming_messages[0], precomputed_transitions[bl_index])
    return messages

def _ancestors(tree, u):
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

def track_lineage_over_time(sample, times, tree, world_map, migration_rates):
    """
    """

    ancestors = [sample] + list(_ancestors(tree=tree, u=sample))
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
    transition_matrix = world_map.build_transition_matrix(migration_rates=migration_rates)
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
                for i in range(1,len(incoming_messages_child)):
                    incoming_messages_child[0] = np.multiply(incoming_messages_child[0], incoming_messages_child[i])
                    incoming_messages_child[0] = incoming_messages_child[0] / np.sum(incoming_messages_child[0])
                child_pos = incoming_messages_child[0]
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
                for i in range(1,len(incoming_messages_parent)):
                    incoming_messages_parent[0] = np.multiply(incoming_messages_parent[0], incoming_messages_parent[i])
                    incoming_messages_parent[0] = incoming_messages_parent[0] / np.sum(incoming_messages_parent[0])
                parent_pos = incoming_messages_parent[0]
            branch_length_to_parent = int(tree.time(node_combo[1]) - times[element])
            outgoing_parent_message = np.matmul(parent_pos, np.linalg.matrix_power(linalg.expm(transition_matrix), branch_length_to_parent))
            node_pos = np.multiply(outgoing_child_message, outgoing_parent_message)
            node_pos = node_pos / np.sum(node_pos)
        positions[times[element]] = node_pos
    return positions

def ts_to_nx(ts):
    """Covert tskit.TreeSequence to networkx graph
    
    Parameters
    ----------
    ts : tskit.TreeSequence
    
    Returns
    -------
    networkx.DiGraph
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
    bins : list

    Returns
    -------
    ts_out : tskit.TreeSequence
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






def calc_tree_log_likelihood_old(tree, sample_location_vectors, transition_matrix=None, precomputed_transitions=None):
    """Calculates the log_likelihood of the tree using Felsenstein's Pruning Algorithm.

    NOTE: Assumes that samples are always tips on the tree.
    NOTE: Ignores samples that are completely detached from the tree(s).
    NOTE: Parent of sample cannot have the same time as sample.

    Parameters
    ----------
    tree : tskit.Tree
        This is a tree taken from the tskit.TreeSequence.
    sample_location_vectors : dict
        Contains all of the location vectors for the samples
    transition_matrix : np.matrix
        Instantaneous migration rate matrix between demes

    Returns
    -------
    tree_likelihood : float
        likelihood of the tree (product of the root likelihoods)
    root_log_likes : list
        List of root likelihoods (sum of the root locations vector)
    """

    if precomputed_transitions == None:
        if transition_matrix == None:
            raise RuntimeError("Must provide either a transition matrix or precomputed transitions.")
        else:
            precomputed_transitions = {}
            for node in tree.nodes(order="timeasc"):
                bl = tree.branch_length(node)
                if bl not in precomputed_transitions:
                    where_next = linalg.expm(transition_matrix*bl)
                    if np.any(where_next <= 0):
                        where_next[where_next <= 0] = 1e-99
                    precomputed_transitions[bl] = where_next
   
    num_nodes = len(tree.postorder())
    num_demes = len(sample_location_vectors[0][0])

    log_messages = np.zeros((num_nodes, num_demes), dtype="float64")
    for l in sample_location_vectors:
        bl = tree.branch_length(l)
        if bl > 0:
            log_messages[l] = np.log(np.dot(sample_location_vectors[l], precomputed_transitions[bl]))

    log_mats = {}
    for trans_mat in precomputed_transitions:
        log_mats[trans_mat] = np.log(precomputed_transitions[trans_mat]).T

    for node in tree.nodes(order="timeasc"):
        children = tree.children(node)
        if len(children) > 0:
            incoming_log_messages = np.zeros((len(children), num_demes), dtype="float64")
            counter = 0
            for child in children:
                incoming_log_messages[counter] = log_messages[child]
                counter += 1
            summed_log_messages = np.sum(incoming_log_messages, axis=0)
            bl = tree.branch_length(node)
            if bl > 0:
                outgoing_log_message = np.array([logsumexp(log_mats[bl] + summed_log_messages, axis=1)])
            else:
                outgoing_log_message = summed_log_messages
            log_messages[node] = outgoing_log_message

    roots = tree.roots
    root_log_likes = [logsumexp(log_messages[r], axis=0) for r in roots if r not in sample_location_vectors]
    tree_likelihood = sum(root_log_likes)
    return tree_likelihood, root_log_likes

def calc_migration_rate_log_likelihood_old(world_map, trees, migration_rates):
    """Calculates the composite log-likelihood of the specified migration rates across trees
    
    Loops through all trees and calculates the log-likelihood for each, before summing together.

    Parameters
    ----------
    world_map : terracotta.WorldMap
    trees : list
        List of tskit.Tree objects
    migration_rates : dict
        Keys are the connection type and values are the instantaneous migration
        rate along that connection

    Returns
    -------
    mr_log_like : float
        Log-likelihood of the specified migration rates
    """

    transition_matrix = world_map.build_transition_matrix(migration_rates=migration_rates)

    exponentiated = linalg.expm(transition_matrix)
    exponentiated[exponentiated < 0] = 0
    precomputed_transitions = {}
    for tree in trees:
        for node in tree.nodes(order="timeasc"):
            bl = tree.branch_length(node)
            if bl not in precomputed_transitions:
                where_next = np.linalg.matrix_power(exponentiated, int(bl)) #forces branch lengths to be integer. Could be an issue for bl<1
                where_next[where_next <= 0] = 1e-99
                precomputed_transitions[bl] = where_next

    log_likelihoods = []
    for tree in trees:
        log_likelihoods.append(calc_tree_log_likelihood_old(tree, world_map.sample_location_vectors, precomputed_transitions=precomputed_transitions)[0])
    mr_log_like = sum(log_likelihoods)

    return mr_log_like, log_likelihoods