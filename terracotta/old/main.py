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

def logsumexp_custom(x, axis):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c), axis=axis))

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
        self.sample_location_vectors = self._build_sample_location_vectors()
        deme_types = self.demes["type"].unique()
        connection_types = []
        for i,type0 in enumerate(deme_types):
            for type1 in deme_types[i:]:
                if type1 > type0:
                    connection_types.append((type0, type1))
                else:
                    connection_types.append((type1, type0))
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
                        ct = connection_types.index((row["type"], neighbor_type))
                    else:
                        ct = connection_types.index((neighbor_type, row["type"]))
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
                plt.plot([deme_0[0], deme_1[0]], [deme_0[1], deme_1[1]], color=colorFader("blue", "green", mr), linewidth=5)
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
            plt.scatter(counts["xcoord"], counts["ycoord"], color="orange", zorder=3) #s=counts["count"]*50, zorder=3)
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
        plt.scatter(self.demes["xcoord"], self.demes["ycoord"], zorder=2, c=location_vector[0][self.demes.index], vmin=0)
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
            plt.title(title)
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
        for index,connection in self.connections.iterrows():
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

def convert_tree_to_tuple_list(tree):
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
    #root_log_likes = [logsumexp_custom(log_messages[r], axis=0) for r in roots if r not in sample_ids]
    tree_likelihood = sum(root_log_likes)
    return tree_likelihood, root_log_likes

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
                outgoing_log_message = np.array([logsumexp_custom(log_mats[bl] + summed_log_messages, axis=1)])
            else:
                outgoing_log_message = summed_log_messages
            log_messages[node] = outgoing_log_message

    roots = tree.roots
    root_log_likes = [logsumexp_custom(log_messages[r], axis=0) for r in roots if r not in sample_location_vectors]
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
    exponentiated = linalg.expm(transition_matrix)
    exponentiated[exponentiated < 0] = 0

    previous_length = None
    previous_mat = None
    precomputed_transitions = np.zeros((len(branch_lengths), len(world_map.demes), len(world_map.demes)), dtype="float64")
    precomputed_log = np.zeros((len(branch_lengths), len(world_map.demes), len(world_map.demes)), dtype="float64")
    counter = 0
    for bl in branch_lengths:
        if previous_length != None:
            diff = bl - previous_length
            where_next = np.dot(previous_mat, np.linalg.matrix_power(exponentiated, diff))
        else:
            where_next = np.linalg.matrix_power(exponentiated, bl)
        precomputed_transitions[counter] = where_next
        precomputed_transitions[counter][precomputed_transitions[counter] <= 0] = 1e-99
        precomputed_log[counter] = np.log(precomputed_transitions[counter]).T
        previous_length = bl
        previous_mat = where_next
        counter += 1

    sample_location_vectors = np.zeros((len(world_map.sample_location_vectors), len(world_map.demes)), dtype="float64")
    sample_ids = np.zeros(len(world_map.sample_location_vectors), dtype="int64")
    counter = 0
    for sample in world_map.sample_location_vectors:
        sample_location_vectors[counter] = world_map.sample_location_vectors[sample]
        sample_ids[counter] = sample
        counter += 1
    
    like, like_list = _parallel_process_trees(
        children=children,
        branch_above=branch_above,
        roots=roots,
        sample_ids=sample_ids,
        sample_location_vectors=sample_location_vectors,
        branch_lengths=branch_lengths,
        precomputed_transitions=precomputed_transitions,
        precomputed_log=precomputed_log
    )

    print(migration_rates, abs(like))

    return abs(like)

def calc_migration_rate_log_likelihood_orig(world_map, trees, migration_rates, branch_lengths):
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
    exponentiated = linalg.expm(transition_matrix)
    exponentiated[exponentiated < 0] = 0

    start = time.time()
    previous_length = None
    previous_mat = None
    precomputed_transitions = {}
    for bl in branch_lengths:
        if previous_length != None:
            diff = bl - previous_length
            where_next = np.dot(previous_mat, np.linalg.matrix_power(exponentiated, diff))
        else:
            where_next = np.linalg.matrix_power(exponentiated, bl)
        precomputed_transitions[bl] = where_next
        precomputed_transitions[bl][precomputed_transitions[bl] <= 0] = 1e-99
        previous_length = bl
        previous_mat = where_next
    print(time.time() - start)

    #print("Calculating loglikelihoods per tree...")
    log_likelihoods = []
    for tree in trees:
        start = time.time()
        log_likelihoods.append(_calc_tree_log_likelihood(tree, world_map.sample_location_vectors, precomputed_transitions=precomputed_transitions)[0])
        print("-", time.time() - start)
    mr_log_like = sum(log_likelihoods)

    return mr_log_like, log_likelihoods




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

    #print("Precalculating transitions...")
    start = time.time()
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
    print(time.time() - start)

    #print("Calculating loglikelihoods per tree...")
    log_likelihoods = []
    for tree in trees:
        log_likelihoods.append(_calc_tree_log_likelihood(tree, world_map.sample_location_vectors, precomputed_transitions=precomputed_transitions)[0])
    mr_log_like = sum(log_likelihoods)

    return mr_log_like, log_likelihoods

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