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
from scipy.optimize import shgo, basinhopping, minimize
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from glob import glob
import sys
import time
import shapely
import shapely.plotting as plotting
from shapely.geometry import Polygon



def _format_parameter_string(s):
    """Formats the parameter string to match expected time:value format

    Parameters
    ----------
    s : string
        Input parameter string

    Returns
    -------
    formatted : string
        Formatted parameter string
    """

    if ":" in s:
        split_string = s.replace(",", ":").split(":")
        epochs = np.array(split_string[::2]).astype(float)
        parameters = np.array(split_string[1::2])
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


class WorldMap:
    """Stores a map of the demes

    Attributes
    ----------
    demes : pandas.DataFrame
    connections : pandas.DataFrame
    samples : pandas.DataFrame
    epochs : numpy.ndarray
    """

    def __init__(self, demes, connections, samples=None):
        """Initializes the WorldMap object

        Parameters
        ----------
        demes : pandas.DataFrame
            From demes.tsv file
        connections : pandas.DataFrame
            From connections.tsv file
        samples : pandas.DataFrame
            From samples.tsv file (optional)
        """

        self.parameters = ["coefficient", "alpha"]

        self.demes = demes.copy()  
        self.demes["suitability"] = self.demes["suitability"].astype(str)
        formatted_suitability_strings = []
        for suitability in self.demes["suitability"]:
            formatted_suitability_strings.append(_format_parameter_string(suitability))
        self.demes["suitability"] = formatted_suitability_strings

        self.connections = connections.copy()
        self.connections["migration_modifier"] = self.connections["migration_modifier"].astype(str)
        formatted_modifier_strings = []
        for modifier in self.connections["migration_modifier"]:
            formatted_modifier_strings.append(_format_parameter_string(modifier))
        self.connections["migration_modifier"] = formatted_modifier_strings
    
        epochs = [0]
        for suitability_string in self.demes["suitability"]:
            epoch_formatter = suitability_string.replace(",", ":").split(":")
            for epoch_assignment in epoch_formatter[::2]:
                epoch_assignment = float(epoch_assignment)
                if epoch_assignment not in epochs:
                    epochs.append(epoch_assignment)
        migration_modifier_variables = []
        for modifier_string in self.connections["migration_modifier"]:
            epoch_formatter = modifier_string.replace(",", ":").split(":")
            for t in range(0, len(epoch_formatter), 2):
                epoch_assignment = float(epoch_formatter[t])
                if epoch_assignment not in epochs:
                    epochs.append(epoch_assignment)
                modifier = epoch_formatter[t+1]
                try:
                    modifier = float(modifier)
                except:
                    if modifier not in self.parameters:
                        self.parameters.append(modifier)
        self.epochs = np.sort(epochs)

        for i,epoch in enumerate(self.epochs):
            all_suitabilities = self.get_all_deme_suitabilities_at_time(epoch)
            self.demes[f"suitability_{epoch}"] = all_suitabilities
            all_migration_modifiers = self.get_all_connection_migration_modifiers_at_time(epoch)
            self.connections[f"migration_modifier_{epoch}"] = all_migration_modifiers

        self.samples = None
        if samples is not None:
            self.samples = samples.copy()
    
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

        return pd.Series([self.get_deme_suitability_at_time(id, time) for id in self.demes["id"]])

    def get_connection_migration_modifier_at_time(self, id, time):
        connection_parameter = self.connections.loc[self.connections["id"]==id,"migration_modifier"].iloc[0]
        epochs = connection_parameter.split(",")
        for t in range(len(epochs)-1):
            details = epochs[t].split(":")
            start = float(details[0])
            end = float(epochs[t+1].split(":")[0])
            if (time >= start) and (time < end):
                return details[1]
        return epochs[-1].split(":")[1]

    def get_all_connection_migration_modifiers_at_time(self, time):
        return pd.Series([self.get_connection_migration_modifier_at_time(id, time) for id in self.connections["id"]])

    def build_transition_matrices(self, parameters):
        """Builds the transition matrix based on the world map and migration rates

        Row is the starting deme, column is the next deme backwards in time.

        Parameters
        ----------
        parameters : np.array or list
        
        Returns
        -------
        transition_matrix : np.array
        """
        
        if "coefficient" in self.parameters:
            m = parameters[self.parameters.index("coefficient")]
        else:
            m = 1
        
        if "alpha" in self.parameters:
            a = parameters[self.parameters.index("alpha")]
        else:
            a = 1

        transition_matrix = np.zeros((len(self.epochs), len(self.demes),len(self.demes)))
        for e in range(len(self.epochs)):
            for _,connection in self.connections.iterrows():
                i_0 = self.demes.loc[self.demes["id"]==connection["deme_0"]].index[0]
                i_1 = self.demes.loc[self.demes["id"]==connection["deme_1"]].index[0]
                suit = (self.demes[f"suitability_{self.epochs[e]}"][i_1] / self.demes[f"suitability_{self.epochs[e]}"][i_0])**a
                mod = connection[f"migration_modifier_{self.epochs[e]}"]
                if mod in self.parameters:
                    mod = parameters[self.parameters.index(mod)]
                else:
                    mod = float(mod)
                transition_matrix[e, i_0, i_1] = (m * suit) * mod
            diag = -np.sum(transition_matrix[e], axis=1)
            np.fill_diagonal(transition_matrix[e], diag)
        return transition_matrix

    def build_sample_locations_array(self):
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
        num_demes = transition_matrices[e].shape[0]
        transitions = np.zeros((len(branch_lengths[e]), num_demes, num_demes), dtype="float64")
        if fast:
            exponentiated = linalg.expm(transition_matrices[e])
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
                transitions[i] = linalg.expm(transition_matrices[e]*branch_lengths[e][i])
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
                    branch_above[e, node] = math.ceil(epochs[e+1] - epochs[e])
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

    Returns
    -------
    pl : list
        Arrays containing ID of parent for each node, one array per tree
    bal : list
        Arrays containing branch above length for each node, one array per tree
    tbw : list
        Arrays containaing time bin widths for each node, one array per tree
    iat : list

    unique_branch_lengths : list
        
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

def _calc_current_pos(id, messages, parents):
    """Calculates current node position as product of child messages

    Parameters
    ----------
    id : int
        ID of node
    messages : np.array
        Messages being passed in tree
    parents : np.array
        Parent IDs for each node
    
    Returns
    -------
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    """

    return np.prod(messages[np.where(parents==id)[0]], axis=0)

def _calc_branch_message(id, current_pos, branch_above, transition_matrices):
    """Calculates the message to be passed along a branch above specified node

    Parameters
    ----------
    id : int
        ID of node
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    branch_above : np.array
        Branch lengths above each node split across epochs
    transition_matrices : np.array
        Rate matrices for each epoch

    Returns
    -------
    message : np.array
        Probability distribution for location of lineage given subtree below
    """

    bl = branch_above[:,id]
    included_epochs = np.where(bl > 0)[0]
    P = np.eye(len(current_pos))
    for epoch in included_epochs:
        P = np.dot(P, linalg.expm(transition_matrices[epoch]*bl[epoch]))
    message = np.dot(current_pos, P)
    return message

def likelihood_of_tree(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        transition_matrices
    ):
    """

    Parameters
    ----------
    parents : np.array
        Parent IDs for each node
    branch_above : np.array
        Branch lengths above each node split across epochs
    ids_asc_time : np.array
        Nodes IDs in time ascending order
    sample_locations_array : np.array
        Array
    sample_ids : np.array
        Defines order of sample node IDs for sample_locations_array
    transition_matrices : np.array
        Rate matrices for each epoch

    Returns
    -------
    loglikelihood : float
        Log-likelihood of tree
    """

    num_demes = len(sample_locations_array[0])
    messages = np.zeros((len(parents), num_demes), dtype="float64")
    loglikelihood = 0
    for id in ids_asc_time: 
        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
        else:
            current_pos = _calc_current_pos(
                id,
                messages,
                parents
            )
        parent = parents[id]
        if parent != -1:
            messages[id] = _calc_branch_message(
                id,
                current_pos,
                branch_above,
                transition_matrices
            )
        else:   # collect roots here
            loglikelihood += np.log(np.sum(current_pos))
    return loglikelihood

def _process_trees(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        transition_matrices
    ):

    composite_likelihood = 0
    for i in range(len(branch_above)):
        like = likelihood_of_tree(
            parents=parents[i],
            branch_above=branch_above[i],
            ids_asc_time=ids_asc_time[i],
            sample_locations_array=sample_locations_array,
            sample_ids=sample_ids,
            transition_matrices=transition_matrices
        )
        composite_likelihood += like
    return abs(composite_likelihood)

def _calc_composite_likelihood_for_parameters(
        parameters,
        world_map,
        parents,
        branch_above,
        unique_branch_lengths,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        output_file=None,
        verbose=False
    ):

    for p in range(len(world_map.parameters)):
        if world_map.parameters[p] != "alpha":
            parameters[p] = np.exp(parameters[p])

    if "alpha" in world_map.parameters:
        alpha = parameters[world_map.parameters.index("alpha")]
    else:
        alpha = 1

    transition_matrices = world_map.build_transition_matrices(parameters=parameters)
    precomputed_transitions = precalculate_transitions(unique_branch_lengths, transition_matrices)

    composite_likelihood = _process_trees(
        parents=parents,
        branch_above=branch_above,
        ids_asc_time=ids_asc_time,
        sample_locations_array=sample_locations_array,
        sample_ids=sample_ids,
        transition_matrices=transition_matrices
    )
    if output_file is not None:
        with open(output_file, "a") as outfile:
            outfile.write(f"{parameters}\t{-composite_likelihood}\n")
    if verbose:
        print(parameters, -composite_likelihood, flush=True)
    return composite_likelihood

def run(
        demes_path,
        connections_path,
        samples_path,
        trees_dir_path,
        chop_time=None,
        time_bins=None,
        output_file=None,
        verbose=False
    ):
    """
    Parameters
    ----------

    Returns
    -------
    """
    
    if output_file is not None:
        with open(output_file, "w") as outfile:
            outfile.write("parameters\tloglikelihood\n")

    demes = pd.read_csv(demes_path, sep="\t")
    connections = pd.read_csv(connections_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes, connections, samples)

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

    sample_locations_array, sample_ids = world_map.build_sample_locations_array()
    parents, branch_above, time_bin_widths, ids_asc_time, unique_branch_lengths = _deconstruct_trees(trees=trees, epochs=world_map.epochs, time_bins=time_bins)

    res = minimize(
        fun=_calc_composite_likelihood_for_parameters,
        x0=np.array([np.log(0.1), 0.5, np.log(1)]),
        bounds=[(-10, 10), (0, 1), (-10, 10)],
        args=(
            world_map,
            parents,
            branch_above,
            unique_branch_lengths,
            ids_asc_time,
            sample_locations_array,
            sample_ids,
            output_file,
            verbose
        ),
        method="L-BFGS-B"
    )

    final = res.x.copy()
    for p in range(len(world_map.parameters)):
        if world_map.parameters[p] != "alpha":
            final[p] = np.exp(final[p])
    return final, -res.fun