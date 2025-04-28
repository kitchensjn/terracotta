import numpy as np
from scipy import linalg
from scipy.special import logsumexp
import tskit
import networkx as nx
import msprime
from os import mkdir
import random
import matplotlib.pyplot as plt
import pandas as pd


def calc_tree_log_likelihood(tree, sample_locs, transition_matrix):
    """Calculates the log_likelihood of the tree using Felsenstein's Pruning Algorithm.

    NOTE: Assumes that samples are always tips on the tree.

    Parameters
    ----------
    tree : tskit.Tree
        This is a tree taken from the tskit.TreeSequence.
    sample_locs : dict
        Contains all of the location vectors for the samples
    transition_matrix : np.matrix
        Instantaneous migration rate matrix between states

    Returns
    -------
    tree_likelihood : float
        likelihood of the tree (product of the root likelihoods)
    root_log_likes : list
        List of root likelihoods (sum of the root locations vector)
    """

    log_messages = {}
    for l in sample_locs:
        log_messages[l] = np.log(np.matmul(sample_locs[l], linalg.expm(transition_matrix*tree.branch_length(l))))

    #node_times = [0]
    #new_max_like = [1]
    #max_like = 0
    nodes = []
    like = []

    for node in tree.nodes(order="timeasc"):
        children = tree.children(node)
        if len(children) > 0:
            incoming_log_messages = []
            for child in children:
                incoming_log_messages.append(log_messages[child])
            summed_log_messages = np.sum(incoming_log_messages, axis=0)
            
            #print(node, np.exp(logsumexp(summed_log_messages)) / np.exp(child_like_prod))

            nodes.append(node)
            like.append(logsumexp(summed_log_messages))

            #if node_like <= max_like:
            #    node_times.append(tree.time(node))
            #    new_max_like.append(node_like)
            #    max_like = node_like
            
            outgoing_log_message = np.array([logsumexp(np.log(linalg.expm(transition_matrix*tree.branch_length(child))).T + summed_log_messages, axis=1)])
            log_messages[node] = outgoing_log_message
        else:
            nodes.append(node)
            like.append(logsumexp(log_messages[node]))
    
    roots = tree.roots
    root_log_likes = [logsumexp(log_messages[r]) for r in roots]
    tree_likelihood = sum(root_log_likes)

    return tree_likelihood, root_log_likes, nodes, like#, node_times, new_max_like

def create_grid_world_map_file(side_length, number_of_tile_types=1, output_path="world_map.tsv"):
    """Creates the world map input file that pairs with grid.slim or msprime of specified size

    Parameters
    ----------
    side_length : int
        Number of populations along one side of the square grid metapopulation
    number_of_tile_types : int
        Set the number of tile types that will be randomly assigned
    path : str
        Path to output file. (default="world_map.tsv")
    """

    with open(output_path, "w") as outfile:
        outfile.write("id\txcoord\tycoord\ttype\tneighbours\n")
        for y in range(side_length):
            for x in range(side_length):
                neighbors = []
                if (y > 0):
                    neighbors.append(str(x+(y-1)*side_length))
                if (x > 0):
                    neighbors.append(str((x-1)+y*side_length))
                if (x < side_length-1):
                    neighbors.append(str((x+1)+y*side_length))
                if (y < side_length-1):
                    neighbors.append(str(x+(y+1)*side_length))
                outfile.write(f"{x+y*side_length}\t{x}\t{y}\t{random.randint(0,number_of_tile_types-1)}\t{",".join(neighbors)}\n")

def create_samples_file_from_tree_sequence(input_trees_path=None, ts=None, output_samples_path="samples.tsv", num_samples=-1, output_trees_path="samples.trees", random_seed=None):
    """Creates the samples input file from a tree sequence.

    Optionally, subsets tree sequence to a specified number of random samples.

    Parameters
    ----------
    input_trees_path : str
    ts : tskit:TreeSequence
    output_samples_path : str
    num_samples : int
    output_trees_path : str
    random_seed : int
    """

    if input_trees_path is not None:
        ts = tskit.load(input_trees_path)
    elif ts is not None:
        pass
    else:
        raise RuntimeError("Must provide either an `input_trees_path` or `ts` as input.")

    if random_seed != None:
        np.seed(random_seed)

    if num_samples > 1:
        s = np.random.choice(ts.num_samples, num_samples, replace=False)
        ts = ts.simplify(samples=s)
        ts.dump(output_trees_path)

    with open(output_samples_path, "w") as outfile:
        outfile.write("sample\tstate\n")
        for sample in ts.samples():
            outfile.write(f'{sample}\t{ts.node(sample).population}\n')

def build_state_space_from_map(df):
    """
    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    graph : nx.Graph
    """

    graph = nx.Graph()
    attr = {}
    for index, row in df.iterrows():
        attr[int(row["id"])] = {"x":row["xcoord"], "y":row["ycoord"]}
        neighbors = row["neighbours"].split(",")
        for n in neighbors:
            graph.add_edge(int(row["id"]), int(n))
    nx.set_node_attributes(graph, attr)
    subgraphs = list(nx.connected_components(graph))
    if len(subgraphs) > 1:
        print("Map has disconnected components - returning only the largest component.")
        graph = graph.subgraph(max(subgraphs, key=len))
    return graph

def build_transition_matrix(state_space, migration_rate):
    """
    Parameters
    ----------
    state_space : networkx.Graph
    migration_rate : float
        Migration rate between neighboring states

    Returns
    -------
    transition_matrix : np.matrix
    """

    transition_matrix = nx.adjacency_matrix(state_space).todense()*migration_rate
    diag = -np.sum(transition_matrix, axis=1)
    np.fill_diagonal(transition_matrix, diag)
    return transition_matrix

def convert_sample_locations(samples, state_space):
    """
    """

    state_indices = list(state_space.nodes())
    sample_loc_probs = {}
    for i,sample in enumerate(samples["state"]):
        loc_prob = np.zeros((1,len(state_indices)))
        loc_prob[0,state_indices.index(sample)] = 1
        sample_loc_probs[i] = loc_prob
    return sample_loc_probs

def simulate_independent_trees_from_grid_demography(grid_side_length, number_of_samples, ploidy, number_of_trees, pop_size, migration_rate, allow_multiple_samples_per_pop):
    """
    """

    if (not allow_multiple_samples_per_pop) and (number_of_samples > grid_side_length**2):
        print("Can't")
        number_of_samples = grid_side_length**2
    demography = msprime.Demography()
    side_length = grid_side_length
    for y in range(side_length):
        for x in range(side_length):
            demography.add_population(name="Pop_"+str(x+y*side_length), initial_size=pop_size)
    for y in range(side_length):
        for x in range(side_length):
            if (y > 0):
                demography.set_migration_rate("Pop_"+str(x+y*side_length), "Pop_"+str(x+(y-1)*side_length), migration_rate)
            if (x > 0):
                demography.set_migration_rate("Pop_"+str(x+y*side_length), "Pop_"+str((x-1)+y*side_length), migration_rate)
            if (x < side_length-1):
                demography.set_migration_rate("Pop_"+str(x+y*side_length), "Pop_"+str((x+1)+y*side_length), migration_rate)
            if (y < side_length-1):
                demography.set_migration_rate("Pop_"+str(x+y*side_length), "Pop_"+str(x+(y+1)*side_length), migration_rate)
    random_samples = np.random.choice(side_length**2, number_of_samples, replace=allow_multiple_samples_per_pop)
    samples = {}
    for s in random_samples:
        samples["Pop_"+str(s)] = samples.get("Pop_"+str(s), 0) + 1
    for i in range(number_of_trees):
        ts = msprime.sim_ancestry(
            samples=samples,
            ploidy=ploidy,
            demography=demography
        )
        yield ts
    
def create_dataset(grid_side_length, number_of_samples, ploidy, number_of_trees, output_path, pop_size, migration_rate, allow_multiple_samples_per_pop):
    """Simulates a dateset using msprime with a grid demography and creates all of the necessary input for terracotta.

    Parameters
    ----------
    grid_side_length : int
        Number of populations along one side of the square grid metapopulation
    number_of_samples : int
        Number of diploid samples
    ploidy : int
        Number of chromosomes
    number_of_trees : int
        Number of independent trees to generate
    pop_size : int
        Population size of individual populations within the metapopulation
    migration_rate : float
        Migration rate between neighboring populations
    output_path : str
        Path to the output directory that will be created and contain dataset
    """

    mkdir(output_path)
    create_grid_world_map_file(
        side_length=grid_side_length,
        output_path=f"{output_path}/world_map.tsv"
    )
    mkdir(f"{output_path}/trees")
    trees = simulate_independent_trees_from_grid_demography(
        grid_side_length=grid_side_length,
        number_of_samples=number_of_samples,
        ploidy=ploidy,
        number_of_trees=number_of_trees,
        pop_size=pop_size,
        migration_rate=migration_rate,
        allow_multiple_samples_per_pop=allow_multiple_samples_per_pop
    )
    for i,tree in enumerate(trees):
        tree.dump(f"{output_path}/trees/{i}.trees")
    create_samples_file_from_tree_sequence(
        ts=tree,
        output_samples_path=f"{output_path}/samples.tsv"
    )