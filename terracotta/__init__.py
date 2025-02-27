import numpy as np
from scipy import linalg
from scipy.special import logsumexp
import tskit
import msprime
from os import mkdir
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations_with_replacement


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

class WorldMap:
    """Stores a map of the demes

    Attributes
    ----------
    demes : pandas.DataFrame
    connections : pandas.DataFrame
    """

    def __init__(self, demes):
        """Initializes the WorldMap object

        Parameters
        ----------
        demes : pandas.DataFrame
            Five columns: "id", "xcoord", "ycoord", "type", "neighbours" (note the "u" in neighbours).
            See README.md for more details about `demes.tsv` file structure
        """

        self.demes = demes
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

    def draw(self, color_demes=False, color_connections=False, samples=None, migration_rates=None, save_to=None):
        """Draws the world map

        Uses matplotlib.pyplot

        Parameters
        ----------
        color_demes : bool
            Whether to color the demes based on type
        color_connections : bool
            Whether to color the connections based on type
        samples : pd.DataFrame
            (default=None, ignored)
        """

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
        if isinstance(samples, pd.DataFrame):
            counts = samples["deme"].value_counts().reset_index()
            counts = counts.merge(self.demes, how="left", left_on="deme", right_on="id").loc[:,["id", "xcoord", "ycoord", "count"]]
            plt.scatter(counts["xcoord"], counts["ycoord"], color="orange", s=counts["count"]*50, zorder=3)
        plt.gca().set_aspect("equal")
        plt.axis("off")
        if save_to != None:
            plt.savefig(save_to)
        plt.show()

    def build_transition_matrix(self, migration_rates):
        """Builds the transition matrix based on the world map and migration rates

        Parameters
        ----------
        migration_rates : dict
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
            rate = migration_rates.get(connection["type"], -1)
            if rate == -1:
                raise RuntimeError(f"Rate for connection of type '{connection["type"]}' is not provided. Please specify and try again.")
            transition_matrix[i_0, i_1] = rate
            transition_matrix[i_1, i_0] = rate
        diag = -np.sum(transition_matrix, axis=1)
        np.fill_diagonal(transition_matrix, diag)
        return transition_matrix

    def build_sample_location_vectors(self, sample_locations):
        """Builds sample location vectors based on the ordering of demes in the world map

        Parameters
        ----------
        sample_locations : dict

        Returns
        -------
        sample_location_vectors : dict
        """

        zeros = np.zeros((1,len(self.demes)))
        sample_location_vectors = {}
        for index,row in sample_locations.iterrows():
            sample_loc = zeros.copy()
            sample_loc[0,self.demes.loc[self.demes["id"]==row["deme"]].index[0]] = 1
            sample_location_vectors[row["id"]] = sample_loc
        return sample_location_vectors


def _calc_tree_log_likelihood(tree, sample_location_vectors, transition_matrix):
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

    log_messages = {}
    for l in sample_location_vectors:
        bl = tree.branch_length(l)
        if bl > 0:
            log_messages[l] = np.log(np.matmul(sample_location_vectors[l], linalg.expm(transition_matrix*bl)))

    for node in tree.nodes(order="timeasc"):
        children = tree.children(node)
        if len(children) > 0:
            incoming_log_messages = []
            for child in children:
                incoming_log_messages.append(log_messages[child])
            summed_log_messages = np.sum(incoming_log_messages, axis=0)
            bl = tree.branch_length(node)
            if bl > 0:
                outgoing_log_message = np.array([logsumexp(np.log(linalg.expm(transition_matrix*bl)).T + summed_log_messages, axis=1)])
            else:
                outgoing_log_message = summed_log_messages
            log_messages[node] = outgoing_log_message
    
    roots = tree.roots
    root_log_likes = [logsumexp(log_messages[r]) for r in roots if r not in sample_location_vectors]
    tree_likelihood = sum(root_log_likes)

    return tree_likelihood, root_log_likes

def calc_migration_rate_log_likelihood(world_map, trees, migration_rates, sample_location_vectors):
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
    sample_location_vectors : dict
        Contains all of the location vectors for the samples

    Returns
    -------
    mr_log_like : float
        Log-likelihood of the specified migration rates
    """

    transition_matrix = world_map.build_transition_matrix(migration_rates=migration_rates)
    log_likelihoods = []
    for tree in trees:
        log_likelihoods.append(_calc_tree_log_likelihood(tree, sample_location_vectors, transition_matrix)[0])
    mr_log_like = sum(log_likelihoods)
    return mr_log_like, log_likelihoods

def _calc_generalized_log_likelihood(arg, sample_location_vectors, transition_matrix):
    """Calculates the log_likelihood of the tree using Felsenstein's Pruning Algorithm.

    NOTE: Assumes that samples are always tips on the tree.
    NOTE: Ignores samples that are completely detached from the tree(s).
    NOTE: Parent of sample cannot have the same time as sample.
    NOTE: Currently, assumes that you know the sample location with certainty... which I think is fair.

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

    children = arg.edges_child
    parents = arg.edges_parent

    log_messages = {}
    for l in sample_location_vectors:
        sample_parents = np.unique(parents[np.where(children == l)[0]])
        for p in sample_parents:
            bl = arg.node(p).time - arg.node(l).time
            if bl > 0:
                log_messages[f"{l}_{p}"] = np.log(np.matmul(sample_location_vectors[l], linalg.expm(transition_matrix*bl)))
            else:
                raise RuntimeError(f"Edge for sample {l} to {p} has length {bl}, which is not allowed.")
    
    nodes = list(arg.nodes(order="timeasc"))
    roots = [int(r) for r in set(parents).difference(set(children))]
    arg_likelihood = _propagate_up(nodes, arg, children, parents, log_messages, transition_matrix, roots)
    return arg_likelihood

def _propagate_up(nodes, arg, children, parents, log_messages, transition_matrix, roots):           
    for i,node in enumerate(nodes):
        node_children = np.unique(children[np.where(parents == node.id)[0]])
        if len(node_children) > 0:
            incoming_log_messages = []
            for child in node_children:
                incoming_log_messages.append(log_messages[f"{child}_{node.id}"])
            summed_log_messages = np.sum(incoming_log_messages, axis=0)
            node_parents = np.unique(parents[np.where(children == node.id)[0]])
            if len(node_parents) == 0:
                log_messages[f"{node.id}_"] = summed_log_messages
            elif len(node_parents) == 1:
                bl = arg.node(node_parents[0]).time - node.time
                if bl > 0:
                    outgoing_log_message = np.array([logsumexp(np.log(linalg.expm(transition_matrix*bl)).T + summed_log_messages, axis=1)])
                    log_messages[f"{node.id}_{node_parents[0]}"] = outgoing_log_message
                else:
                    log_messages[f"{node.id}_{node_parents[0]}"] = summed_log_messages
            else:
                num_positions = len(summed_log_messages[0])
                pos_root_likelihoods = []
                for pos in range(num_positions):
                    set_pos = np.zeros((1,num_positions))
                    set_pos[0,pos] = 1
                    for p in node_parents:
                        bl = arg.node(p).time - node.time
                        if bl > 0:
                            #outgoing_log_message = np.array([logsumexp(np.log(linalg.expm(transition_matrix*bl)).T + set_pos, axis=1)]) #### THIS IS A BUG
                            outgoing_log_message = np.log(np.matmul(set_pos, linalg.expm(transition_matrix*bl)))
                            log_messages[f"{node.id}_{p}"] = outgoing_log_message
                        else:
                            raise RuntimeError(f"Edge for node {node.id} to {p} has length {bl}, which is not allowed.")
                    pos_root_likelihoods.append(_propagate_up(nodes[i+1:], arg, children, parents, log_messages, transition_matrix, roots) + summed_log_messages[0][pos])
                arg_likelihood = logsumexp(pos_root_likelihoods)
                return arg_likelihood

    root_log_likes = [logsumexp(log_messages[f"{r}_"]) for r in roots]
    arg_likelihood = sum(root_log_likes)

    return arg_likelihood

    
def calc_generalized_migration_rate_log_likelihood(world_map, trees, migration_rates, sample_location_vectors):
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
    sample_location_vectors : dict
        Contains all of the location vectors for the samples

    Returns
    -------
    mr_log_like : float
        Log-likelihood of the specified migration rates
    """

    transition_matrix = world_map.build_transition_matrix(migration_rates=migration_rates)
    log_likelihoods = []
    for arg in trees:
        log_likelihoods.append(_calc_generalized_log_likelihood(arg, sample_location_vectors, transition_matrix))
    mr_log_like = sum(log_likelihoods)
    return mr_log_like, log_likelihoods

def _create_grid_demes_file(side_length, number_of_deme_types=1, output_path="demes.tsv"):
    """Creates the world map input file that pairs with grid.slim or msprime of specified size

    Parameters
    ----------
    side_length : int
        Number of populations along one side of the square grid metapopulation
    number_of_deme_types : int
        Set the number of deme types that will be randomly assigned. (default=1)
    path : str
        Path to output file. (default="demes.tsv")
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
                outfile.write(f"{x+y*side_length}\t{x}\t{y}\t{random.randint(0,number_of_deme_types-1)}\t{",".join(neighbors)}\n")

def _set_up_msprime_demography(world_map, pop_size, migration_rates):
    """Creates the msprime.Demography object for simulating trees

    Parameters
    ----------
    world_map : terracotta.WorldMap
    pop_size : int
        The population size of each deme
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type
    """

    demography = msprime.Demography()
    for id in world_map.demes["id"]:
        demography.add_population(name="Pop_"+str(id), initial_size=pop_size)
    for index,connection in world_map.connections.iterrows():
        rate = migration_rates.get(connection["type"], -1)
        if rate == -1:
            raise RuntimeError(f"Rate for connection of type '{connection["type"]}' is not provided. Please specify and try again.")
        demography.set_migration_rate("Pop_"+str(connection["deme_0"]), "Pop_"+str(connection["deme_1"]), rate)
        demography.set_migration_rate("Pop_"+str(connection["deme_1"]), "Pop_"+str(connection["deme_0"]), rate)
    return demography

def _simulate_arg(
        world_map,
        sequence_length,
        recombination_rate,
        ploidy,
        number_of_samples,
        allow_multiple_samples_per_deme,
        pop_size,
        migration_rates
    ):
    """
    """

    demography = _set_up_msprime_demography(world_map=world_map, pop_size=pop_size, migration_rates=migration_rates)
    random_samples = np.random.choice(world_map.demes["id"], number_of_samples, replace=allow_multiple_samples_per_deme)
    samples = {}
    for s in random_samples:
        samples["Pop_"+str(s)] = samples.get("Pop_"+str(s), 0) + 1
    arg = msprime.sim_ancestry(
        samples=samples,
        ploidy=ploidy,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        demography=demography,
        record_full_arg=True
    )
    return arg

def _simulate_independent_trees(
        world_map,
        number_of_trees,
        number_of_samples,
        ploidy,
        allow_multiple_samples_per_deme,
        pop_size,
        migration_rates
    ):
    """Simulates trees under a demographic model set by the world map

    Parameters
    ----------
    world_map : terracotta.WorldMap
    number_of_trees : int
        The number of independent trees to simulate
    number_of_samples : int
    ploidy : int
        The ploidy of the samples
    allow_multiple_samples_per_deme : bool
    pop_size : int
        The population size of each deme
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type
    """

    demography = _set_up_msprime_demography(world_map=world_map, pop_size=pop_size, migration_rates=migration_rates)
    random_samples = np.random.choice(world_map.demes["id"], number_of_samples, replace=allow_multiple_samples_per_deme)
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

def _create_samples_file_from_tree_sequence(input_trees_path=None, ts=None, output_samples_path="samples.tsv", num_samples=-1, output_trees_path="samples.trees", random_seed=None):
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
        outfile.write("id\tdeme\n")
        for sample in ts.samples():
            outfile.write(f"{sample}\t{ts.node(sample).population}\n")

def create_samples_and_arg_files(
        demes_path,
        number_of_samples,
        sequence_length,
        recombination_rate,
        ploidy,
        allow_multiple_samples_per_deme,
        pop_size,
        migration_rate=None,
        migration_rates=None,
        output_directory="."
    ):
    """
    Parameters
    ----------
    demes_path : string
    number_of_samples : int
    sequence_length : int
    recombination_rate : float
    ploidy : int
        The ploidy of the samples
    allow_multiple_samples_per_deme : bool
    pop_size : int
        The population size of each deme
    migration_rate : int
        Single migration rate between neighboring demes. (default=None, ignored)
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type. (default=None, ignored)
    output_directory : string
        (default=".")
    """

    demes = pd.read_csv(demes_path, sep="\t")

    if migration_rate == None and migration_rates == None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`.")
    elif migration_rate != None and migration_rates != None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`, not both.")
    elif migration_rate != None:
        deme_types = demes["type"].unique()
        transitions = list(combinations_with_replacement(deme_types, 2))
        migration_rates = {i:migration_rate for i in range(len(transitions))}

    if (not allow_multiple_samples_per_deme) and (number_of_samples > len(demes)):
        raise RuntimeError(f"There are more samples than demes ({number_of_samples} versus {len(demes)}). You must either allow multiple samples per deme or reduce the number of samples.")
    world_map = WorldMap(demes=demes)
    arg = _simulate_arg(
        world_map=world_map,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        ploidy=ploidy,
        number_of_samples=number_of_samples,
        allow_multiple_samples_per_deme=allow_multiple_samples_per_deme,
        pop_size=pop_size,
        migration_rates=migration_rates
    )
    arg.dump(f"{output_directory}/arg.trees")
    _create_samples_file_from_tree_sequence(
        ts=arg,
        output_samples_path=f"{output_directory}/samples.tsv"
    )

def create_samples_and_trees_files(
        demes_path,
        number_of_trees,
        number_of_samples,
        ploidy,
        allow_multiple_samples_per_deme,
        pop_size,
        migration_rate=None,
        migration_rates=None,
        output_directory="."
    ):
    """
    Parameters
    ----------
    demes_path : string
    number_of_trees : int
        The number of independent trees to simulate
    number_of_samples : int
    ploidy : int
        The ploidy of the samples
    allow_multiple_samples_per_deme : bool
    pop_size : int
        The population size of each deme
    migration_rate : int
        Single migration rate between neighboring demes. (default=None, ignored)
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type. (default=None, ignored)
    output_directory : string
        (default=".")
    """

    demes = pd.read_csv(demes_path, sep="\t")

    if migration_rate == None and migration_rates == None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`.")
    elif migration_rate != None and migration_rates != None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`, not both.")
    elif migration_rate != None:
        deme_types = demes["type"].unique()
        transitions = list(combinations_with_replacement(deme_types, 2))
        migration_rates = {i:migration_rate for i in range(len(transitions))}

    if (not allow_multiple_samples_per_deme) and (number_of_samples > len(demes)):
        raise RuntimeError(f"There are more samples than demes ({number_of_samples} versus {len(demes)}). You must either allow multiple samples per deme or reduce the number of samples.")
    world_map = WorldMap(demes=demes)
    mkdir(f"{output_directory}/trees")
    trees = _simulate_independent_trees(
        world_map=world_map,
        number_of_trees=number_of_trees,
        number_of_samples=number_of_samples,
        ploidy=ploidy,
        allow_multiple_samples_per_deme=allow_multiple_samples_per_deme,
        pop_size=pop_size,
        migration_rates=migration_rates
    )           
    for i,tree in enumerate(trees):
        tree.dump(f"{output_directory}/trees/{i}.trees")
    _create_samples_file_from_tree_sequence(
        ts=tree,
        output_samples_path=f"{output_directory}/samples.tsv"
    )

def create_grid_demography_dataset(
        side_length,
        number_of_deme_types,
        number_of_trees,
        number_of_samples,
        ploidy,
        allow_multiple_samples_per_deme,
        pop_size,
        migration_rate=None,
        migration_rates=None
    ):
    """Creates a new dataset based on the specified grid metapopulation demographic model

    Creates "demes.tsv", "samples.tsv", "trees/" folder with ".trees" files

    Parameters
    ----------
    side_length : int
        Number of demes along one side length of the grid
    number_of_deme_types : int
        Set the number of deme types that will be randomly assigned
    number_of_trees : int
        Number of trees to simulate from the demographic model
    number_of_samples : int
        Number of samples to place in the demes
    ploidy : int
        The ploidy (number of chromosomes) of those samples
    allow_multiple_samples_per_deme : bool
        Whether to allow randomly placing multiple samples into a deme
    pop_size : int
        Population size of each deme
    migration_rate : int
        Single migration rate between neighboring demes. (default=None, ignored)
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type. (default=None, ignored)
    """

    if migration_rate == None and migration_rates == None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`.")
    elif migration_rate != None and migration_rates != None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`, not both.")
    elif migration_rate != None:
        transitions = list(combinations_with_replacement([i for i in range(number_of_deme_types)], 2))
        migration_rates = {i:migration_rate for i in range(len(transitions))}

    _create_grid_demes_file(
        side_length=side_length,
        number_of_deme_types=number_of_deme_types,
        output_path="demes.tsv"
    )
    create_samples_and_trees_files(
        demes_path="demes.tsv",
        number_of_samples=number_of_samples,
        allow_multiple_samples_per_deme=allow_multiple_samples_per_deme,
        number_of_trees=number_of_trees,
        ploidy=ploidy,
        pop_size=pop_size,
        migration_rates=migration_rates
    )

def create_dataset_from_slim_output(ts, side_length, gap_between_trees=1, num_random_samples=-1):
    """Creates input files from SLiM simulation tree sequence

    Creates "demes.tsv", "samples.tsv", "trees/" folder with ".trees" files

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence output by the SLiM simulation
    side_length : int
        The number of demes along one side of the square grid demography
    gap_between_trees : int
        The gap between sampled trees in the tree sequence. (default=1 : all trees included)
    num_random_samples : int
        Simplify the tree sequence to this many samples. (default=-1 : ignored, all samples included)
    """

    if num_random_samples != -1:
        samples = list(np.random.choice(list(ts.samples()), num_random_samples, replace=False))
        ts = ts.simplify(samples=samples)
    _create_samples_file_from_tree_sequence(
        ts=ts,
        output_samples_path="samples.tsv"
    )
    mkdir("trees")
    for i in range(0, ts.num_trees, gap_between_trees):
        tree = ts.at_index(i)
        interval = tree.interval
        single_tree_ts = ts.keep_intervals([interval], simplify=True).trim()
        single_tree_ts.dump(f"trees/{i}.trees")
    _create_grid_demes_file(
        side_length=side_length,
        number_of_deme_types=1,
        output_path="demes.tsv"
    )
