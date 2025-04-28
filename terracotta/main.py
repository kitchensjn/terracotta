import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy import linalg
from scipy.special import logsumexp


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

    def draw(self, color_demes=False, color_connections=False, migration_rates=None, save_to=None):
        """Draws the world map

        Uses matplotlib.pyplot

        Parameters
        ----------
        color_demes : bool
            Whether to color the demes based on type
        color_connections : bool
            Whether to color the connections based on type
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
        if isinstance(self.samples, pd.DataFrame):
            counts = self.samples["deme"].value_counts().reset_index()
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

def calc_migration_rate_log_likelihood(world_map, trees, migration_rates):
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
    log_likelihoods = []
    for tree in trees:
        log_likelihoods.append(_calc_tree_log_likelihood(tree, world_map.sample_location_vectors, transition_matrix)[0])
    mr_log_like = sum(log_likelihoods)
    return mr_log_like, log_likelihoods