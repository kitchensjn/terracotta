import numpy as np
from scipy import linalg
from scipy.special import logsumexp


def calc_tree_log_likelihood(tree, sample_locs, transition_matrix):
    """Calculates the log_likelihood of the tree using Felsenstein's Pruning Algorithm.

    NOTE: This only works for trees with a single root.
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
        likelihood of the tree (sum of the root locations likelihood vector)
    locs[node] : np.array
        root locations likelihood vector
    """

    log_messages = {}
    for l in sample_locs:
        log_messages[l] = np.log(np.matmul(sample_locs[l], linalg.expm(transition_matrix*tree.branch_length(l))))

    for node in tree.nodes(order="timeasc"):
        children = tree.children(node)
        if len(children) > 0:
            incoming_log_messages = []
            for child in children:
                incoming_log_messages.append(log_messages[child])
            summed_log_messages = np.sum(incoming_log_messages, axis=0)
            outgoing_log_message = np.array([logsumexp(np.log(linalg.expm(transition_matrix*tree.branch_length(child))).T + summed_log_messages, axis=1)])
            log_messages[node] = outgoing_log_message
    return logsumexp(outgoing_log_message), outgoing_log_message

def create_grid_world_map_file(side_length, path="world_map.tsv"):
    """Creates the world map input file that pairs with grid.slim of specified size

    Parameters
    ----------
    side_length : int
        Number of populations along one side of the square grid metapopulation (`metapopSide` in grid.slim)
    path : str
        Path to output file. (default="world_map.tsv")
    """

    with open(path, "w") as outfile:
        outfile.write("id\txcoord\tycoord\tneighbours\n")
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
                outfile.write(f"{x+y*side_length}\t{x}\t{y}\t{",".join(neighbors)}\n")
            
