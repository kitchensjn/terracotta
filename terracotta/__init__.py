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