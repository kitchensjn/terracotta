
import numpy as np
import scipy
import matplotlib.pyplot as plt


def plot_messages(messages, save_to=None):
    """Plots the location probability distributions from each child and at the root

    Parameters
    ----------
    messages : np.array
        Messages being passed in tree. Shape is #nodes x #demes.
    save_to : str
        Path to output file. Default is None, ignored.
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    ax[0].bar(range(len(messages[0])), messages[0], width=1, color="#E95E0D", alpha=0.5)
    ax[0].bar(range(len(messages[1])), messages[1], width=1, alpha=0.5)
    ax[1].bar(range(len(messages[2])), messages[2], width=1, color="#000000")
    if save_to is not None:
        plt.savefig(save_to)
    plt.show()

def _calc_current_pos(id, messages, parents, pop_sizes):
    """Calculates current node position as product of child messages

    Parameters
    ----------
    id : int
        ID of node
    messages : np.array
        Messages being passed in tree. Shape is #nodes x #demes.
    parents : np.array
        Parent IDs for each node. Length is #nodes.
    pop_sizes : np.array
        Modified suitability values for the demes. Length is #demes.
    
    Returns
    -------
    current_pos : np.array
        Probability distribution of node's current position given subtree below. Length is #demes.
    """

    return np.multiply(1/pop_sizes, np.prod(messages[np.where(parents==id)[0]], axis=0))

def _calc_branch_message(id, current_pos, branch_above, transition_matrices):
    """Calculates the message to be passed along a branch above specified node

    Parameters
    ----------
    id : int
        ID of node
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    branch_above : np.array
        Branch lengths above each node split across epochs. Shape is #epochs x #nodes.
    transition_matrices : np.array
        Backward in time transition rates between demes across epochs. Shape is #epochs x #demes x #demes.

    Returns
    -------
    message : np.array
        Probability distribution for location of lineage given subtree below. Length is #demes.
    """

    bl = branch_above[:,id]
    included_epochs = np.where(bl > 0)[0]
    for epoch in included_epochs:
        current_pos = scipy.sparse.linalg.expm_multiply(transition_matrices[epoch]*bl[epoch], current_pos)
    return current_pos

def likelihood_of_tree(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        transition_matrices,
        pop_sizes
    ):
    """
    Parameters
    ----------
    parents : np.array
        Parent IDs for each node. Length is #nodes.
    branch_above : np.array
        Branch lengths above each node split across epochs. Shape is #epochs x #nodes.
    ids_asc_time : np.array
        IDs of nodes in tree in time ascending order. Length is #nodes.
    sample_locations_array : np.array
        Vector representation of sample locations. Shape is #samples x #demes.
    sample_ids : np.array
        IDs of sample nodes in tree. Length is #samples.
    transition_matrices : np.array
        Backward in time transition rates between demes across epochs. Shape is #epochs x #demes x #demes.
    pop_sizes : np.array
        Modified suitability values for the demes. Length is #demes.

    Returns
    -------
    loglikelihood : float
        Log-likelihood of the tree after pruning.
    messages : np.array
        Messages being passed in tree. Shape is #nodes x #demes.
    """

    num_demes = len(sample_locations_array[0])
    messages = np.zeros((len(parents), num_demes), dtype="float64")
    loglikelihood = 0
    for id in ids_asc_time: 
        
        # Within this loop, repeatedly implement Equation 2 from the manuscript

        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
        else:
            current_pos = _calc_current_pos(
                id,
                messages,
                parents,
                pop_sizes
            )
        
        # Also rescale so that underflow is not a problem and track scaler
        s = np.sum(current_pos)
        current_pos /= s
        loglikelihood += np.log(s)
        
        messages[id] = _calc_branch_message(
            id,
            current_pos,
            branch_above,
            transition_matrices
        )

    return loglikelihood, messages



# Tree
# --------
#   -2-
#  |   |
# 1|   |1
#  |   |
#  0   1

ids_asc_time = np.array([0, 1, 2])
parents = np.array([2, 2, -1])
branch_above = np.array([[1, 1, 0]])
sample_ids = np.array([0, 1])

# World map
# ---------
#  0  --  1  --  2  --  3  --  4  --  5  --  6  --  7  --  8  --  9 
# 0.1 -- 0.2 -- 0.3 -- 0.4 -- 0.5 -- 0.6 -- 0.7 -- 0.8 -- 0.9 -- 1.0
# Samples:       0                                  1

s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
sample_locations_array = np.array([
    [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0]
])

# Equation 1 from manuscript - b not included so assume that b = 1.
m = 1
a = 1
s = s**a
transition_matrices = np.array([
    [
        [-(m*(s[1]/s[0])), m*(s[0]/s[1]), 0, 0, 0, 0, 0, 0, 0, 0],
        [m*(s[1]/s[0]), -(m*(s[0]/s[1])+m*(s[2]/s[1])), m*(s[1]/s[2]), 0, 0, 0, 0, 0, 0, 0],
        [0, m*(s[2]/s[1]), -(m*(s[1]/s[2])+m*(s[3]/s[2])), m*(s[2]/s[3]), 0, 0, 0, 0, 0, 0],
        [0, 0, m*(s[3]/s[2]), -(m*(s[2]/s[3])+m*(s[4]/s[3])), m*(s[3]/s[4]), 0, 0, 0, 0, 0],
        [0, 0, 0, m*(s[4]/s[3]), -(m*(s[3]/s[4])+m*(s[5]/s[4])), m*(s[4]/s[5]), 0, 0, 0, 0],
        [0, 0, 0, 0, m*(s[5]/s[4]), -(m*(s[4]/s[5])+m*(s[6]/s[5])), m*(s[5]/s[6]), 0, 0, 0],
        [0, 0, 0, 0, 0, m*(s[6]/s[5]), -(m*(s[5]/s[6])+m*(s[7]/s[6])), m*(s[6]/s[7]), 0, 0],
        [0, 0, 0, 0, 0, 0, m*(s[7]/s[6]), -(m*(s[6]/s[7])+m*(s[8]/s[7])), m*(s[7]/s[8]), 0],
        [0, 0, 0, 0, 0, 0, 0, m*(s[8]/s[7]), -(m*(s[7]/s[8])+m*(s[9]/s[8])), m*(s[8]/s[9])],
        [0, 0, 0, 0, 0, 0, 0, 0, m*(s[9]/s[8]), -(m*(s[8]/s[9]))]
    ]
])

like, messages = likelihood_of_tree(
    parents,
    branch_above,
    ids_asc_time,
    sample_locations_array,
    sample_ids,
    transition_matrices,
    s
)

plot_messages(messages)