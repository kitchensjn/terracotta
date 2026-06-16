import numpy as np
import scipy
import time
import matplotlib.pyplot as plt

def plot_messages(messages, save_to=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    ax[0].bar(range(len(messages[0])), messages[0], width=1, color="#E95E0D", alpha=0.5)
    ax[0].bar(range(len(messages[1])), messages[1], width=1, alpha=0.5)
    ax[1].bar(range(len(messages[2])), messages[2], width=1, color="#000000")
    if save_to is not None:
        plt.savefig(save_to)
    plt.show()

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
        P = np.dot(P, scipy.linalg.expm(transition_matrices[epoch]*bl[epoch]))
    message = np.dot(P, current_pos)
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
            messages[id] = current_pos
            loglikelihood += np.log(np.sum(current_pos))
    return loglikelihood, messages


def _calc_current_pos_log(id, messages, parents):
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

    return np.sum(messages[np.where(parents==id)[0]], axis=0)

def logsumexp_custom(x, axis):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c), axis=axis))

def _calc_branch_message_log(id, current_pos, branch_above, transition_matrices):
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
        P = np.dot(P, scipy.linalg.expm(transition_matrices[epoch]*bl[epoch]))
    message = logsumexp_custom(current_pos + np.log(P), axis=1)
    return message

def likelihood_of_tree_log(
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
            current_pos = _calc_current_pos_log(
                id,
                messages,
                parents
            )
        parent = parents[id]
        if parent != -1:
            messages[id] = _calc_branch_message_log(
                id,
                current_pos,
                branch_above,
                transition_matrices
            )
        else:   # collect roots here
            loglikelihood += logsumexp_custom(current_pos, axis=0)
    return loglikelihood, messages


def _calc_branch_message_rescaled(id, current_pos, branch_above, transition_matrices):
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
    for epoch in included_epochs:
        current_pos = scipy.sparse.linalg.expm_multiply(transition_matrices[epoch]*bl[epoch], current_pos)
    return current_pos

def likelihood_of_tree_rescaled(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        transition_matrices
    ):

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
        
        s = np.sum(current_pos)
        current_pos /= s
        loglikelihood += np.log(s)
        
        parent = parents[id]
        if parent != -1:
            messages[id] = _calc_branch_message_rescaled(
                id,
                current_pos,
                branch_above,
                transition_matrices
            )
        else:   # collect roots here
            messages[id] = current_pos

    return loglikelihood, messages



ids_asc_time = np.array([0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10])
parents = np.array([6, 6, 8, 7, 7, 10, 8, 9, 9, 10, -1])
branch_above = np.array([[10000, 10000, 15000, 5000, 5000, 25000, 5000, 20000, 10000, 5000, 0]])
sample_locations_array = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0]
])
sample_locations_array = np.maximum(sample_locations_array, 1e-99)
sample_ids = np.array([0, 1, 2, 3, 4, 5])

transition_matrices = np.array([
    [
        [-3, 2, 0],
        [2, -3, 1],
        [1, 1, -1]
    ]
])

start = time.time()
like, messages = likelihood_of_tree(
    parents,
    branch_above,
    ids_asc_time,
    sample_locations_array,
    sample_ids,
    transition_matrices
)
print(time.time() - start)

sample_locations_array_log = np.log(sample_locations_array)

start = time.time()
like, messages = likelihood_of_tree_log(
    parents,
    branch_above,
    ids_asc_time,
    sample_locations_array_log,
    sample_ids,
    transition_matrices
)
print(time.time() - start)

start = time.time()
like, messages = likelihood_of_tree_rescaled(
    parents,
    branch_above,
    ids_asc_time,
    sample_locations_array,
    sample_ids,
    transition_matrices
)
print(time.time() - start)