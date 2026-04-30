import numpy as np
from scipy import linalg
import time
import matplotlib.pyplot as plt


def calc_stationary_distribution(transition_matrix):
    """Calculates the equilibrium frequencies based on the instantaneous transition rate matrix

    Parameters
    ----------
    transition_matrix : np.array
        Instantaneous rate matrix

    Returns
    -------
    stationary_distribution : np.array
        Equilibrium frequencies of demes
    """

    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    stationary_distribution = w/np.sum(w)
    return stationary_distribution

def convert_to_opposite_rate_matrix(transition_matrix):
    stationary_distribution = calc_stationary_distribution(transition_matrix)
    opposite = np.zeros((len(transition_matrix), len(transition_matrix)))
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix)):
            if i != j:
                opposite[i][j] = (stationary_distribution[j] * transition_matrix[j][i]) / stationary_distribution[i]
    diag = -np.sum(opposite, axis=1)
    np.fill_diagonal(opposite, diag)
    return opposite

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

def _calc_branch_message(current_pos, branch_above, transition_matrices, direction="backward"):
    """Calculates the message to be passed along a branch above specified node

    Parameters
    ----------
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

    included_epochs = np.where(branch_above > 0)[0]
    P = np.eye(len(current_pos))
    for epoch in included_epochs:
        P = np.dot(P, linalg.expm(transition_matrices[epoch]*branch_above[epoch]))
    if direction == "backward":
        message = np.dot(current_pos, P)
    else:
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
                current_pos,
                branch_above[:,id],
                transition_matrices
            )
        else:   # collect roots here
            messages[id] = current_pos
            loglikelihood += np.log(np.sum(current_pos))
    return loglikelihood, messages

def _calc_all_messages(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        backward_transition_matrices,
        forward_transition_matrices
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
    messages = np.ones((len(parents)*2, num_demes), dtype="float64")
    for id in ids_asc_time:
        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]]
        else:
            current_pos = _calc_current_pos(
                id,
                messages,
                parents
            )
        current_pos = current_pos / np.sum(current_pos)
        parent = parents[id]
        if parent != -1:
            messages[id] = _calc_branch_message(
                current_pos,
                branch_above[:,id],
                backward_transition_matrices,
                direction="backward"
            )
    for id in ids_asc_time[::-1]:
        parent_of = np.where(parents==id)[0]
        for c in range(len(parent_of)):
            alt_children = np.delete(parent_of, c)
            current_pos = np.prod(np.concatenate((messages[[id+len(parents)]], messages[alt_children])), axis=0)
            current_pos = current_pos / np.sum(current_pos)
            messages[parent_of[c]+len(parents)] = _calc_branch_message(
                current_pos,
                branch_above[:,parent_of[c]],
                forward_transition_matrices,
                direction="forward"
            )
    return messages

def estimate_node_positions(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        transition_matrices
    ):
    """Marginal scaled likelihoods of node positions
    """
    
    forward_transition_matrices = np.array([convert_to_opposite_rate_matrix(arr) for arr in transition_matrices])

    messages = _calc_all_messages(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        transition_matrices,
        forward_transition_matrices
    )

    for id in ids_asc_time:
        if id in sample_ids:
            print(id, sample_locations_array[np.where(sample_ids==id)[0][0]])
        else:
            combined = np.prod(np.concatenate((messages[[id+len(parents)]], messages[np.where(parents==id)[0]])), axis=0)
            print(id, combined/np.sum(combined))



ids_asc_time = np.array([0, 1, 2, 3, 4])
parents = np.array([3, 3, 4, 4, -1])
branch_above = np.array([[1, 1, 2, 1, 0]])
sample_locations_array = np.array([
    [1, 0],
    [0, 1],
    [0, 1]
])
sample_ids = np.array([0, 1, 2])

transition_matrices = np.array([
    [
        [-2, 2],
        [1, -1]
    ]
])

#print(convert_to_opposite_rate_matrix(transition_matrices[-1]))


#print(_calc_branch_message(
#    3,
#    np.array([0, 1]),
#    branch_above,
#    transition_matrices
#))

#like, messages = likelihood_of_tree(
#    parents,
#    branch_above,
#    ids_asc_time,
#    sample_locations_array,
#    sample_ids,
#    transition_matrices
#)




print(0.36652471*0.31673764*0.36652471*0.33250708+0.36652471*0.31673764*0.63347529*0.66749292)
print(0.63347529*0.68326236*0.31673764*0.33250708+0.63347529*0.68326236*0.68326236*0.66749292)


estimate_node_positions(
    parents,
    branch_above,
    ids_asc_time,
    sample_locations_array,
    sample_ids,
    transition_matrices
)