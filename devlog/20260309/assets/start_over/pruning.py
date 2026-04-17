import numpy as np
from scipy import linalg
import time
import matplotlib.pyplot as plt


## ADDITIONAL
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

def plot_messages(messages, ids_asc_time):
    for i in range(len(ids_asc_time)):
        plt.bar(range(len(messages[i])), messages[i], width=1, color="#E95E0D")
        plt.show()


## MAIN

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
        transition_matrices,
        stationary_distribution
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
    stationary_distribution : np.array
        Stationary distribution of deepest transition matrix

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
            #loglikelihood += np.log(np.sum(np.multiply(stationary_distribution, current_pos)))
    return loglikelihood, messages




ids_asc_time = np.array([0, 1, 2])
parents = np.array([2, 2, -1])
branch_above = np.array([[1, 1, 0]])
sample_locations_array = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
])
sample_ids = np.array([0, 1])


m = 1
s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
s = s ** 0
s = s / sum(s)

transition_matrices = np.array([
    [
        [-(m*s[1]), m*s[1], 0, 0, 0, 0, 0, 0, 0, 0],
        [m*s[0], -(m*s[0]+m*s[2]), m*s[2], 0, 0, 0, 0, 0, 0, 0],
        [0, m*s[1], -(m*s[1]+m*s[3]), m*s[3], 0, 0, 0, 0, 0, 0],
        [0, 0, m*s[2], -(m*s[2]+m*s[4]), m*s[4], 0, 0, 0, 0, 0],
        [0, 0, 0, m*s[3], -(m*s[3]+m*s[5]), m*s[5], 0, 0, 0, 0],
        [0, 0, 0, 0, m*s[4], -(m*s[4]+m*s[6]), m*s[6], 0, 0, 0],
        [0, 0, 0, 0, 0, m*s[5], -(m*s[5]+m*s[7]), m*s[7], 0, 0],
        [0, 0, 0, 0, 0, 0, m*s[6], -(m*s[6]+m*s[8]), m*s[8], 0],
        [0, 0, 0, 0, 0, 0, 0, m*s[7], -(m*s[7]+m*s[9]), m*s[9]],
        [0, 0, 0, 0, 0, 0, 0, 0, m*s[8], -(m*s[8])]
    ]
])
stationary_distribution = calc_stationary_distribution(transition_matrices[-1])

like, messages = likelihood_of_tree(
    parents,
    branch_above,
    ids_asc_time,
    sample_locations_array,
    sample_ids,
    transition_matrices,
    stationary_distribution
)

#print(like)
#plot_messages(messages, ids_asc_time)





ids_asc_time = np.array([0, 1, 2])
parents = np.array([2, 2, -1])
branch_above = np.array([[1, 1, 0]])
sample_locations_array = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
])
sample_ids = np.array([0, 1])


m = 1
n = 0

transition_matrices = np.array([
    [
        [-m, m, 0, 0, 0, 0, 0, 0, 0, 0],
        [n, -(m+n), m, 0, 0, 0, 0, 0, 0, 0],
        [0, n, -(m+n), m, 0, 0, 0, 0, 0, 0],
        [0, 0, n, -(m+n), m, 0, 0, 0, 0, 0],
        [0, 0, 0, n, -(m+n), m, 0, 0, 0, 0],
        [0, 0, 0, 0, n, -(m+n), m, 0, 0, 0],
        [0, 0, 0, 0, 0, n, -(m+n), m, 0, 0],
        [0, 0, 0, 0, 0, 0, n, -(m+n), m, 0],
        [0, 0, 0, 0, 0, 0, 0, n, -(m+n), m],
        [0, 0, 0, 0, 0, 0, 0, 0, n, -n]
    ]
])
stationary_distribution = calc_stationary_distribution(transition_matrices[-1])

like, messages = likelihood_of_tree(
    parents,
    branch_above,
    ids_asc_time,
    sample_locations_array,
    sample_ids,
    transition_matrices,
    stationary_distribution
)

#print(like)
#plot_messages(messages, ids_asc_time)







def _calc_current_pos_penalty(id, messages, parents, pop_sizes):
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

    return np.multiply(1/pop_sizes, np.prod(messages[np.where(parents==id)[0]], axis=0))

def likelihood_of_tree_penalty(
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
            current_pos = _calc_current_pos_penalty(
                id,
                messages,
                parents,
                pop_sizes
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
            #loglikelihood += np.log(np.sum(np.multiply(stationary_distribution, current_pos)))
    return loglikelihood, messages





ids_asc_time = np.array([0, 1, 2])
parents = np.array([2, 2, -1])
branch_above = np.array([[1, 1, 0]])
sample_locations_array = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
])
sample_ids = np.array([0, 1])


m = 1
s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
a = 30
s = s**a
s = s*1000

transition_matrices = np.array([
    [
        [-(m*(s[1]/s[0])), m*(s[1]/s[0]), 0, 0, 0, 0, 0, 0, 0, 0],
        [m*(s[0]/s[1]), -(m*(s[0]/s[1])+m*(s[2]/s[1])), m*(s[2]/s[1]), 0, 0, 0, 0, 0, 0, 0],
        [0, m*(s[1]/s[2]), -(m*(s[1]/s[2])+m*(s[3]/s[2])), m*(s[3]/s[2]), 0, 0, 0, 0, 0, 0],
        [0, 0, m*(s[2]/s[3]), -(m*(s[2]/s[3])+m*(s[4]/s[3])), m*(s[4]/s[3]), 0, 0, 0, 0, 0],
        [0, 0, 0, m*(s[3]/s[4]), -(m*(s[3]/s[4])+m*(s[5]/s[4])), m*(s[5]/s[4]), 0, 0, 0, 0],
        [0, 0, 0, 0, m*(s[4]/s[5]), -(m*(s[4]/s[5])+m*(s[6]/s[5])), m*(s[6]/s[5]), 0, 0, 0],
        [0, 0, 0, 0, 0, m*(s[5]/s[6]), -(m*(s[5]/s[6])+m*(s[7]/s[6])), m*(s[7]/s[6]), 0, 0],
        [0, 0, 0, 0, 0, 0, m*(s[6]/s[7]), -(m*(s[6]/s[7])+m*(s[8]/s[7])), m*(s[8]/s[7]), 0],
        [0, 0, 0, 0, 0, 0, 0, m*(s[7]/s[8]), -(m*(s[7]/s[8])+m*(s[9]/s[8])), m*(s[9]/s[8])],
        [0, 0, 0, 0, 0, 0, 0, 0, m*(s[8]/s[9]), -(m*(s[8]/s[9]))]
    ]
])

like, messages = likelihood_of_tree_penalty(
    parents,
    branch_above,
    ids_asc_time,
    sample_locations_array,
    sample_ids,
    transition_matrices,
    s
)

print(like)
plot_messages(messages, ids_asc_time)







exit()




#for i in range(len(ids_asc_time)):
#    plt.bar(range(len(messages[i])), messages[i], width=1, color="#E95E0D")
#    plt.savefig(f"figures/{i}.png")
#    plt.show()




# Tree from appendix of https://lukejharmon.github.io/pcm/chapter8_fitdiscrete/

ids_asc_time = np.array([0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10])
parents = np.array([6, 6, 8, 7, 7, 10, 8, 9, 9, 10, -1])
branch_above = np.array([[1, 1, 1.5, 0.5, 0.5, 2.5, 0.5, 2.0, 1.0, 0.5, 0]])
sample_locations_array = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0]
])
sample_ids = np.array([0, 1, 2, 3, 4, 5])

transition_matrices = np.array([
    [
        [-2, 1, 1],
        [1, -2, 1],
        [1, 1, -2]
    ]
])

print(transition_matrices[0])

stationary_distribution = np.array([1/3, 1/3, 1/3])

print(stationary_distribution)

like = likelihood_of_tree(
    parents,
    branch_above,
    ids_asc_time,
    sample_locations_array,
    sample_ids,
    transition_matrices,
    stationary_distribution
)

print(like)


# Tree from `geiger` package

ids_asc_time = np.array([0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10, 11, 12, 24, 23, 21, 22, 20, 19, 18, 17, 16, 15, 14, 13])
parents = np.array([21, 21, 20, 19, 18, 17, 22, 24, 24, 23, 15, 14, 13, -1, 13, 14, 15, 16, 17, 18, 19, 20, 16, 22, 23])
branch_above = np.array([[0.05500, 0.05500, 0.11000, 0.18333, 0.19250, 0.22800, 0.08667, 0.02000, 0.02000, 0.03500, 0.46550, 0.53409, 0.58333, 0, 0.04924, 0.06859, 0.13404, 0.10346, 0.03550, 0.00917, 0.07333, 0.05500, 0.24479, 0.05167, 0.01500]])
sample_locations_array = np.array([
    [0, 1, 0],  #ful
    [0, 1, 0],  #for
    [0, 0, 1],  #mag
    [0, 1, 0],  #con
    [0, 1, 0],  #sca
    [0, 1, 0],  #dif
    [0, 1, 0],  #pal
    [0, 1, 0],  #par
    [0, 1, 0],  #psi
    [0, 1, 0],  #pau
    [0, 1, 0],  #Pla
    [1, 0, 0],  #fus
    [0, 1, 0],  #Pin
])

sample_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
transition_matrices = np.array([
    [
        [-2.652820e+00, 2.652820, 1.088657e-16],
        [4.973356e-20, -1.662759, 1.662759e+00],
        [2.788172e-16, 16.148001, -1.614800e+01]
    ]
])

stationary_distribution = calc_stationary_distribution(transition_matrices[-1])
stationary_distribution = np.array([1/3, 1/3, 1/3])

like = likelihood_of_tree(
    parents,
    branch_above,
    ids_asc_time,
    sample_locations_array,
    sample_ids,
    transition_matrices,
    stationary_distribution
)

print(like)