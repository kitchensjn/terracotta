import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.linalg import expm, eig
from numba import njit, prange


def _calc_current_pos(id, messages, parents):
    """Calculates current node position as product of child messages

    Parameters
    ----------
    id : int
        ID of node
    messages : np.array
        Messages being passed in tree. Shape is #nodes x #demes.
    parents : np.array
        Parent IDs for each node. Length is #nodes.
    
    Returns
    -------
    current_pos : np.array
        Probability distribution of node's current position given subtree below. Length is #demes.
    """

    current_pos = np.prod(messages[np.where(parents==id)[0]], axis=0)
    return current_pos

def _calc_branch_message(
        id,
        current_pos,
        branch_above,
        unique_branch_lengths,
        precalculated_transitions
    ):
    """Calculates the message to be passed along a branch above specified node

    Parameters
    ----------
    id : int
        ID of node
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    branch_above : np.array
        Branch lengths above each node split across epochs. Shape is #epochs x #nodes.
    unique_branch_lengths : list
        Arrays containing unique branch lengths in each epoch.
    precalculated_transitions : list
        Arrays of transition probabilities corresponding with the unique_branch_lengths.

    Returns
    -------
    current_pos : np.array
        Probability distribution for location of lineage given subtree below. Length is #demes.
    """

    bl = branch_above[:,id]
    included_epochs = np.where(bl > 0)[0]
    for epoch in included_epochs:
        trans_prob = precalculated_transitions[epoch][np.argwhere(unique_branch_lengths[epoch]==bl[epoch])[0][0]]
        trans_prob = np.maximum(trans_prob, 0)
        current_pos = np.matmul(trans_prob, current_pos)
    return current_pos

def likelihood_of_tree(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        unique_branch_lengths,
        precalculated_transitions
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
    unique_branch_lengths : list
        Arrays containing unique branch lengths in each epoch.
    precalculated_transitions : list
        Arrays of transition probabilities corresponding with the unique_branch_lengths.

    Returns
    -------
    loglikelihood : float
        Log-likelihood of the tree after pruning
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
                parents
            )

        # Also rescale so that underflow is not a problem and track scaler
        s = np.sum(current_pos)
        current_pos /= s
        loglikelihood += np.log(s)
        
        messages[id] = _calc_branch_message(
            id,
            current_pos,
            branch_above,
            unique_branch_lengths,
            precalculated_transitions
        )

    return loglikelihood

@njit()
def _calc_current_pos_log(id, messages, parents):
    """Calculates current node position as product of child messages

    Parameters
    ----------
    id : int
        ID of node
    messages : np.array
        Messages being passed in tree. Shape is #nodes x #demes.
    parents : np.array
        Parent IDs for each node. Length is #nodes.
    
    Returns
    -------
    current_pos : np.array
        Probability distribution of node's current position given subtree below. Length is #demes.
    """

    current_pos = np.sum(messages[np.where(parents==id)[0]], axis=0)[np.newaxis, :]
    return current_pos

@njit()
def logsumexp_custom(x, axis):
    """LogSumExp function that can be accessed from numba

    Parameters
    ----------
    x : numpy.ndarray
        Vector to transform
    axis : int
        Which axis of the numpy.array to transform over
    """

    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c), axis=axis))

@njit()
def _calc_branch_message_log(
        id,
        current_pos,
        branch_above,
        unique_branch_lengths,
        precalculated_transitions
    ):
    """Calculates the message to be passed along a branch above specified node

    Parameters
    ----------
    id : int
        ID of node
    current_pos : np.array
        Probability distribution of node's current position given subtree below
    unique_branch_lengths : list
        Arrays containing unique branch lengths in each epoch
    precalculated_transitions : list
        Arrays of transition probabilities corresponding with the unique_branch_lengths, converted to log space
    
    Returns
    -------
    message : np.array
        Probability distribution for location of lineage given subtree below. Length is #demes.
    """

    bl = branch_above[:,id]
    included_epochs = np.where(bl > 0)[0]
    for epoch in included_epochs:
        trans_prob = precalculated_transitions[epoch][np.argwhere(unique_branch_lengths[epoch]==bl[epoch])[0][0]]
        current_pos = logsumexp_custom(trans_prob + current_pos, axis=1)[np.newaxis, :]
    return current_pos

@njit()
def likelihood_of_tree_log(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        unique_branch_lengths,
        precalculated_transitions
    ):
    """Computes log-likelihood of tree in log-space to avoid underflow

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
    unique_branch_lengths : list
        Arrays containing unique branch lengths in each epoch
    precalculated_transitions : list
        Arrays of transition probabilities corresponding with the unique_branch_lengths, converted to log space

    Returns
    -------
    loglikelihood : float
        Log-likelihood of the tree
    """

    num_demes = len(sample_locations_array[0])
    messages = np.zeros((len(parents), num_demes), dtype="float64")
    loglikelihood = 0
    for id in ids_asc_time:
        if id in sample_ids:
            current_pos = sample_locations_array[np.where(sample_ids==id)[0][0]][np.newaxis, :]
        else:
            current_pos = _calc_current_pos_log(
                id,
                messages,
                parents
            )
        messages[id] = _calc_branch_message_log(
            id,
            current_pos,
            branch_above,
            unique_branch_lengths,
            precalculated_transitions
        )
        if parents[id] == -1:
            loglikelihood += logsumexp_custom(current_pos, axis=1)[0]
    return loglikelihood

@njit(parallel=True)
def _process_trees(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        unique_branch_lengths,
        precalculated_transitions
    ):
    """
    Parameters
    ----------
    parents : list
        Arrays containing ID of parent for each node, one array per tree
    branch_above : list
        Arrays containing branch above length (split across epochs) for each node, one array per tree
    ids_asc_time : List
        Arrays of nodes IDs in time ascending order, one array per tree
    sample_locations_array : numpy.ndarray
        Probability distribution vector for each sample location, converted to log space
    sample_ids : numpy.ndarray
        Order of sample IDs for `sample_locations_array`
    unique_branch_lengths : list
        Arrays containing unique branch lengths in each epoch
    precalculated_transitions : list
        Arrays of transition probabilities corresponding with the unique_branch_lengths, converted to log space
    
    Returns
    -------
    composite_likelihood : float
        Composite log-likelihood across the trees
    """

    composite_likelihood = 0
    for i in prange(len(branch_above)):
        like = likelihood_of_tree_log(
            parents=parents[i],
            branch_above=branch_above[i],
            ids_asc_time=ids_asc_time[i],
            sample_locations_array=sample_locations_array,
            sample_ids=sample_ids,
            unique_branch_lengths=unique_branch_lengths,
            precalculated_transitions=precalculated_transitions
        )
        composite_likelihood += like
    return composite_likelihood

def precalculate_transitions(branch_lengths, transition_matrices, method="MP"):
    """Calculates the transition probabilities between demes for each branch length

    Parameters
    ----------
    branch_lengths : list
        Arrays of unique branch lengths in each epoch
    transition_matrices : np.ndarray
        Instantaneous migration rate matrices, output of WorldMap.build_transition_matrices()
    method : string
        Code for the method used for the matrix exponentiation. Default is MP (matrix power).

    Returns
    -------
    precalculated_transitions : list
        Arrays of transition probabilities associated with each branch length, one array per epoch
    """

    precalculated_transitions = []
    if method == "MP":
        for e in range(len(branch_lengths)):
            num_demes = transition_matrices[e].shape[0]
            transitions = np.zeros((len(branch_lengths[e]), num_demes, num_demes), dtype="float64")
            exponentiated = expm(transition_matrices[e])
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
            precalculated_transitions.append(transitions)
    else:
        for e in range(len(branch_lengths)):
            num_demes = transition_matrices[e].shape[0]
            transitions = np.zeros((len(branch_lengths[e]), num_demes, num_demes), dtype="float64")
            for i in range(len(branch_lengths[e])):
                transitions[i] = expm(transition_matrices[e]*branch_lengths[e][i])
            precalculated_transitions.append(transitions)
    return precalculated_transitions

def calc_composite_likelihood_for_parameters(
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
    """
    Parameters
    ----------
    parameters : numpy.ndarray
        Combination of parameters used to build the migration surface
    world_map : terracotta.WorldMap
        Custom object built using the `demes.tsv`, `connections.tsv`, and `samples.tsv` files
    parents : list
        Arrays containing ID of parent for each node, one array per tree
    branch_above : list
        Arrays containing branch above length (split across epochs) for each node, one array per tree
    unique_branch_lengths : list
        List of lists containing unique branch lengths in each epoch
    ids_asc_time : list
        Arrays of nodes IDs in time ascending order, one array per tree
    sample_locations_array : numpy.ndarray
        Probability distribution vector for each sample location (generally 0 in all demes except one)
    sample_ids : numpy.ndarray
        Order of sample IDs for `sample_locations_array`
    output_file : str
        Path to an output file to write to (default is `None`, ignored)
    verbose : bool
        Whether to print log-likelihoods to the terminal (default is False)

    Returns
    -------
    composite_likelihood : float
        Log-likelihood of the parameter combination
    """

    for p in range(len(world_map.parameters)):
        if world_map.parameters[p] != "alpha":
            parameters[p] = np.exp(parameters[p])

    if "alpha" in world_map.parameters:
        alpha = parameters[world_map.parameters.index("alpha")]
    else:
        alpha = 1

    transition_matrices = world_map.build_transition_matrices(parameters=parameters)
    pop_sizes = world_map.suitabilities ** alpha
    precalculated_transitions = precalculate_transitions(unique_branch_lengths, transition_matrices, method="MP")
    precalculated_transitions_log = np.log(np.maximum(precalculated_transitions, 1e-99))
    
    composite_likelihood = _process_trees(
        parents=parents,
        branch_above=branch_above,
        ids_asc_time=ids_asc_time,
        sample_locations_array=sample_locations_array,
        sample_ids=sample_ids,
        unique_branch_lengths=unique_branch_lengths,
        precalculated_transitions=precalculated_transitions_log
    )

    if output_file is not None:
        with open(output_file, "a") as outfile:
            outfile.write(f"{parameters}\t{composite_likelihood}\n")
    if verbose:
        print(parameters, composite_likelihood, flush=True)
    return composite_likelihood



if __name__ == "__main__":
    # Tree
    # --------
    #   -2-
    #  |   |
    # 1|   |1
    #  |   |
    #  0   1

    ids_asc_time = np.array([0, 1, 2])
    parents = np.array([2, 2, -1])
    branch_above = np.array([
        [1, 1, 0]
    ])
    sample_ids = np.array([0, 1])

    # World map
    # ---------
    # ID:           0  --  1  --  2  --  3  --  4  --  5  --  6  --  7  --  8  --  9 
    # Suitability: 0.1 -- 0.2 -- 0.3 -- 0.4 -- 0.5 -- 0.6 -- 0.7 -- 0.8 -- 0.9 -- 1.0
    # Samples:                    0                                  1

    s = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ])
    coal_rate = 1/s
    coal_rate_log = np.log(coal_rate)

    sample_locations_array = np.array([
        [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0]
    ])
    sample_locations_array = np.maximum(sample_locations_array, 1e-99)
    sample_locations_array_log = np.log(sample_locations_array)

    # Equation 1 from manuscript - b not included so assume that b = 1.
    m = 1
    a = 1
    s = s**a
    
    transition_matrices = np.array([
        [
            [-(m*(s[0][1]/s[0][0])), m*(s[0][0]/s[0][1]), 0, 0, 0, 0, 0, 0, 0, 0],
            [m*(s[0][1]/s[0][0]), -(m*(s[0][0]/s[0][1])+m*(s[0][2]/s[0][1])), m*(s[0][1]/s[0][2]), 0, 0, 0, 0, 0, 0, 0],
            [0, m*(s[0][2]/s[0][1]), -(m*(s[0][1]/s[0][2])+m*(s[0][3]/s[0][2])), m*(s[0][2]/s[0][3]), 0, 0, 0, 0, 0, 0],
            [0, 0, m*(s[0][3]/s[0][2]), -(m*(s[0][2]/s[0][3])+m*(s[0][4]/s[0][3])), m*(s[0][3]/s[0][4]), 0, 0, 0, 0, 0],
            [0, 0, 0, m*(s[0][4]/s[0][3]), -(m*(s[0][3]/s[0][4])+m*(s[0][5]/s[0][4])), m*(s[0][4]/s[0][5]), 0, 0, 0, 0],
            [0, 0, 0, 0, m*(s[0][5]/s[0][4]), -(m*(s[0][4]/s[0][5])+m*(s[0][6]/s[0][5])), m*(s[0][5]/s[0][6]), 0, 0, 0],
            [0, 0, 0, 0, 0, m*(s[0][6]/s[0][5]), -(m*(s[0][5]/s[0][6])+m*(s[0][7]/s[0][6])), m*(s[0][6]/s[0][7]), 0, 0],
            [0, 0, 0, 0, 0, 0, m*(s[0][7]/s[0][6]), -(m*(s[0][6]/s[0][7])+m*(s[0][8]/s[0][7])), m*(s[0][7]/s[0][8]), 0],
            [0, 0, 0, 0, 0, 0, 0, m*(s[0][8]/s[0][7]), -(m*(s[0][7]/s[0][8])+m*(s[0][9]/s[0][8])), m*(s[0][8]/s[0][9])],
            [0, 0, 0, 0, 0, 0, 0, 0, m*(s[0][9]/s[0][8]), -(m*(s[0][8]/s[0][9]))]
        ]
    ])

    unique_branch_lengths = []
    for e in range(len(branch_above)):
        unique_branch_lengths.append(np.unique(branch_above[e]))

    precalculated_transitions = precalculate_transitions(unique_branch_lengths, transition_matrices)
    precalculated_transitions = np.maximum(1e-99, precalculated_transitions)
    precalculated_transitions_log = np.log(precalculated_transitions)

    like = likelihood_of_tree(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array,
        sample_ids,
        unique_branch_lengths,
        precalculated_transitions
    )

    print(like)

    like = likelihood_of_tree_log(
        parents,
        branch_above,
        ids_asc_time,
        sample_locations_array_log,
        sample_ids,
        unique_branch_lengths,
        precalculated_transitions_log
    )

    print(like)