from numba import jit, njit, prange
import numpy as np
import pandas as pd
import tskit
import time
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
from glob import glob
from scipy import linalg


rate = 1e-5
demes = pd.read_csv("datasets/one_sample_per_deme/demes.tsv", sep="\t")
samples = pd.read_csv("datasets/one_sample_per_deme/samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)
trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]).first() for ts in glob(f"datasets/one_sample_per_deme/m{rate}/rep0/trees/*")]

cl = []
bal = []
r = []
for tree in trees:
    child_list, branch_above_list, roots = tct.convert_tree_to_tuple_list(tree)
    cl.append(child_list)
    bal.append(branch_above_list)
    r.append(roots)

total_number_of_edges = 0
for tree in trees:
    total_number_of_edges += tree.num_edges+1
branch_lengths = np.zeros(total_number_of_edges, dtype="int64")
edge_counter = 0
for tree in trees:
    for node in tree.nodes(order="timeasc"):
        branch_lengths[edge_counter] = int(tree.branch_length(node))
        edge_counter += 1
branch_lengths = np.unique(np.array(branch_lengths))

mr = np.array([1e-5])
transition_matrix = world_map.build_transition_matrix(migration_rates=mr)

def true_transitions(branch_lengths, transition_matrix):
    exponentiated = linalg.expm(transition_matrix)
    num_demes = exponentiated.shape[0]
    transitions = np.zeros((len(branch_lengths), num_demes, num_demes), dtype="float64")
    for i in range(len(branch_lengths)):
        transitions[i] = np.linalg.matrix_power(exponentiated, branch_lengths[i])
    return transitions


def precalculate_transitions(branch_lengths, transition_matrix):
    """Calculates the transition probabilities between demes for each branch length

    Parameters
    ----------
    branch_lengths : np.ndarray
        Array of branch lengths of increasing size
    transition_matrix : np.ndarray
        Instantaneous migration rate matrix, output of WorldMap.build_transition_matrix()
    
    Returns
    -------
    transitions : np.ndarray
    """
    exponentiated = linalg.expm(transition_matrix)
    num_demes = exponentiated.shape[0]
    transitions = np.zeros((len(branch_lengths), num_demes, num_demes), dtype="float64")
    transitions[0] = np.linalg.matrix_power(exponentiated, branch_lengths[0])
    for i in range(1, len(branch_lengths)):
        transitions[i] = np.dot(
            transitions[i-1],
            np.linalg.matrix_power(exponentiated, branch_lengths[i] - branch_lengths[i-1])
        )
    return transitions

def grahams_suggestion(branch_lengths, transition_matrix, fast=True):
    """Calculates the transition probabilities between demes for each branch length

    Parameters
    ----------
    branch_lengths : np.ndarray
        Array of branch lengths of increasing size
    transition_matrix : np.ndarray
        Instantaneous migration rate matrix, output of WorldMap.build_transition_matrix()
    fast : bool
        Whether to use the faster but less numerically stable algorithm (default is True)
    
    Returns
    -------
    transitions : np.ndarray
        3D array with transitions probabilities for each branch length
    """
    exponentiated = linalg.expm(transition_matrix)
    num_demes = exponentiated.shape[0]
    transitions = np.zeros((len(branch_lengths), num_demes, num_demes), dtype="float64")
    if fast:
        transitions[0] = np.linalg.matrix_power(exponentiated, branch_lengths[0])
        for i in range(1, len(branch_lengths)):
            for j in range(i-1, -1, -1):
                if branch_lengths[j] > 0:
                    power = branch_lengths[i] / branch_lengths[j]
                    if power % 1 == 0:
                        transitions[i] = np.linalg.matrix_power(transitions[j], int(power))
                        break
                else:
                    transitions[i] = np.linalg.matrix_power(exponentiated, branch_lengths[i])
                    break
                if j == 0:
                    transitions[i] = np.linalg.matrix_power(exponentiated, branch_lengths[i])
        return transitions
    for i in range(len(branch_lengths)):
        transitions[i] = np.linalg.matrix_power(exponentiated, branch_lengths[i])
    return transitions

print("Branch lengths:", branch_lengths)

start = time.time()
t0 = true_transitions(branch_lengths, transition_matrix)
print("Slow method:", time.time() - start, "seconds")

start = time.time()
t1 = precalculate_transitions(branch_lengths, transition_matrix)
print("Current method:", time.time() - start, "seconds")

start = time.time()
t2 = grahams_suggestion(branch_lengths, transition_matrix)
print("Graham's suggestion:", time.time() - start, "seconds")

bl = len(branch_lengths)-1
#print((t0[bl]-t1[bl])/t0[bl])
#print()
#print((t0[bl]-t2[bl])/t0[bl])
