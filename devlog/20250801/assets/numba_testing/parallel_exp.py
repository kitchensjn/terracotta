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


def precalculate_transitions(branch_lengths, transition_matrix, fast=True):
    """Calculates the transition probabilities between demes for each branch length

    Parameters
    ----------
    branch_lengths : np.ndarray
        Array of branch lengths of increasing size
    transition_matrix : np.ndarray
        Instantaneous migration rate matrix, output of WorldMap.build_transition_matrix()
    fast : bool
        Speeds up calculation but can aggregate floating point errors for longer times
    """

    num_demes = transition_matrix.shape[0]
    exponentiated = linalg.expm(transition_matrix)
    previous_length = -1
    precomputed_transitions = np.zeros((len(branch_lengths), num_demes, num_demes), dtype="float64")
    precomputed_log = np.zeros((len(branch_lengths), num_demes, num_demes), dtype="float64")
    counter = 0
    for bl in branch_lengths:
        if fast and previous_length != -1:
            diff = bl - previous_length
            where_next = np.dot(previous_mat, np.linalg.matrix_power(exponentiated, diff))
        else:
            where_next = np.linalg.matrix_power(exponentiated, bl)
        precomputed_transitions[counter] = where_next
        precomputed_transitions[counter][precomputed_transitions[counter] <= 1e-99] = 1e-99
        precomputed_log[counter] = np.log(precomputed_transitions[counter]).T
        previous_length = bl
        previous_mat = where_next
        counter += 1
    return precomputed_transitions, precomputed_log

def precalculate_transitions_new(branch_lengths, exponentiated):
    num_demes = exponentiated.shape[0]
    transitions = np.zeros((len(branch_lengths), num_demes, num_demes), dtype="float64")
    transitions[0] = np.linalg.matrix_power(exponentiated, branch_lengths[0])
    for i in range(1, len(branch_lengths)):
        transitions[i] = np.dot(
            transitions[i-1],
            np.linalg.matrix_power(exponentiated, branch_lengths[i] - branch_lengths[i-1])
        )
    return transitions


@njit()
def compute_matrix_powers_numba(A, max_power):
    n = A.shape[0]
    powers = np.empty((max_power, n, n), dtype=A.dtype)
    current = np.eye(n, dtype=A.dtype)

    for k in range(max_power):
        current = current @ A
        powers[k] = current

    return powers  # powers[k] == A^(k+1)



@njit()
def compute_selected_powers_numba(branch_lengths, exponentiated):
    max_power = branch_lengths[-1]
    n = exponentiated.shape[0]
    
    result = np.empty((len(branch_lengths), n, n), dtype=exponentiated.dtype)
    current = np.eye(n, dtype=exponentiated.dtype)
    
    power_idx = 0
    for k in range(1, max_power + 1):
        current = current @ exponentiated
        if k == branch_lengths[power_idx]:
            result[power_idx] = current
            power_idx += 1
            if power_idx >= len(branch_lengths):
                break

    return result  # result[i] == A^{powers_to_compute[i]}



@njit()
def precalculate_transitions_numba(branch_lengths, exponentiated):
    num_demes = exponentiated.shape[0]
    precomputed_transitions = np.zeros((len(branch_lengths), num_demes, num_demes), dtype="float64")
    for i in range(len(branch_lengths)):
        precomputed_transitions[i] = np.linalg.matrix_power(exponentiated, branch_lengths[i])
    return precomputed_transitions

rate = 1e-5
demes = pd.read_csv("datasets/one_sample_per_deme/demes.tsv", sep="\t")
samples = pd.read_csv("datasets/one_sample_per_deme/samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)
trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]).first() for ts in glob(f"datasets/one_sample_per_deme/m{rate}/rep0/trees/*")]

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

start = time.time()
transitions = precalculate_transitions(branch_lengths, transition_matrix)
print(time.time() - start)

start = time.time()
exponentiated = linalg.expm(transition_matrix)
transitions = precalculate_transitions_new(branch_lengths, exponentiated)
transitions[transitions <= 1e-99] = 1e-99
np.log(transitions)
print(time.time() - start)

start = time.time()
exponentiated = linalg.expm(transition_matrix)
transitions = precalculate_transitions_new(branch_lengths, exponentiated)
transitions[transitions <= 1e-99] = 1e-99
np.log(transitions)
print(time.time() - start)



exit()

start = time.time()

exponentiated = linalg.expm(transition_matrix)
precalculate_transitions_numba_new(
    branch_lengths=branch_lengths,
    exponentiated=exponentiated
)

print(time.time() - start)


start = time.time()

exponentiated = linalg.expm(transition_matrix)
output_0 = precalculate_transitions_numba_new(
    branch_lengths=branch_lengths,
    exponentiated=exponentiated
)

print(time.time() - start)


start = time.time()

output_1, num = precalculate_transitions(
    branch_lengths=branch_lengths,
    transition_matrix=transition_matrix
)

print(time.time() - start)

#print(output_0[0])
#print(output_1[0])