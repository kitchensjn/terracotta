from joblib import Parallel, delayed
import time
import numpy as np
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import pandas as pd
from scipy import linalg
import tskit
from glob import glob


def exponentiate_matrix(bl, exponentiated):
    where_next = np.linalg.matrix_power(exponentiated, bl)
    where_next[where_next <= 0] = 1e-99
    return where_next


demes = pd.read_csv(f"demes_elev_two_type.tsv", sep="\t")
demes["type"] = 0 #ignoring elevation type
samples = pd.read_csv(f"samples_elev_two_type.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)
trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000]).first() for ts in glob(f"trees/*")]

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


transition_matrix = world_map.build_transition_matrix(migration_rates={0:0.02})
exponentiated = linalg.expm(transition_matrix)
exponentiated[exponentiated < 0] = 0

start = time.time()
results = Parallel(n_jobs=-1)(delayed(exponentiate_matrix)(i, exponentiated=exponentiated) for i in branch_lengths)
print(time.time() - start)

start = time.time()
previous_length = None
previous_mat = None
precomputed_transitions = {}
for bl in branch_lengths:
    #if previous_length != None:
    #    diff = bl - previous_length
    #    where_next = np.dot(previous_mat, np.linalg.matrix_power(exponentiated, diff))
    #else:
    where_next = np.linalg.matrix_power(exponentiated, bl)
    precomputed_transitions[bl] = where_next
    precomputed_transitions[bl][precomputed_transitions[bl] <= 0] = 1e-99
    previous_length = bl
    previous_mat = where_next
print(time.time() - start)
exit()




start = time.time()
results = Parallel(n_jobs=-1)(delayed(square)(i) for i in range(10000))
print(time.time() - start)

start = time.time()
results = Parallel(n_jobs=-1)(delayed(square)(i) for i in range(10000))
print(time.time() - start)

start = time.time()
for i in range(10000):
    n = i * i
print(time.time() - start)

