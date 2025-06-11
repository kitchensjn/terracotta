import time
import numpy as np
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import pandas as pd
from scipy import linalg
import tskit
from glob import glob



demes = pd.read_csv(f"demes_elev_two_type.tsv", sep="\t")
demes["type"] = 0 #ignoring elevation type
samples = pd.read_csv(f"samples_elev_two_type.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)
#trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000]).first() for ts in glob(f"trees/*")[:1]]
trees = [tskit.load(ts).simplify().first() for ts in glob(f"trees/*")[:1]]

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

start = time.time()
trans, log = tct.precalculate_transitions(
    branch_lengths=branch_lengths,
    exponentiated_matrix=exponentiated
)
print(time.time() - start)

start = time.time()
trans_slow, log_slow = tct.precalculate_transitions(
    branch_lengths=branch_lengths,
    exponentiated_matrix=exponentiated,
    fast=False
)
print(time.time() - start)

percent_error = np.divide(trans-trans_slow, trans_slow)
print(np.sum(percent_error >= 1e-10))