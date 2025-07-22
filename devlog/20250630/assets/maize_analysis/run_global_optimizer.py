import pandas as pd
import numpy as np
from glob import glob
import tskit
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
from scipy.optimize import shgo


demes = pd.read_csv("demes.tsv", sep="\t")
samples = pd.read_csv("samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)

world_map.draw(figsize=(10, 10), color_connections=True, show_samples=True)

trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000]).first() for ts in glob(f"trees/*")]

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

res = shgo(
    tct.calc_migration_rate_log_likelihood,
    [(0, 1), (0, 1)],
    args=(world_map, cl, bal, r, branch_lengths)
)
print(res)