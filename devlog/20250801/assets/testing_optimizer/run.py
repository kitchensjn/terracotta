import pandas as pd
import numpy as np
from glob import glob
import tskit
import sys
sys.path.append("..")
import terracotta as tct
import importlib
importlib.reload(tct)
from scipy.optimize import minimize, shgo
import time


demes = pd.read_csv("dataset_1/demes.tsv", sep="\t")
samples = pd.read_csv("dataset_1/samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)

migration_rate_string = "0.01_0.01_0.10"
rep = 0
#trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000]).first() for ts in glob(f"dataset/trees/{migration_rate_string}/rep{rep}/*")]
trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000]).first() for ts in glob(f"dataset_1/trees/*")]

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


for i in range(-6, 2):
    for j in range(-7, 2):
        for k in range(-7, 2):
            for l in range(-7, 2):
                tct.calc_log_migration_rate_log_likelihood(
                    log_migration_rates=np.array([i,j,k,l]),
                    world_map=world_map,
                    children=cl,
                    branch_above=bal,
                    roots=r,
                    branch_lengths=branch_lengths
                )

exit()

res = shgo(
    tct.calc_log_migration_rate_log_likelihood,
    bounds=[(-20, 7), (-20, 7), (-20, 7)],
    n=100,
    iters=5,
    sampling_method="sobol",
    args=(world_map, cl, bal, r, branch_lengths)
)

print(res.xl)