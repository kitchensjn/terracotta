import pandas as pd
import numpy as np
from glob import glob
import tskit
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
from scipy.optimize import minimize

rate = 0.01
demes = pd.read_csv("datasets/one_sample_per_deme/demes.tsv", sep="\t")
samples = pd.read_csv("datasets/one_sample_per_deme/samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)
trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]).first() for ts in glob(f"datasets/one_sample_per_deme/m{rate}/*")]
#trees = [tskit.load(ts).simplify().first() for ts in glob(f"datasets/m{rate}/*")[:10]]

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

mr = np.array([0.01])

res = minimize(
    tct.calc_migration_rate_log_likelihood,
    mr,
    method="nelder-mead",
    bounds=[(0, 1)],
    args=(world_map, cl, bal, r, branch_lengths)
)
print(res)

#print(tct.calc_migration_rate_log_likelihood_old(
#    world_map=world_map,
#    trees=trees,
#    migration_rates=mr
#)[0])

#print(tct.calc_migration_rate_log_likelihood(
#    migration_rates=mr,
#    world_map=world_map,
#    children=cl,
#    branch_above=bal,
#    roots=r,
#    branch_lengths=branch_lengths
#))