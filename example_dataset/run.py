import pandas as pd
import sys
sys.path.append("..")
import terracotta as tct
import tskit
from glob import glob
import numpy as np
from scipy.optimize import minimize


demes = pd.read_csv("demes.tsv", sep="\t")
samples = pd.read_csv("samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)
trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0]+[10**i for i in range(1,10)]).first() for ts in glob(f"trees/*")]

world_map.draw(
    figsize=(15,15),
    color_demes=True,
    show_samples=True
)

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

mr = np.array([2.5e-03, 4e-02, 0])

print(
    minimize(
        tct.calc_migration_rate_log_likelihood,
        mr,
        method="nelder-mead",
        bounds=[(0, 1), (0, 1), (0, 1)],
        args=(world_map, cl, bal, r, branch_lengths)
    )
)
exit()

positions = tct.track_lineage_over_time(
    sample=1500,
    times=range(0,11000,1000),
    tree=trees[0],
    world_map=world_map,
    migration_rates=mr
)

for time in range(0, 11000, 1000):
    world_map.draw_estimated_location(
        location_vector=positions[time],
        figsize=(15,15),
        title=f"{time} generations in past"
    )