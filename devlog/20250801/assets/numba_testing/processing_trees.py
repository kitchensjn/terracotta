import pandas as pd
import numpy as np
from glob import glob
import tskit
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
from scipy.optimize import minimize
import time

rate = 1e-5
demes = pd.read_csv("datasets/one_sample_per_deme/demes.tsv", sep="\t")
samples = pd.read_csv("datasets/one_sample_per_deme/samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)
trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]).first() for ts in glob(f"datasets/one_sample_per_deme/m{rate}/rep0/trees/*")]
#trees = [tskit.load(ts).simplify().first() for ts in glob(f"datasets/m{rate}/*")[:10]]




def convert_tree_to_tuple_list(tree):
    """Breaks a tskit.Tree into primitive lists of children, branch_lengths, and root IDs
    """
    all_children = []
    branch_above = []
    max_num_children = 0
    for node in tree.nodes(order="timeasc"):
        children = []
        for child in tree.children(node):
            children.append(child)
        num_children = len(children)
        if num_children > max_num_children:
            max_num_children = num_children
        branch_above.append(tree.branch_length(node))
        all_children.append(children)
    num_nodes = len(tree.postorder())
    final_children = np.zeros((num_nodes, max_num_children), dtype="int64")
    final_children += -1
    for i,node in enumerate(all_children):
        for j,child in enumerate(node):
            final_children[i,j] = child
    return final_children, np.array(branch_above), np.array(tree.roots)


from numba import njit

@njit()
def numba_array_obj_test(a):
    return a

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

start = time.time()
print(tct.calc_migration_rate_log_likelihood(
    migration_rates=mr,
    world_map=world_map,
    children=cl,
    branch_above=bal,
    roots=r,
    branch_lengths=branch_lengths
))
print(time.time() - start)