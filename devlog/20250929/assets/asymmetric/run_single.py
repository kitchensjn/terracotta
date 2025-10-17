import tskit
import pandas as pd
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
from glob import glob
import numpy as np


demes_path="demes.tsv"
samples_path="samples.tsv"
trees_dir_path="symmetric"
time_bins = None

demes = pd.read_csv(demes_path, sep="\t")
samples = pd.read_csv(samples_path, sep="\t")
world_map = tct.WorldMap(demes, samples, asymmetric=True)

if trees_dir_path[-1] != "/":
    trees_dir_path += "/"

trees = []
for ts in glob(trees_dir_path+"*"):
    tree = tskit.load(ts).simplify()
    if time_bins is not None:
        tree = tct.nx_bin_ts(tree, time_bins)
    trees.append(tree.first())

def _deconstruct_trees(trees, epochs):
    """

    Note: It would be great if pl and bal were numpy.ndarray, but that would force
    the trees to have the same number of nodes, which is unrealistic.
    """
    
    pl = []
    bal = []
    roots = []
    all_branch_lengths = [[] for e in epochs]
    for tree in trees:
        num_nodes = len(tree.postorder())
        parents = np.full(num_nodes, -1, dtype="int64")
        branch_above = np.zeros((len(epochs), num_nodes), dtype="int64")
        for node in tree.nodes(order="timeasc"):
            parent = tree.parent(node)
            if parent != -1:
                node_time = tree.time(node)
                parent_time = tree.time(parent)
                starting_epoch = np.digitize(node_time, epochs)-1
                ending_epoch = np.digitize(parent_time, epochs)-1
                if starting_epoch == ending_epoch:
                    branch_above[starting_epoch, node] = parent_time - node_time
                else:
                    branch_above[starting_epoch, node] = epochs[starting_epoch+1] - node_time
                    for e in range(starting_epoch+1, ending_epoch):
                        branch_above[e, node] = epochs[e+1] - epochs[e]
                    branch_above[ending_epoch, node] = parent_time - epochs[ending_epoch]
            parents[node] = parent
        pl.append(parents)
        bal.append(branch_above)
        roots.append(np.where(parents==-1)[0])
        for e in range(len(epochs)):
            all_branch_lengths[e].extend(branch_above[e])
    unique_branch_lengths = []
    for e in range(len(epochs)):
        unique_branch_lengths.append(np.unique(all_branch_lengths[e]))
    return pl, bal, roots, unique_branch_lengths

pl, bal, r, ubl = _deconstruct_trees(trees=trees, epochs=world_map.epochs)  # needed to use numba

print(tct.calc_migration_rate_log_likelihood(
    np.array([0.01, 0.02, 0.01, 0.01]),
    world_map,
    pl,
    bal,
    r,
    ubl
))