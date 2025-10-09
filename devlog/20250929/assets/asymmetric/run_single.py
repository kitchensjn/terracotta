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
trees_dir_path="asymmetric"
time_bins = None

demes = pd.read_csv(demes_path, sep="\t")
samples = pd.read_csv(samples_path, sep="\t")
world_map = tct.WorldMap(demes, samples)

if trees_dir_path[-1] != "/":
    trees_dir_path += "/"

trees = []
for ts in glob(trees_dir_path+"*"):
    tree = tskit.load(ts).simplify()
    if time_bins is not None:
        tree = tct.nx_bin_ts(tree, time_bins)
    trees.append(tree.first())
pl, bal, r, ubl = tct.deconstruct_trees(trees=trees, epochs=world_map.epochs)  # needed to use numba

print(tct.calc_migration_rate_log_likelihood(
    np.array([0.01, 0.00, 0.01, 0.01]),
    world_map,
    pl,
    bal,
    r,
    ubl
))