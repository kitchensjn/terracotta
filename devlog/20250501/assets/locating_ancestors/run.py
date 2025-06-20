import pandas as pd
import numpy as np
from glob import glob
import tskit
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import matplotlib.pyplot as plt


directory = "."

demes = pd.read_csv(f"{directory}/demes.tsv", sep="\t")

demes["type"] = 0 #ignoring elevation type

samples = pd.read_csv(f"{directory}/samples.tsv", sep="\t")

world_map = tct.WorldMap(demes, samples)

trees = [tskit.load(ts).first() for ts in glob(f"{directory}/trees/*")]

locations = tct.locate_nodes_in_tree(
    world_map=world_map,
    tree=trees[0],
    migration_rates={0:1.2733254244553078e-06}
)

def ancestors(tree, u):
    """Find all of the ancestors above a node for a tree

    Taken directly from https://github.com/tskit-dev/tskit/issues/2706

    Parameters
    ----------
    tree : tskit.Tree
    u : int
        The ID for the node of interest

    Returns
    -------
    An iterator over the ancestors of u in this tree
    """

    u = tree.parent(u)
    while u != -1:
         yield u
         u = tree.parent(u)


sample = 8
lineage = ancestors(trees[0], sample)
world_map.draw_estimated_location(location_vector=locations[sample], figsize=(15,15))
for ancestor in lineage:
    world_map.draw_estimated_location(location_vector=locations[ancestor], figsize=(15,15), show_samples=list(trees[0].leaves(ancestor)))



