import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import tskit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


demes = pd.read_csv("dataset/demes.tsv", sep="\t")
samples = pd.read_csv("dataset/hc_samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)

tree = tskit.load("dataset/trees/0.trees").simplify()

tree = tree.first()

sample = 7

location = tct.track_lineage_over_time(
    sample=sample,
    times=range(0,21000,1000),
    tree=tree,
    world_map=world_map,
    migration_rates=[0.00617594]
)

for t in range(0,21000,1000):
    print(t)
    #print(location[t])
    world_map.draw(figsize=(10,10), location_vector=location[t], show=False)

    xs = []
    ys = []
    mrcas = []

    for _,d in world_map.samples.iterrows():
        coords = world_map.get_coordinates_of_deme(d["deme"])
        xs.append(coords[0])
        ys.append(coords[1])
        mrcas.append(tree.mrca(sample, d["id"]))
    
    plt.scatter(xs, ys, c=np.log(mrcas), marker="x", s=100, zorder=2)

    plt.show()

    