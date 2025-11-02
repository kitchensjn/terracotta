import tskit
import pandas as pd
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
from glob import glob
import numpy as np


#result, log_likelihood = tct.run(
#    demes_path="dataset/demes.tsv",
#    samples_path="dataset/samples.tsv",
#    trees_dir_path="dataset/0.01_0.01_0.01_0.01",
#    asymmetric=True
#)

#print(result, log_likelihood)

demes = pd.read_csv("dataset/demes.tsv", sep="\t")
samples = pd.read_csv("dataset/samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples, asymmetric=True)
tree = tskit.load("dataset/0.01_0.01_0.01_0.01/0.trees").first()

locations = tct.locate_nodes_in_tree(
    tree=tree,
    world_map=world_map,
    migration_rates=[0.01, 0.0001, 0.0001, 0.01] #result
)

node = 0
while node < 200:
    world_map.draw(figsize=(10,10), location_vector=locations[node], title=f"{round(tree.time(node))} generations ago")
    node = tree.parent(node)







#locations = tct.locate(
#    demes_path="dataset/demes.tsv",
#    samples_path="dataset/samples.tsv",
#    tree_path="dataset/0.01_0.01_0.01_0.01/0.trees",
#    migration_rates=result,
#    asymmetric=True
#)