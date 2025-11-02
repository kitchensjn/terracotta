import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import tskit
import pandas as pd


result, log_likelihood = tct.run(
    demes_path="demes.tsv",
    samples_path="samples.tsv",
    trees_dir_path="trees",
    asymmetric=True
)

locations = tct.locate(
    demes_path="demes.tsv",
    samples_path="samples.tsv",
    tree_path="trees/0.trees",
    migration_rates=result,
    asymmetric=True
)

demes = pd.read_csv("demes.tsv", sep="\t")
samples = pd.read_csv("samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples, asymmetric=True)
tree = tskit.load("trees/0.trees").first()


node = 3
while node != -1:
    world_map.draw(figsize=(5,5), location_vector=locations[node], title=f"{round(tree.time(node))} generations ago")
    node = tree.parent(node)

