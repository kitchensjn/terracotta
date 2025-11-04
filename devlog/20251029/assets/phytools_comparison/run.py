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
    asymmetric=True,
    output_file="results.txt"
)





#locations = tct.locate(
#    demes_path="demes.tsv",
#    samples_path="samples.tsv",
#    tree_path="trees/0.trees",
#    migration_rates=result,
#    asymmetric=True
#)

demes = pd.read_csv("demes.tsv", sep="\t")
samples = pd.read_csv("samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples, asymmetric=True)
tree = tskit.load("trees/0.trees").first()

locations = tct.track_lineage_in_tree(
    node=3,
    times=range(0,100,10),
    tree=tree,
    world_map=world_map,
    migration_rates=result
)

for i,time in enumerate(range(0, 100, 10)):
    world_map.draw(figsize=(5,5), location_vector=locations[i], title=f"{time} generations ago")

