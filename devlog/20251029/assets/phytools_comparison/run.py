import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import tskit
import pandas as pd


demes = pd.DataFrame({"id":[0,1], "xcoord":[0,1], "ycoord":[0,1], "suitability":[1,1]})
connections = pd.DataFrame({"id":[0,1], "deme_0":[0,1], "deme_1":[1,0], "migration_modifier":["a","b"]})
demes.to_csv("demes.tsv", sep="\t", index=False)
connections.to_csv("connections.tsv", sep="\t", index=False)

print(tct.run_for_parameters(
    parameters=[1, 0.015609, 0.017491],
    demes_path="demes.tsv",
    connections_path="connections.tsv",
    samples_path="samples.tsv",
    trees_dir_path="trees"
))

exit()




result, log_likelihood = tct.run(
    demes_path="demes.tsv",
    connections_path="connections.tsv",
    samples_path="samples.tsv",
    trees_dir_path="trees",
    output_file="results.txt"
)

print(result, log_likelihood)

exit()





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

