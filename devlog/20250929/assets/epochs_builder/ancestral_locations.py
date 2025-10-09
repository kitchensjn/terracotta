import tskit
import pandas as pd
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
from glob import glob



tss = [tskit.load(tree_file) for tree_file in glob("datasets/asymmetric/trees/*")]

for ts in tss:
    print(ts.node(ts.first().root).time, ts.population(ts.node(ts.first().root).population).metadata["name"].split("_")[-1])

exit()



sample = 0
genome_position = 0

tree = ts.at(genome_position)
ancestors = [sample] + list(tct.ancs(tree=tree, u=sample))

true_positions = {}
a = sample
while a != -1:
    true_positions[ts.node(a).time] = int(ts.population(ts.node(a).population).metadata["name"].split("_")[-1])
    a = tree.parent(a)

df = pd.DataFrame({"time":true_positions.keys(), "deme":true_positions.values()})
print(df)