import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import pandas as pd
import numpy as np
import math

map_builder = pd.read_csv("two_type_mid_boundary.csv", header=None, names=["i", "x", "y", "c"])
side_length = int(math.sqrt(len(map_builder)))
max_x = map_builder["x"].max()
max_y = map_builder["y"].max()
included_demes = map_builder.loc[map_builder["c"] != "lightgrey"]
ids = included_demes["i"].values
colors = np.unique(included_demes["c"].values)

with open("demes.tsv", "w") as outfile:
    outfile.write("\t".join(["id", "xcoord", "ycoord", "type", "neighbours"]) + "\n")
    for _, deme in included_demes.iterrows():
        x = int(deme["x"]/side_length)
        y = int(deme["y"]/side_length)
        neighbors = []
        if (deme["y"] > 0):
            n = int(x+(y-1)*side_length)
            if n in ids:
                neighbors.append(str(n))
        if (deme["x"] > 0):
            n = int((x-1)+y*side_length)
            if n in ids:
                neighbors.append(str(n))
        if (deme["x"] < max_x):
            n = int((x+1)+y*side_length)
            if n in ids:
                neighbors.append(str(n))
        if (deme["y"] < max_y):
            n = int(x+(y+1)*side_length)
            if n in ids:
                neighbors.append(str(n))
        outfile.write(f"{x+y*side_length}\t{x}\t{y}\t{np.argmax(colors == deme["c"])}\t{",".join(neighbors)}\n")


tct.create_samples_and_trees_files(
    demes_path="demes.tsv",
    number_of_samples=25,
    allow_multiple_samples_per_deme=False,
    number_of_trees=100,
    ploidy=1,
    pop_size=500,
    migration_rates={0:0.05, 1:0.002, 2:0.05}
)