import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import pandas as pd


#tct.create_random_grid_demes_file(5, number_of_deme_types=2, output_path="dataset/sim/demes.tsv")
#tct.create_random_samples_file(
#    "dataset/sim/demes.tsv",
#    9,
#    allow_multiple_samples_per_deme=False,
#    output_path="dataset/sim/samples.tsv"
#)

demes = pd.read_csv("dataset/sim/demes.tsv", sep="\t")
samples = pd.read_csv("dataset/sim/samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)

#world_map.draw(figsize=(7,7), color_connections=True, color_demes=True)

tct.create_trees_files(
    demes_path="dataset/sim/demes.tsv",
    samples_path="dataset/sim/samples.tsv",
    number_of_trees=1,
    pop_size=50,
    migration_rates={
        0:0.001,
        1:0.010,
        2:0.100
    },
    output_directory="dataset/sim"
)