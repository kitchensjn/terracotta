import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import pandas as pd



demes = pd.read_csv("demes.tsv", sep="\t")
samples = pd.read_csv("samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)


tct.create_trees_files(
    demes_path="demes.tsv",
    samples_path="samples.tsv",
    number_of_trees=100,
    pop_size=10,
    migration_rates={
        0:0.01,
        1:0.01,
        2:0.10
    },
    asymmetric=False,
    output_directory="."
)

exit()

tct.create_trees_files(
    demes_path="demes.tsv",
    samples_path="samples.tsv",
    number_of_trees=1000,
    pop_size=10,
    migration_rates={
        0:0.01,
        1:0.00,
        2:0.01,
        3:0.01
    },
    output_directory="."
)