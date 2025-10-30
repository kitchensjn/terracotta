import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import pandas as pd


#tct.create_random_grid_demes_file(5, number_of_deme_types=1, output_path="dataset/demes.tsv")

#tct.create_random_samples_file(
#    "dataset/demes.tsv",
#    25,
#    allow_multiple_samples_per_deme=False,
#    output_path="dataset/samples.tsv"
#)

tct.create_trees_files(
    demes_path="dataset/demes.tsv",
    samples_path="dataset/samples.tsv",
    number_of_trees=1000,
    pop_size=50,
    asymmetric=False,
    migration_rates={
        0:0.01,
        1:0.01,
        2:0.01
    },
    output_directory="dataset"
)

