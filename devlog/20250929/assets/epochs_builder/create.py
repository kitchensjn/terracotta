import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import pandas as pd


#tct.create_random_grid_demes_file(5, number_of_deme_types=1, output_path="datasets/base.tsv")

#tct.create_random_samples_file(
#    "datasets/base/demes.tsv",
#    25,
#    allow_multiple_samples_per_deme=False,
#    output_path="datasets/base/samples.tsv"
#)

tct.create_trees_files(
    demes_path="datasets/asymmetric/demes.tsv",
    samples_path="datasets/asymmetric/samples.tsv",
    number_of_trees=1000,
    pop_size=50,
    migration_rates={
        0:0.01,
        1:0.00,
        2:0.01,
        3:0.01
    },
    output_directory="datasets/asymmetric"
)