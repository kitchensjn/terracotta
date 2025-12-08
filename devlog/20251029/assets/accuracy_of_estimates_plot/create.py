import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import pandas as pd


rate = 0.00001

tct.create_random_grid_demes_file(5, number_of_deme_types=1, output_path=f"datasets/{rate}/demes.tsv")

tct.create_random_samples_file(
    f"datasets/{rate}/demes.tsv",
    25,
    allow_multiple_samples_per_deme=False,
    output_path=f"datasets/{rate}/samples.tsv"
)

tct.create_trees_files(
    demes_path=f"datasets/{rate}/demes.tsv",
    samples_path=f"datasets/{rate}/samples.tsv",
    number_of_trees=100,
    pop_size=50,
    asymmetric=True,
    migration_rate=rate,
    output_directory=f"datasets/{rate}"
)