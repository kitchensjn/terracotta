import sys
import sim_utils
import pandas as pd
import math
from os import mkdir
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct



num_rings = 3
num_demes = sim_utils.get_number_of_demes(num_rings)

output_directory = "dataset"
mkdir(output_directory)

sim_utils.create_hexagonal_tri_grid(
    num_rings,
    output_directory=output_directory
)

demes = pd.read_csv(f"{output_directory}/demes.tsv", sep="\t")
demes["suitability"] = demes["suitability"].astype(str)
for i,deme in demes.iterrows():
    distance_from_center = math.sqrt(deme["xcoord"]**2 + deme["ycoord"]**2)
    demes.loc[i, "suitability"] = str(max(0.001, 1 - distance_from_center/5))
demes.to_csv(f"{output_directory}/demes.tsv", sep="\t", index=False)

sim_utils.create_random_samples_file(
    demes_path=f"{output_directory}/demes.tsv",
    number_of_samples=50,
    allow_multiple_samples_per_deme=True,
    output_path=f"{output_directory}/samples.tsv"
)

sim_utils.create_trees_files(
    demes_path=f"{output_directory}/demes.tsv",
    connections_path=f"{output_directory}/connections.tsv",
    samples_path=f"{output_directory}/samples.tsv",
    number_of_trees=100,
    pop_size=10,
    migration_rate=0.01,
    output_directory="dataset"
)