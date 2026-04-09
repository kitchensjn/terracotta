import tskit
import numpy as np
from os import mkdir
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import pandas as pd


output_file = "output.tsv"

output = tct.run(
    demes_path="dataset/demes.tsv",
    connections_path="dataset/connections.tsv",
    samples_path="dataset/samples.tsv",
    trees_dir_path="dataset/trees",
    output_file=output_file,
    verbose=True
)

with open(output_file, "a") as outfile:
    outfile.write(f"\n{output[0]}\t{output[1]}\n")