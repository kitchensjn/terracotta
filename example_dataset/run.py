import pandas as pd
import sys
sys.path.append("..")
import terracotta as tct
import tskit
from glob import glob
import numpy as np
from scipy.optimize import minimize


demes = pd.read_csv("demes.tsv", sep="\t")
samples = pd.read_csv("samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)
trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0]+[10**i for i in range(1,10)]).first() for ts in glob(f"trees/*")]

world_map.draw(
    figsize=(15,15),
    color_demes=True,
    show_samples=True
)


### This is currently slow with the example dataset. Likely to change

result, log_likelihood = tct.run(
    demes_path="demes.tsv",
    samples_path="samples.tsv",
    trees_dir_path="trees",
    asymmetric=True,
    output_file="results.txt"
)