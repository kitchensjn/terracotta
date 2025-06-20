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
trees = [tskit.load(ts).simplify().first() for ts in glob(f"trees/*")]

world_map.draw(
    figsize=(15,15),
    color_demes=True,
    show_samples=True
)

mr = np.array([4.32e-03, 4.95e-02, 0])

positions = tct.track_lineage_over_time(
    sample=1500,
    times=range(0,11000,1000),
    tree=trees[0],
    world_map=world_map,
    migration_rates=mr
)

for time in range(0, 11000, 1000):
    world_map.draw_estimated_location(
        location_vector=positions[time],
        figsize=(15,15),
        title=f"{time} generations in past"
    )