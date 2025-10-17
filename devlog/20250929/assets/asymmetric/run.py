import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import pandas as pd
import tskit
import numpy as np
import time
from glob import glob



#demes = pd.read_csv("datasets/uninhabitable/demes.tsv", sep="\t")
#samples = pd.read_csv("datasets/uninhabitable/samples.tsv", sep="\t")

#world_map = tct.WorldMap(demes, samples)

#world_map.build_transition_matrices(migration_rates=[1,2,3,4])


res = tct.run(
    demes_path="demes.tsv",
    samples_path="samples.tsv",
    trees_dir_path="trees",
    time_bins=[0]+[10**i for i in range(0,10)],
    asymmetric=False
)

print(res)
