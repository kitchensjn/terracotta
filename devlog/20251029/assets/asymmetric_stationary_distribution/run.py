import tskit
import pandas as pd
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
from glob import glob
import numpy as np


#tct.run_for_single(
#    migration_rates=np.array([0.01, 0.01, 0.00, 0.01]),
#    demes_path="dataset/demes.tsv",
#    samples_path="dataset/samples.tsv",
#    trees_dir_path="dataset/trees",
#    #time_bins=[0]+[10**i for i in range(0,10)],
#    asymmetric=True
#)

#exit()
print(tct.run(
    demes_path="dataset/demes.tsv",
    samples_path="dataset/samples.tsv",
    trees_dir_path="dataset/0.01_0.001_0.0001_0.00001",
    time_bins=[0]+[10**i for i in range(0,10)],
    asymmetric=True
))