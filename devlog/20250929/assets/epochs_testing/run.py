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


res = tct.run(
    demes_path="dataset/sim/demes.tsv",
    samples_path="dataset/sim/samples.tsv",
    trees_dir_path="dataset/sim/trees"
)

print(res)
