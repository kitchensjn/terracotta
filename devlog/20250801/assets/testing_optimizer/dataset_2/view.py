import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import pandas as pd
import numpy as np


demes = pd.read_csv("demes.tsv", sep="\t")
samples = pd.read_csv("samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)

print(np.log(0.001))

exit()
world_map.draw(
    figsize=(15,15),
    #migration_rates={0:np.log(0.001),1:np.log(0.010),2:np.log(0.010),3:np.log(0.100)},
    color_connections=True,
    save_to="true_map.svg"
)

#world_map.draw(
#    figsize=(15,15),
#    migration_rates={0:np.log(0.001)-(-7.42078725), 1:np.log(0.010)-(-5.35346175), 2:np.log(0.010)-(-5.53996652), 3:np.log(0.100)-(-2.98229226)},
#    save_to="estimated_map.svg"
#)