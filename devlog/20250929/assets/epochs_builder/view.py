import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import pandas as pd


demes = pd.read_csv("datasets/asymmetric/demes.tsv", sep="\t")

world_map = tct.WorldMap(demes)

world_map.draw((7,7), color_demes=True, color_connections=True)