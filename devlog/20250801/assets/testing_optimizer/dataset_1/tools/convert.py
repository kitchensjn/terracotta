import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)


tct.create_demes_file_from_world_builder(world_builder_path="terracotta_map_builder.csv")