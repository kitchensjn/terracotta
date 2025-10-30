import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)


print(tct.run_for_single(
    migration_rates=[0.015609, 0.017491],
    demes_path="demes.tsv",
    samples_path="samples.tsv",
    trees_dir_path="trees",
    asymmetric=True
))