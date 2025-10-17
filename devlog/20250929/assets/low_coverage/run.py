import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)


res = tct.run(
    demes_path="dataset/demes.tsv",
    samples_path="dataset/hc_samples.tsv",
    trees_dir_path="dataset/trees",
    time_bins=[0]+[10**i for i in range(0,10)],
    asymmetric=False
)

print(res)