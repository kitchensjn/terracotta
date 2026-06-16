import tskit
import pandas as pd
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)


hc_samples = list(range(20))
lc_samples = list(range(20,100))

ts = tskit.load("dataset/arg.trees")

hc_ts, hc_map_nodes = ts.simplify(samples=hc_samples, map_nodes=True)
hc_ts.dump("dataset/hc_arg.trees")

tct.create_trees_folder_from_ts(
    ts=hc_ts,
    output_path="dataset",
    gap_between_trees=50
)


lc_ts, lc_map_nodes = ts.simplify(samples=lc_samples, map_nodes=True)

with open("dataset/lc_arg.vcf", "w") as vcf_file:
    lc_ts.write_vcf(vcf_file)

samples = pd.read_csv("dataset/samples.tsv", sep="\t")
samples.loc[hc_samples].to_csv("dataset/hc_samples.tsv", sep="\t", index=False)

low_samples = samples.loc[lc_samples].reset_index()
low_samples["arg_id"] = low_samples["id"]
low_samples["id"] = low_samples.index
low_samples["vcf_id"] = "tsk_" + low_samples["id"].astype(str)
low_samples = low_samples[["id", "deme", "vcf_id", "arg_id"]]
low_samples.to_csv("dataset/lc_samples.tsv", sep="\t", index=False)