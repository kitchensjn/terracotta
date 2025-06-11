import pandas as pd

demes = pd.read_csv("../../20250501/assets/better_mcmc/demes.tsv", sep="\t")
demes.loc[demes["type"] > 1, "type"] = 1
demes.to_csv("datasets/alternate_maps/high_vs_low_elevation/demes.tsv", sep="\t", index=False)