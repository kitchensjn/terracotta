import pandas as pd

samples = pd.read_csv("dataset/samples.tsv", sep="\t")
filtered = samples.drop_duplicates("deme").reset_index(drop=True)
filtered["id"] = filtered.index
filtered.to_csv("dataset/samples_small.tsv", index=False, sep="\t")