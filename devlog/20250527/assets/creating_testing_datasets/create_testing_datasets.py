import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import pandas as pd


#samples = pd.read_csv("datasets/all_samples/samples.tsv", sep="\t")
#samples = samples.drop_duplicates(subset=["deme"])
#samples["id"] = range(0, len(samples))
#samples.to_csv("datasets/one_sample_per_deme/samples.tsv", sep="\t")


tct.create_trees_files(
    "datasets/diploid_one_sample_per_deme/demes.tsv",
    "datasets/diploid_one_sample_per_deme/samples.tsv",
    number_of_trees=1,
    record_provenance=True,
    pop_size=500,
    ploidy=2,
    migration_rate=1e-05,
    output_directory="datasets/diploid_one_sample_per_deme/m1e-05/"
)