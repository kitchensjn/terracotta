import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct


tct.run_for_rate_combo(
    migration_rates=[0.01, 0.0001, 0.01, 0.01],
    demes_path="datasets/0.01_0.0001_0.01_0.01/demes.tsv",
    samples_path="datasets/0.01_0.0001_0.01_0.01/samples.tsv",
    trees_dir_path="datasets/0.01_0.0001_0.01_0.01/trees"
)