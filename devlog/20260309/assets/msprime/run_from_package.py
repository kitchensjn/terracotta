import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct

mle, likelihood = tct.run(
    demes_path="dataset/demes.tsv",
    connections_path="dataset/connections.tsv",
    samples_path="dataset/samples.tsv",
    trees_dir_path="dataset/trees",
    output_file="dataset/output.tsv",
    verbose=True
)

print(mle)