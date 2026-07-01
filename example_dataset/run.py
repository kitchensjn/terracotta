import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct


result, loglikelihood = tct.run(
    demes_path="dataset/demes.tsv",
    connections_path="dataset/connections.tsv",
    samples_path="dataset/samples.tsv",
    trees_dir_path="dataset/trees",
    output_file="output.tsv"
)

print(loglikelihood)