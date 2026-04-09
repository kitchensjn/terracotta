import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import numpy as np


#print(tct.run_for_parameters(
#    parameters=[
#        np.log(1),
#        np.log(306.1803),
#        np.log(265.4571),
#        np.log(160.7738),
#        np.log(273.2361),
#        np.log(330.4791),
#        np.log(344.8335)
#    ],
#    demes_path="data/demes.tsv",
#    connections_path="data/connections.tsv",
#    samples_path="data/samples.tsv",
#    trees_dir_path="data/trees"
#))

#exit()

dataset = "single"

#print(tct.run_for_parameters(
#    parameters=[
#        np.log(1),
#        np.log(1),
#        np.log(2)
#    ],
#    demes_path=f"{dataset}/demes.tsv",
#    connections_path=f"{dataset}/connections.tsv",
#    samples_path=f"{dataset}/samples.tsv",
#    trees_dir_path=f"{dataset}/trees"
#))

#exit()



result, log_likelihood = tct.run(
    demes_path=f"{dataset}/demes.tsv",
    connections_path=f"{dataset}/connections.tsv",
    samples_path=f"{dataset}/samples.tsv",
    trees_dir_path=f"{dataset}/trees",
    output_file="results.txt",
    num_walkers=1000,
    num_iters=1,
)

print(result, log_likelihood)