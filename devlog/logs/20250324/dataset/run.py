import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import pandas as pd
import tskit
from glob import glob


demes = pd.read_csv("demes.tsv", sep="\t")
world_map = tct.WorldMap(demes)

sample_locations = pd.read_csv("samples.tsv", sep="\t")
sample_location_vectors = world_map.build_sample_location_vectors(sample_locations=sample_locations)

trees = [tskit.load(ts).first() for ts in glob("trees/*")]

migration_rates = []
for m in range(8,-1,-1):
    #if m == 2:
    #    mult = range(1,10)
    #else:
    #    mult = [1]
    mult = [1]
    for i in mult:
        migration_rates.append(i*10**(-m))
with open("out_bug_corrected.tsv", "w") as outfile:
    outfile.write("m0\tm1\tloglikelihood\n")
    for m0 in migration_rates:
        for m1 in migration_rates:
            ll = tct.calc_migration_rate_log_likelihood(
                world_map=world_map,
                trees=trees,
                sample_location_vectors=sample_location_vectors,
                migration_rates={0:m0, 1:m1, 2:m0}
            )[0]
            print(m0, m1, ll)
            outfile.write(f"{m0}\t{m1}\t{ll}\n")