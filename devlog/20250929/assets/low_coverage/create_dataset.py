import tskit
import msprime
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)


tct.create_random_grid_demes_file(
    side_length=20,
    number_of_deme_types=1,
    output_path="dataset/demes.tsv"
)

tct.create_random_samples_file(
    demes_path="dataset/demes.tsv",
    number_of_samples=100,
    allow_multiple_samples_per_deme=True,
    output_path="dataset/samples.tsv"
)

tct.create_arg_file(
    demes_path="dataset/demes.tsv",
    samples_path="dataset/samples.tsv",
    sequence_length=5000000,
    recombination_rate=1e-8,
    pop_size=50,
    migration_rate=0.01,
    mutation_rate=1e-8,
    output_path="dataset/arg.trees"
)

#ts = msprime.sim_ancestry(
#    2,
#    sequence_length=100,
#    random_seed=1234
#)

#mts = msprime.sim_mutations(
#    ts,
#    rate=0.01,
#    random_seed=5678
#)

#with open("example.vcf", "w") as vcf_file:
#    mts.write_vcf(vcf_file)