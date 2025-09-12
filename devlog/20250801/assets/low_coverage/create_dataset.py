import tskit
import msprime
import sys

ts = msprime.sim_ancestry(
    2,
    sequence_length=100,
    random_seed=1234
)

mts = msprime.sim_mutations(
    ts,
    rate=0.01,
    random_seed=5678
)

with open("example.vcf", "w") as vcf_file:
    mts.write_vcf(vcf_file)