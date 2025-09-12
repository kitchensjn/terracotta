import tskit
import pandas as pd
import io
import msprime
import numpy as np
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import math
import time


def read_vcf(path):
    """Read a vcf into a pandas.DataFrame
    From https://stackoverflow.com/questions/70219758/vcf-data-to-pandas-dataframe
    """

    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})

def fetch_read(chrom, pos, sample, strand, vcf, genotype_split_character=None):
    """
    """

    row = vcf[(vcf["CHROM"]==str(chrom)) & (vcf["POS"]==pos)]
    sample_genotype = row[sample].iloc[0]
    if genotype_split_character == None:
        if "|" in sample_genotype:
            genotype_split_character = "|"
        elif "/" in sample_genotype:
            genotype_split_character = "/"
        else:
            raise RuntimeError("Unknown genotype split character. Please provide as `genotype_split_character`.")
    read = sample_genotype.split(genotype_split_character)[strand]
    return read




demes = pd.read_csv("dataset/demes.tsv", sep="\t")
samples = pd.read_csv("dataset/samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)

transition_matrix = world_map.build_transition_matrix(migration_rates=np.array([1e-5]))
branch_lengths = np.array([100])
precomputed_transitions = tct.precalculate_transitions(branch_lengths, transition_matrix) # only need the one array
precomputed_log = np.log(precomputed_transitions)

sample_locations_array, sample_ids = world_map._build_sample_locations_array()

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


sites = mts.tables.sites
muts = mts.tables.mutations

vcf = read_vcf("example.vcf")

def calc_overlap(lineage0, lineage1):
    combined = np.vstack((lineage0,lineage1))
    summed_log_messages = np.sum(combined, axis=0)
    c = np.max(summed_log_messages)
    log_sum_exp = c + np.log(np.sum(np.exp(summed_log_messages - c)))
    return log_sum_exp

def propagate_back(lineage, log_trans):
    combined = log_trans[0] + lineage
    c = np.max(combined)
    lineage = c + np.log(np.sum(np.exp(combined - c), axis=1))
    return lineage


sample0_orig_id = "tsk_0"
sample1_orig_id = "tsk_1"

for _,row in vcf.iterrows():
    for i in range(2):
        for j in range(2):
            sample0 = fetch_read(row["CHROM"], row["POS"], sample0_orig_id, i, vcf)
            sample1 = fetch_read(row["CHROM"], row["POS"], sample1_orig_id, j, vcf)

            if (sample0 == sample1) and (sample0 == "1"):

                site_id = np.where(sites.position==row["POS"])[0][0]
                mutation_times = muts[np.where(muts.site == site_id)[0]].time
                if len(mutation_times) > 1:
                    raise RuntimeError(f"Site {row["POS"]} has multiple mutations in the tree sequence.")
                mut_time = mutation_times[0]

                id0 = world_map.samples.id[np.where(world_map.samples.orig_id==sample0_orig_id)[0][0]]
                sla0 = sample_locations_array[np.where(sample_ids==id0)[0][0]]
                id1 = world_map.samples.id[np.where(world_map.samples.orig_id==sample1_orig_id)[0][0]]
                sla1 = sample_locations_array[np.where(sample_ids==id1)[0][0]]

                how_many_props = math.ceil(mut_time / branch_lengths[0])

                sla0 = np.log(np.dot(sla0, precomputed_transitions[0]))
                sla1 = np.log(np.dot(sla1, precomputed_transitions[0]))

                outputs = [calc_overlap(lineage0=sla0, lineage1=sla1)]
                
                for t in range(how_many_props-1):
                    sla0 = propagate_back(sla0, precomputed_log)
                    sla1 = propagate_back(sla1, precomputed_log)
                    outputs.append(calc_overlap(lineage0=sla0, lineage1=sla1))

                outputs = np.array(outputs)
                c = np.max(outputs)
                final = c + np.log(np.sum(np.exp(outputs - c)))
                print(row["POS"], i, j, mut_time, final)
        
    
    
    
    