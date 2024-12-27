import terracotta as tct
import pandas as pd
import random
import matplotlib.pyplot as plt
from glob import glob
import tskit
import time
import networkx as nx


simulation_setup = "datasets/effect_of_population_size/g10_s25_p1_t100_n"
dataset_migration_rates = [10**(-m) for m in range(1,8)]
dataset_population_size = [5*(10**n) for n in range(0,4)]

for dmr in dataset_migration_rates:
    for dps in dataset_population_size: 
        directory = f"{simulation_setup}{dps}_m{dmr}"
        print(directory)

        world_map = pd.read_csv(f"{directory}/world_map.tsv", sep="\t")

        tct.draw_map(world_map)
        exit()

        state_space = tct.build_state_space_from_map(world_map)

        samples_df = pd.read_csv(f"{directory}/samples.tsv", sep="\t")
        sample_loc_probs = tct.convert_sample_locations(
            samples=samples_df,
            state_space=state_space
        )

        trees = [tskit.load(ts).first() for ts in glob(f"{directory}/trees/*")]

        start = time.perf_counter()

        migration_rates = [i*10**(-m) for m in range(8,0,-1) for i in range(1,10,4)]
        with open(f"{directory}/loglikelihoods_chopped_5000.tsv","w") as outfile:
            outfile.write("migration_rate\tloglikelihood\n")
            for mr in migration_rates:
                transition_matrix = tct.build_transition_matrix(state_space=state_space, migration_rate=mr)
                log_likelihoods = []
                for tree in trees:
                    log_likelihoods.append(tct.calc_tree_log_likelihood(
                        tree=tree,
                        sample_locs=sample_loc_probs,
                        transition_matrix=transition_matrix
                    )[0])
                summed = sum(log_likelihoods)
                print(mr, summed)
                outfile.write(f"{mr}\t{summed}\n")

        finish = time.perf_counter()

        print(f"Finished in {round(finish-start, 2)} seconds(s)")