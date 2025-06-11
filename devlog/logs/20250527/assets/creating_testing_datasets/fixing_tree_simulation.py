import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import pandas as pd
import tskit
import matplotlib.pyplot as plt
from glob import glob
import plotly.express as px
from itertools import combinations_with_replacement
from os import mkdir
import msprime


def _set_up_msprime_demography(world_map, pop_size, migration_rates):
    """Creates the msprime.Demography object for simulating trees

    Parameters
    ----------
    world_map : terracotta.WorldMap
    pop_size : int
        The population size of each deme
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type
    """

    demography = msprime.Demography()
    for id in world_map.demes["id"]:
        demography.add_population(name="Pop_"+str(id), initial_size=pop_size)
    for index,connection in world_map.connections.iterrows():
        rate = migration_rates.get(connection["type"], -1)
        if rate == -1:
            raise RuntimeError(f"Rate for connection of type '{connection["type"]}' is not provided. Please specify and try again.")
        demography.set_migration_rate("Pop_"+str(connection["deme_0"]), "Pop_"+str(connection["deme_1"]), rate)
        demography.set_migration_rate("Pop_"+str(connection["deme_1"]), "Pop_"+str(connection["deme_0"]), rate)
    return demography

def _simulate_independent_trees(
        world_map,
        number_of_trees,
        ploidy,
        pop_size,
        migration_rates,
        record_provenance
    ):
    """Simulates trees under a demographic model set by the world map

    Parameters
    ----------
    world_map : terracotta.WorldMap
    number_of_trees : int
        The number of independent trees to simulate
    ploidy : int
        The ploidy of the samples
    allow_multiple_samples_per_deme : bool
    pop_size : int
        The population size of each deme
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type
    """

    demography = _set_up_msprime_demography(world_map=world_map, pop_size=pop_size, migration_rates=migration_rates)
    samples = []
    for s in world_map.samples["deme"]:
        samples.append(msprime.SampleSet(1, population="Pop_"+str(s)))
    #samples = {}
    #for s in world_map.samples["deme"]:
    #    samples["Pop_"+str(s)] = samples.get("Pop_"+str(s), 0) + 1
    for i in range(number_of_trees):
        ts = msprime.sim_ancestry(
            samples=samples,
            ploidy=ploidy,
            demography=demography,
            record_full_arg=True,
            record_provenance=record_provenance
        )
        yield ts

def create_trees_files(
        demes_path,
        samples_path,
        number_of_trees,
        pop_size,
        record_provenance=True,
        migration_rate=None,
        migration_rates=None,
        output_directory="."
    ):
    """
    Parameters
    ----------
    demes_path : string
    samples_path : string
    number_of_trees : int
        The number of independent trees to simulate
    pop_size : int
        The population size of each deme
    record_provenance : bool
        Whether msprime should record the provenance of the trees. (default=True)
    migration_rate : float
        Single migration rate between neighboring demes. (default=None, ignored)
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type. (default=None, ignored)
    output_directory : string
        Path to directory where file will be written. (default=".")
    """

    ploidy = 1

    demes = pd.read_csv(demes_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")

    if migration_rate == None and migration_rates == None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`.")
    elif migration_rate != None and migration_rates != None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`, not both.")
    elif migration_rate != None:
        deme_types = demes["type"].unique()
        transitions = list(combinations_with_replacement(deme_types, 2))
        migration_rates = {i:migration_rate for i in range(len(transitions))}

    world_map = tct.WorldMap(demes=demes, samples=samples)

    mkdir(f"{output_directory}/trees")
    trees = _simulate_independent_trees(
        world_map=world_map,
        number_of_trees=number_of_trees,
        ploidy=ploidy,
        pop_size=pop_size,
        migration_rates=migration_rates,
        record_provenance=record_provenance
    )           
    for i,tree in enumerate(trees):
        tree.dump(f"{output_directory}/trees/{i}.trees")


create_trees_files(
    "datasets/all_samples/demes.tsv",
    "datasets/all_samples/samples.tsv",
    number_of_trees=2,
    record_provenance=True,
    pop_size=500,
    migration_rate=1e-05,
    output_directory="."
)