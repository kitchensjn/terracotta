import pandas as pd
import msprime
from itertools import combinations_with_replacement
from os import mkdir
from terracotta.main import *
import math


# IF YOU DON'T HAVE A TREES FILE...

def create_random_samples_file(
        demes_path,
        number_of_samples,
        allow_multiple_samples_per_deme=True,
        output_path="samples.tsv"
    ):
    """Creates a samples file associated with a demes file with randomly placed samples

    Parameters
    ----------
    demes_path : str
        Path to the demes file
    number_of_samples : int
        Number of samples to be placed on the map
    allow_multiple_samples_per_deme : bool
        Whether to allow samples to be placed in the same deme. (default=True)
    output_path : str
        Path to directory where file will be written. (default="samples.tsv")
    """
    
    demes = pd.read_csv(demes_path, sep="\t")
    random_samples = np.random.choice(demes["id"], number_of_samples, replace=allow_multiple_samples_per_deme)
    with open(output_path, "w") as samples_file:
        samples_file.write("\t".join(["id", "deme"]) + "\n")
        for id,sample in enumerate(random_samples):
            samples_file.write(f"{id}\t{sample}\n")

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
    for _,connection in world_map.connections[0].iterrows():
        rate = migration_rates.get(connection["type"], -1)
        if rate == -1:
            raise RuntimeError(f"Rate for connection of type '{connection["type"]}' is not provided. Please specify and try again.")
        demography.set_migration_rate("Pop_"+str(connection["deme_0"]), "Pop_"+str(connection["deme_1"]), rate)
        #demography.set_migration_rate("Pop_"+str(connection["deme_1"]), "Pop_"+str(connection["deme_0"]), rate)        
    current_connection_type = world_map.connections[0]["type"]
    for epoch in range(1, len(world_map.epochs)):
        time = world_map.epochs[epoch]
        for i,connection in world_map.connections[epoch].iterrows():
            if connection["type"] != current_connection_type[i]:
                rate = migration_rates.get(connection["type"], -1)
                if rate == -1:
                    raise RuntimeError(f"Rate for connection of type '{connection["type"]}' is not provided. Please specify and try again.")
                demography.add_migration_rate_change(time=time, source="Pop_"+str(connection["deme_0"]), dest="Pop_"+str(connection["deme_1"]), rate=rate)
                #demography.add_migration_rate_change(time=time, source="Pop_"+str(connection["deme_1"]), dest="Pop_"+str(connection["deme_0"]), rate=rate)
        current_connection_type = world_map.connections[epoch]["type"]
    return demography

def _simulate_independent_trees(
        world_map,
        number_of_trees,
        ploidy,
        pop_size,
        migration_rates,
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
    for i in range(number_of_trees):
        ts = msprime.sim_ancestry(
            samples=samples,
            ploidy=ploidy,
            demography=demography,
            record_full_arg=True
        )
        yield ts

def create_trees_files(
        demes_path,
        samples_path,
        number_of_trees,
        pop_size,
        ploidy=1,
        migration_rate=None,
        migration_rates=None,
        asymmetric=False,
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
    ploidy : int
        The ploidy of the individuals. (default=1, haploid)
    record_provenance : bool
        Whether msprime should record the provenance of the trees. (default=True)
    migration_rate : float
        Single migration rate between neighboring demes. (default=None, ignored)
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type. (default=None, ignored)
    output_directory : string
        Path to directory where file will be written. (default=".")
    """

    demes = pd.read_csv(demes_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = WorldMap(demes=demes, samples=samples, asymmetric=asymmetric)

    if migration_rate == None and migration_rates == None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`.")
    elif migration_rate != None and migration_rates != None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`, not both.")
    elif migration_rate != None:
        migration_rates = {i:migration_rate for i in world_map.existing_connection_types}

    mkdir(f"{output_directory}/trees")
    trees = _simulate_independent_trees(
        world_map=world_map,
        number_of_trees=number_of_trees,
        ploidy=ploidy,
        pop_size=pop_size,
        migration_rates=migration_rates,
    )           
    for i,tree in enumerate(trees):
        tree.dump(f"{output_directory}/trees/{i}.trees")

## World Builder

def create_demes_file_from_world_builder(
        world_builder_path,
        output_path="demes.tsv"
    ):
    """Writes the demes file from the output of world builder

    Parameters
    ----------
    world_builder_path
        Path to the output CSV from world builder
    output_path : str
        Path to output file. (default="demes.tsv")
    """

    map_builder = pd.read_csv(world_builder_path, header=None, names=["i", "x", "y", "orig_type", "n", "c"])
    side_length = int(math.sqrt(len(map_builder)))
    max_x = map_builder["x"].max()
    max_y = map_builder["y"].max()
    included_demes = map_builder.loc[map_builder["c"] != "lightgrey"]
    ids = included_demes["i"].values
    colors = np.unique(included_demes["c"].values)

    with open(output_path, "w") as demes_file:
        demes_file.write("\t".join(["id", "xcoord", "ycoord", "type", "neighbours"]) + "\n")
        for _, deme in included_demes.iterrows():
            demes_file.write(f"{deme["i"]}\t{deme["x"]}\t{deme["y"]}\t{np.argmax(colors == deme["c"])}\t{deme["n"]}\n")




def create_demes_file_from_world_builder_old(
        world_builder_path,
        output_path="demes.tsv"
    ):
    """Writes the demes file from the output of world builder

    Parameters
    ----------
    world_builder_path
        Path to the output CSV from world builder
    output_path : str
        Path to output file. (default="demes.tsv")
    """

    map_builder = pd.read_csv(world_builder_path, header=None, names=["i", "x", "y", "orig_type", "n", "c"])
    side_length = int(math.sqrt(len(map_builder)))
    max_x = map_builder["x"].max()
    max_y = map_builder["y"].max()
    included_demes = map_builder.loc[map_builder["c"] != "lightgrey"]
    ids = included_demes["i"].values
    colors = np.unique(included_demes["c"].values)

    with open(output_path, "w") as demes_file:
        demes_file.write("\t".join(["id", "xcoord", "ycoord", "type", "neighbours"]) + "\n")
        for _, deme in included_demes.iterrows():
            x = int(deme["x"])
            y = int(deme["y"])
            neighbors = []
            if (deme["y"] > 0):
                n = int(x+(y-1)*side_length)
                if n in ids:
                    neighbors.append(str(n))
            if (deme["x"] > 0):
                n = int((x-1)+y*side_length)
                if n in ids:
                    neighbors.append(str(n))
            if (deme["x"] < max_x):
                n = int((x+1)+y*side_length)
                if n in ids:
                    neighbors.append(str(n))
            if (deme["y"] < max_y):
                n = int(x+(y+1)*side_length)
                if n in ids:
                    neighbors.append(str(n))
            demes_file.write(f"{x+y*side_length}\t{x}\t{y}\t{np.argmax(colors == deme["c"])}\t{",".join(neighbors)}\n")

def create_samples_file_from_world_builder(
        world_builder_path,
        output_path="samples.tsv"
    ):
    """Writes the samples file from the output of world builder

    File will always be "samples.tsv".

    Parameters
    ----------
    world_builder_path
        Path to the output CSV from world builder
    output_path : str
        Path to output file. (default="samples.tsv")
    """

    map_builder = pd.read_csv(world_builder_path, header=None, names=["i", "x", "y", "c", "s"])
    side_length = int(math.sqrt(len(map_builder)))
    max_x = map_builder["x"].max()
    max_y = map_builder["y"].max()
    included_demes = map_builder.loc[map_builder["c"] != "lightgrey"]
    ids = included_demes["i"].values

    with open(output_path, "w") as samples_file:
        samples_file.write("\t".join(["id", "deme"]) + "\n")
        sample_counter = 0
        for _, deme in included_demes.iterrows():
            x = int(deme["x"])
            y = int(deme["y"])
            neighbors = []
            if (deme["y"] > 0):
                n = int(x+(y-1)*side_length)
                if n in ids:
                    neighbors.append(str(n))
            if (deme["x"] > 0):
                n = int((x-1)+y*side_length)
                if n in ids:
                    neighbors.append(str(n))
            if (deme["x"] < max_x):
                n = int((x+1)+y*side_length)
                if n in ids:
                    neighbors.append(str(n))
            if (deme["y"] < max_y):
                n = int(x+(y+1)*side_length)
                if n in ids:
                    neighbors.append(str(n))
            for s in range(deme["s"]):
                samples_file.write(f"{sample_counter}\t{x+y*side_length}\n")
                sample_counter += 1

## Random Grid

def create_random_grid_demes_file(side_length, number_of_deme_types=1, output_path="demes.tsv"):
    """Creates the world map input file that pairs with grid.slim or msprime of specified size

    Parameters
    ----------
    side_length : int
        Number of populations along one side of the square grid metapopulation
    number_of_deme_types : int
        Set the number of deme types that will be randomly assigned. (default=1)
    path : str
        Path to output file. (default="demes.tsv")
    """

    with open(output_path, "w") as outfile:
        outfile.write("id\txcoord\tycoord\ttype\tneighbours\n")
        for y in range(side_length):
            for x in range(side_length):
                neighbors = []
                if (y > 0):
                    neighbors.append(str(x+(y-1)*side_length))
                if (x > 0):
                    neighbors.append(str((x-1)+y*side_length))
                if (x < side_length-1):
                    neighbors.append(str((x+1)+y*side_length))
                if (y < side_length-1):
                    neighbors.append(str(x+(y+1)*side_length))
                outfile.write(f"{x+y*side_length}\t{x}\t{y}\t{random.randint(0,number_of_deme_types-1)}\t{",".join(neighbors)}\n")

## ARG instead of trees

def create_arg_file(
        demes_path,
        samples_path,
        sequence_length,
        recombination_rate,
        pop_size,
        migration_rate=None,
        migration_rates=None,
        mutation_rate=None,
        output_path="arg.trees"
    ):
    """Simulates an ARG on the stepping stone demography

    Parameters
    ----------
    demes_path : string
    samples_path : string
    sequence_length : int
        The number of independent trees to simulate
    recombination_rate : float
    pop_size : int
        The population size of each deme
    migration_rate : float
        Single migration rate between neighboring demes. (default=None, ignored)
    migration_rates : dict
        Key is the type of connection and value is the migration rate of that connection type. (default=None, ignored)
    mutation_rate : float
        Mutation rate if users wants to simulation mutations on the ARG. (default=None, ignored)
    output_directory : string
        Path to directory where file will be written. (default=".")

    """

    ploidy = 1

    demes = pd.read_csv(demes_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")

    if migration_rate is None and migration_rates is None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`.")
    elif migration_rate is not None and migration_rates is not None:
        raise RuntimeError("Must provide either `migration_rate` or `migration_rates`, not both.")
    elif migration_rate is not None:
        deme_types = demes["type"].unique()
        transitions = list(combinations_with_replacement(deme_types, 2))
        migration_rates = {i:migration_rate for i in range(len(transitions))}

    world_map = WorldMap(demes=demes, samples=samples)

    demography = _set_up_msprime_demography(world_map=world_map, pop_size=pop_size, migration_rates=migration_rates)
    samples = []
    for s in world_map.samples["deme"]:
        samples.append(msprime.SampleSet(1, population="Pop_"+str(s)))
    arg = msprime.sim_ancestry(
        samples=samples,
        ploidy=ploidy,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        demography=demography,
        record_full_arg=True
    )
    if mutation_rate is not None:
        arg = msprime.sim_mutations(
            arg,
            rate=mutation_rate
        )
    arg.dump(output_path)


def create_trees_folder_from_ts(ts, output_path=".", gap_between_trees=1):
    mkdir(f"{output_path}/trees")
    for i in range(0, ts.num_trees, gap_between_trees):
        tree = ts.at_index(i)
        interval = tree.interval
        single_tree_ts = ts.keep_intervals([interval], simplify=True).trim()
        single_tree_ts.dump(f"{output_path}/trees/{i}.trees")