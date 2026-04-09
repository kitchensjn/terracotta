import tskit
import numpy as np
from os import mkdir
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import pandas as pd


def _create_samples_file_from_tree_sequence(
        input_trees_path=None,
        ts=None,
        output_samples_path="samples.tsv",
        num_samples=-1,
        output_trees_path="samples.trees",
        random_seed=None
    ):
    """Creates the samples input file from a tree sequence.

    Optionally, subsets tree sequence to a specified number of random samples.

    Parameters
    ----------
    input_trees_path : str
    ts : tskit:TreeSequence
    output_samples_path : str
    num_samples : int
    output_trees_path : str
    random_seed : int
    """

    if input_trees_path is not None:
        ts = tskit.load(input_trees_path)
    elif ts is not None:
        pass
    else:
        raise RuntimeError("Must provide either an `input_trees_path` or `ts` as input.")

    if random_seed != None:
        np.seed(random_seed)

    if num_samples > 1:
        s = np.random.choice(ts.num_samples, num_samples, replace=False)
        ts = ts.simplify(samples=s)
        ts.dump(output_trees_path)

    with open(output_samples_path, "w") as outfile:
        outfile.write("id\tdeme\n")
        for sample in ts.samples():
            outfile.write(f"{sample}\t{ts.node(sample).population}\n")

def _create_grid_files(side_length, demes_path="demes.tsv", connections_path="connections.tsv"):
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

    with open(demes_path, "w") as demesfile:
        demesfile.write("id\txcoord\tycoord\tsuitability\n")
        with open(connections_path, "w") as connectionsfile:
            connectionsfile.write("id\tdeme_0\tdeme_1\tmigration_modifier\n")
            connection_id = 0
            for y in range(side_length):
                for x in range(side_length):
                    demesfile.write(f"{x+y*side_length}\t{x}\t{y}\t{1}\n")
                    if (y > 0):
                        connectionsfile.write(f"{connection_id}\t{x+y*side_length}\t{x+(y-1)*side_length}\t{"a"}\n")
                        connection_id += 1
                    if (x > 0):
                        connectionsfile.write(f"{connection_id}\t{x+y*side_length}\t{(x-1)+y*side_length}\t{"b"}\n")
                        connection_id += 1
                    if (x < side_length-1):
                        connectionsfile.write(f"{connection_id}\t{x+y*side_length}\t{(x+1)+y*side_length}\t{"c"}\n")
                        connection_id += 1
                    if (y < side_length-1):
                        connectionsfile.write(f"{connection_id}\t{x+y*side_length}\t{x+(y+1)*side_length}\t{"d"}\n")
                        connection_id += 1
                    

def create_dataset_from_slim_output(ts, side_length, gap_between_trees=1, num_random_samples=-1, output_directory="."):
    """Creates input files from SLiM simulation tree sequence

    Creates "demes.tsv", "samples.tsv", "trees/" folder with ".trees" files

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence output by the SLiM simulation
    side_length : int
        The number of demes along one side of the square grid demography
    gap_between_trees : int
        The gap between sampled trees in the tree sequence. (default=1 : all trees included)
    num_random_samples : int
        Simplify the tree sequence to this many samples. (default=-1 : ignored, all samples included)
    """

    if num_random_samples != -1:
        samples = list(np.random.choice(list(ts.samples()), num_random_samples, replace=False))
        ts = ts.simplify(samples=samples)
    _create_samples_file_from_tree_sequence(
        ts=ts,
        output_samples_path=f"{output_directory}/samples.tsv"
    )
    mkdir(f"{output_directory}/trees")
    for i in range(0, ts.num_trees, gap_between_trees):
        tree = ts.at_index(i)
        interval = tree.interval
        single_tree_ts = ts.keep_intervals([interval], simplify=True).trim()
        single_tree_ts.dump(f"{output_directory}/trees/{i}.trees")
    _create_grid_files(
        side_length=side_length,
        demes_path=f"{output_directory}/demes.tsv",
        connections_path=f"{output_directory}/connections.tsv"
    )


ts = tskit.load("slim_005.trees")
create_dataset_from_slim_output(ts, 5, num_random_samples=100, output_directory="dataset")