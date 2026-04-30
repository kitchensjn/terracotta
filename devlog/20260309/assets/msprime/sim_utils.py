import math
import pandas as pd
import numpy as np
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
from os import mkdir
import msprime


def _populate_vertices(num_rings, gap_between_rings=1):
    """
    Parameters
    ----------
    num_rings
    gap_between_rings : int

    Returns
    -------
    """

    if num_rings < 0:
        raise RuntimeError("`num_rings` must be >=0.")

    vertices = [
        {
            "id": 0,
            "xcoord": 0,
            "ycoord": 0
        }
    ]

    counter = 1

    for ring in range(1, num_rings+1):
        for corner in range(6):
            angle = (corner/6)*(2*math.pi)
            x = math.sin(angle) * (gap_between_rings*ring)
            y = math.cos(angle) * (gap_between_rings*ring)
            vertices.append({
                "id": counter,
                "xcoord": x,
                "ycoord": y
            })
            counter += 1
            
            next_angle = ((corner+1)/6)*(2*math.pi)
            next_x = math.sin(next_angle) * (gap_between_rings*ring)
            next_y = math.cos(next_angle) * (gap_between_rings*ring)
            for vertex in range(1, ring):
                new_x = x + (vertex/ring) * (next_x-x)
                new_y = y + (vertex/ring) * (next_y-y)
                vertices.append({
                    "id": counter,
                    "xcoord": new_x,
                    "ycoord": new_y
                })
                counter += 1

    vertices = pd.DataFrame(vertices)
    return vertices

def _populate_faces(num_rings):
    """Creates all of the edges in hexagonal world map

    Repurposed from other code so not the most efficient

    Parameters
    ----------
    num_rings : int
        Number of rings in the hexagonal world map
    
    Returns
    -------
    faces : pandas.DataFrame
        Associating demes with polygon faces
    """

    triangles = []
    starting_a = 0
    starting_b = 1
    for ring in range(num_rings+1):
        working_a = starting_a + 0
        working_b = starting_b + 0
        for edge in range(6):
            for i in range(ring):
                if (edge == 5) and (i == ring-1):
                    triangles.append((starting_a, starting_b, working_b))
                    starting_a = working_a + 1
                    starting_b = working_b + 1
                else:
                    triangles.append((working_a, working_b, working_b+1))
                if i < ring-1:
                    if (edge == 5) and (i == ring-2):
                        triangles.append((starting_a, working_a, working_b+1))
                    else:
                        triangles.append((working_a, working_a+1, working_b+1))
                        working_a += 1
                working_b += 1

    edges = []
    for t,tri in enumerate(triangles):
        for i in range(len(tri)):
            if (i < len(tri)-1):
                edges.append((t, tri[i], tri[i+1]))
            else:
                edges.append((t, tri[0], tri[-1]))

    faces = pd.DataFrame(edges, columns=["face", "deme_0", "deme_1"])
    return faces

def create_hexagonal_tri_grid(num_rings, output_directory="."):
    vertices = _populate_vertices(num_rings)
    vertices["suitability"] = 1
    vertices.to_csv(f"{output_directory}/demes.tsv", sep="\t", index=False)

    faces = _populate_faces(num_rings)
    edges = faces[["deme_0", "deme_1"]].drop_duplicates()
    edges_reverse = edges[["deme_1", "deme_0"]].rename(columns={"deme_1":"deme_0", "deme_0":"deme_1"})
    edges = pd.concat([edges, edges_reverse])
    edges = edges.sort_values(by=["deme_0", "deme_1"]).reset_index(drop=True)
    edges["migration_modifier"] = 1
    edges["id"] = range(len(edges))
    edges = edges[["id", "deme_0", "deme_1", "migration_modifier"]]
    edges.to_csv(f"{output_directory}/connections.tsv", sep="\t", index=False)

def get_num_vertices_in_ring(ring):
    if ring == 0:
        return 1
    edge_length = ring-1
    return 6 + edge_length*6

def get_starting_vertex_of_ring(ring):
    """
    First deme in ring is always positioned directly above center

    Parameters
    ----------
    ring : int
        ID of ring
    
    Returns
    -------
    total : int
        ID of first deme in that ring
    """

    if ring < 0:
        raise RuntimeError("`num_rings` must be >=0.")
    
    total = 0
    for i in range(ring):
        total += get_num_vertices_in_ring(i)
    return total

def get_number_of_demes(num_rings):
    """Calculates the number of demes in a world map with `num_rings` rings

    Parameters
    ----------
    num_rings : int
        Number of rings in the hexagonal world map
    
    Returns
    -------
    int : number of demes
    """

    return get_starting_vertex_of_ring(num_rings+1)
    
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
        Path to the `demes.tsv` file
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

def _set_up_msprime_demography(
        world_map,
        pop_size,
        migration_rate,
        migration_modifier_variables
    ):
    """Creates the msprime.Demography object for simulating trees

    Parameters
    ----------
    world_map : terracotta.WorldMap
        Custom object built using the `demes.tsv`, `connections.tsv`, and `samples.tsv` files
    pop_size : int
        The population size of each deme
    migration_rate : float
        Single migration rate between neighboring demes (default=None, ignored)
    migration_modifier_variables : dict
        Associates string variable from `connections.tsv` with modification value
    """

    demography = msprime.Demography()
    for i,row in world_map.demes.iterrows():
        suitability_formatter = row["suitability"].replace(",", ":").split(":")
        demography.add_population(name="Pop_"+str(row["id"]), initial_size=pop_size*float(suitability_formatter[1]))
        for epoch in range(2,len(suitability_formatter),2):
            demography.add_population_parameters_change(time=float(suitability_formatter[epoch]), population="Pop_"+str(row["id"]), initial_size=pop_size*float(suitability_formatter[epoch+1]))
    for _,connection in world_map.connections.iterrows():
        migration_formatter = connection["migration_modifier"].replace(",", ":").split(":")
        try:
            modifier = float(migration_formatter[1])
        except:
            modifier = migration_modifier_variables.get(migration_formatter[1], -1)
            if modifier == -1:
                raise RuntimeError("Did not find rate.")
        
        i_0 = world_map.demes.loc[world_map.demes["id"]==connection["deme_0"]].index[0]
        i_1 = world_map.demes.loc[world_map.demes["id"]==connection["deme_1"]].index[0]
        epoch_time = migration_formatter[0]
        demography.set_migration_rate(
            source="Pop_"+str(connection["deme_0"]),
            dest="Pop_"+str(connection["deme_1"]),
            rate=migration_rate*(world_map.demes[f"suitability_{epoch_time}"][i_1]/world_map.demes[f"suitability_{epoch_time}"][i_0])*modifier
        )
        for epoch in range(2,len(migration_formatter),2):
            epoch_time = migration_formatter[epoch]
            try:
                modifier = float(migration_formatter[epoch+1])
            except:
                modifier = migration_modifier_variables.get(migration_formatter[epoch+1], -1)
                if modifier == -1:
                    raise RuntimeError("Did not find rate.")
            demography.add_migration_rate_change(
                time=float(migration_formatter[epoch]),
                source="Pop_"+str(connection["deme_0"]),
                dest="Pop_"+str(connection["deme_1"]),
                rate=migration_rate*(world_map.demes[f"suitability_{epoch_time}"][i_1]/world_map.demes[f"suitability_{epoch_time}"][i_0])*modifier
            )
    return demography

def _simulate_independent_trees(
        world_map,
        number_of_trees,
        ploidy,
        pop_size,
        migration_rate,
        migration_modifier_variables
    ):
    """Simulates trees under a demographic model set by the world map

    Parameters
    ----------
    world_map : terracotta.WorldMap
        Custom object built using the `demes.tsv`, `connections.tsv`, and `samples.tsv` files
    number_of_trees : int
        The number of independent trees to simulate
    ploidy : int
        The ploidy of the samples
    pop_size : int
        The population size of each deme
    migration_rate : float
        Single migration rate between neighboring demes (default=None, ignored)
    migration_modifier_variables : dict
        Associates string variable from `connections.tsv` with modification value
    """

    demography = _set_up_msprime_demography(
        world_map=world_map,
        pop_size=pop_size,
        migration_rate=migration_rate,
        migration_modifier_variables=migration_modifier_variables
    )
    samples = []
    for s in world_map.samples["deme"]:
        samples.append(msprime.SampleSet(1, population="Pop_"+str(s)))
    ts = msprime.sim_ancestry(
        samples=samples,
        ploidy=ploidy,
        demography=demography,
        record_full_arg=True,
        num_replicates=number_of_trees
    )
    return ts

def create_trees_files(
        demes_path,
        connections_path,
        samples_path,
        number_of_trees,
        pop_size,
        migration_rate,
        migration_modifier_variables=None,
        ploidy=1,
        output_directory="."
    ):
    """
    Parameters
    ----------
    demes_path : str
        Path to the `demes.tsv` file
    connections_path : str
        Path to the `connections.tsv` file
    samples_path : str
        Path to the `samples.tsv` file
    number_of_trees : int
        The number of independent trees to simulate
    pop_size : int
        The population size of each deme
    migration_rate : float
        Single migration rate between neighboring demes (default=None, ignored)
    migration_modifier_variables : dict
    ploidy : int
        The ploidy of the individuals. (default=1, haploid)
    output_directory : str
        Path to directory where files will be written. (default=".")
    """

    demes = pd.read_csv(demes_path, sep="\t")
    connections = pd.read_csv(connections_path, sep="\t")
    samples = pd.read_csv(samples_path, sep="\t")
    world_map = tct.WorldMap(demes=demes, connections=connections, samples=samples)

    mkdir(f"{output_directory}/trees")
    trees = _simulate_independent_trees(
        world_map=world_map,
        number_of_trees=number_of_trees,
        ploidy=ploidy,
        pop_size=pop_size,
        migration_rate=migration_rate,
        migration_modifier_variables=migration_modifier_variables
    )           
    for i,tree in enumerate(trees):
        tree.dump(f"{output_directory}/trees/{i}.trees")