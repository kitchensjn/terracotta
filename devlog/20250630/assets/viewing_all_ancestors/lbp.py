import numpy as np
from scipy import linalg
from scipy.special import logsumexp
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import tskit
import pandas as pd
import matplotlib.pyplot as plt
import random


class Message:
    
    def __init__(self, sender, receiver, factor, message, fixed=False):
        self.sender = sender
        self.receiver = receiver
        self.factor = factor
        self.message = message
        self.fixed = fixed
    
    def update(self, incoming):
        if not self.fixed:
            for i in range(1,len(incoming)):
                incoming[0] = np.multiply(incoming[0], incoming[i])
            node_loc = incoming[0] / np.sum(incoming[0])
            self.message = np.matmul(node_loc, self.factor)


class BeliefPropagation():

    def __init__(self, ts, branch_lengths, precalculated_factors, world_map):
        self.ts = ts
        self.world_map = world_map
        self.messages = self.create_message_list(ts, world_map, branch_lengths, precalculated_factors)    #add checks to confirm that messages is of correct content
        self.links = self.link_messages()

    def create_message_list(self, ts, world_map, branch_lengths, precalculated_factors):
        messages = []
        for edge in ts.edges():
            edge_length = int(ts.node(edge.parent).time-ts.node(edge.child).time)
            if edge_length in branch_lengths:
                bl_index = np.where(branch_lengths==edge_length)[0][0]
                factor = precalculated_factors[bl_index]
            else:
                raise RuntimeError("Unexpected time jump. " + str(edge_length))
            if edge.child in world_map.samples.id:
                if edge.parent in world_map.samples.id:
                    #we don't care about updating a message between two nodes with known positions
                    pass
                else:
                    #we only need the child-parent message, not parent-child message
                    child_loc = world_map.samples.deme[np.where(world_map.samples.id==edge.child)[0][0]]
                    location = np.zeros((1,len(world_map.demes)))
                    location[0,np.where(world_map.demes.id==child_loc)[0][0]] = 1
                    messages.append(
                        Message(
                            sender=edge.child,
                            receiver=edge.parent,
                            factor=factor,
                            message=np.matmul(location, factor),
                            fixed=True
                        )
                    )     
            elif edge.parent in world_map.samples.id:
                #we only need the parent-child message, not child-parent message
                parent_loc = world_map.samples.deme[np.where(world_map.samples.id==edge.parent)[0][0]]
                location = np.zeros((1,len(world_map.demes)))
                location[0,np.where(world_map.demes.id==parent_loc)[0][0]] = 1
                messages.append(
                    Message(
                        sender=edge.parent,
                        receiver=edge.child,
                        factor=factor,
                        message=np.matmul(location, factor),
                        fixed=True
                    )
                )
            else:
                #we need both messages
                messages.extend([
                    Message(
                        sender=edge.child,
                        receiver=edge.parent,
                        factor=factor,
                        message=np.ones((1,len(world_map.demes)))
                    ),
                    Message(
                        sender=edge.parent,
                        receiver=edge.child,
                        factor=factor,
                        message=np.ones((1,len(world_map.demes)))
                    )
                ])
        return messages

    def link_messages(self):
        links = {}
        for i,m1 in enumerate(self.messages):
            for j,m2 in enumerate(self.messages):
                if (m2.receiver == m1.sender) and (m2.sender != m1.receiver):
                    links[i] = links.get(i, []) + [j]
        return links
    
    def locate_node(self, id):
        to_combine = []
        for message in self.messages:
            if message.receiver == id:
                to_combine.append(message.message)
        for i in range(1,len(to_combine)):
            to_combine[0] = np.multiply(to_combine[0], to_combine[i])
        return to_combine[0] / np.sum(to_combine[0])
    
    def locate_lineage_at_times(self, sample, genome_position, times, transition_matrix):
        tree = self.ts.at(genome_position)
        ancestors = [sample] + list(tct.ancs(tree=tree, u=sample))
        node_times = []
        for a in ancestors:
            node_times.append(tree.time(a))
        pc_combos = []
        for t in times:
            for i,v in enumerate(node_times):
                if v > t:
                    child = ancestors[i-1]
                    parent = ancestors[i]
                    break
                elif v == t:
                    child = ancestors[i]
                    parent = ancestors[i]
                    break
            pc_combos.append((child, parent))
        positions = {}
        for element, node_combo in enumerate(pc_combos):
            if node_combo[0] == node_combo[1]:
                if node_combo[0] in self.world_map.sample_location_vectors:
                    node_pos = self.world_map.sample_location_vectors[node_combo[0]]
                else:
                    to_combine = []
                    for message in self.messages:
                        if message.receiver == node_combo[0]:
                            to_combine.append(message.message)
                    if len(to_combine) > 0:
                        for i in range(1,len(to_combine)):
                            to_combine[0] = np.multiply(to_combine[0], to_combine[i])
                        node_pos = to_combine[0] / np.sum(to_combine[0])
                    else:
                        raise RuntimeError(f"No incoming messages to {node_combo[0]}. Can't locate.")
            else:
                if node_combo[0] in world_map.sample_location_vectors:
                    child_pos = world_map.sample_location_vectors[node_combo[0]]
                else:
                    incoming_messages_child = []
                    for message in self.messages:
                        if (message.receiver == node_combo[0]) and (message.sender != node_combo[1]):
                            incoming_messages_child.append(message.message)
                    if len(incoming_messages_child) > 0:
                        for i in range(1,len(incoming_messages_child)):
                            incoming_messages_child[0] = np.multiply(incoming_messages_child[0], incoming_messages_child[i])
                        child_pos = incoming_messages_child[0] / np.sum(incoming_messages_child[0])
                    else:
                        raise RuntimeError(f"No incoming messages to {node_combo[0]}. Can't locate.")
                branch_length_to_child = int(times[element] - tree.time(node_combo[0]))
                outgoing_child_message = np.matmul(child_pos, np.linalg.matrix_power(linalg.expm(transition_matrix), branch_length_to_child))
                if node_combo[1] in world_map.sample_location_vectors:
                    parent_pos = world_map.sample_location_vectors[node_combo[1]]
                else:
                    incoming_messages_parent = []
                    for message in self.messages:
                        if (message.receiver == node_combo[1]) and (message.sender != node_combo[0]):
                            incoming_messages_parent.append(message.message)
                    if len(incoming_messages_parent) > 0:
                        for i in range(1,len(incoming_messages_parent)):
                            incoming_messages_parent[0] = np.multiply(incoming_messages_parent[0], incoming_messages_parent[i])
                        parent_pos = incoming_messages_parent[0] / np.sum(incoming_messages_parent[0])
                    else:
                        raise RuntimeError(f"No incoming messages to {node_combo[1]}. Can't locate.")
                branch_length_to_parent = int(tree.time(node_combo[1]) - times[element])
                outgoing_parent_message = np.matmul(parent_pos, np.linalg.matrix_power(linalg.expm(transition_matrix), branch_length_to_parent))
                node_pos = np.multiply(outgoing_child_message, outgoing_parent_message)
                node_pos = node_pos / np.sum(node_pos)
            positions[times[element]] = node_pos
        return positions

    
    def iterate(self):
        for i,message in enumerate(self.messages):
            incoming_messages_index = self.links.get(i, [])
            if len(incoming_messages_index) > 0:
                message.update(incoming=[self.messages[im].message for im in incoming_messages_index])


def simplify_with_recombination(ts, flag_recomb=False, keep_nodes=None):
    """Simplifies a tree sequence while keeping recombination nodes

    Removes unary nodes that are not recombination nodes. Does not remove non-genetic ancestors.
    Edges intervals are not updated. This differs from how tskit's TreeSequence.simplify() works.

    Parameters
    ----------
    ts : tskit.TreeSequence
    flag_recomb (optional) : bool
        Whether to add msprime node flags. Default is False.
    keep_nodes (optional) : list
        List of node IDs that should be kept. Default is None, so empty list.

    Returns
    -------
    ts_sim : tskit.TreeSequence
        Simplified tree sequence
    maps_sim : numpy.ndarray
        Mapping for nodes in the simplified tree sequence versus the original
    """

    if keep_nodes == None:
        keep_nodes = []

    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0)
    child_node, parents_count = np.unique(uniq_child_parent[:, 0], return_counts=True) #For each child, count how many parents it has.
    parent_node, children_count = np.unique(uniq_child_parent[:, 1], return_counts=True) #For each child, count how many parents it has.
    multiple_parents = child_node[parents_count > 1] #Find children who have more than 1 parent. 
    recomb_nodes = ts.edges_parent[np.isin(ts.edges_child, multiple_parents)] #Find the parent nodes of the children with multiple parents. 
    
    if flag_recomb:
        ts_tables = ts.dump_tables()
        node_table = ts_tables.nodes
        flags = node_table.flags
        flags[recomb_nodes] = 131072 #msprime.NODE_IS_RE_EVENT
        node_table.flags = flags
        ts_tables.sort() 
        ts = ts_tables.tree_sequence()
    
    keep_nodes = np.unique(np.concatenate((keep_nodes, recomb_nodes)))
    potentially_uninformative = np.intersect1d(child_node[np.where(parents_count!=0)[0]], parent_node[np.where(children_count==1)[0]])
    truly_uninformative = np.delete(potentially_uninformative, np.where(np.isin(potentially_uninformative, keep_nodes)))
    all_nodes = np.array(range(ts.num_nodes))
    important = np.delete(all_nodes, np.where(np.isin(all_nodes, truly_uninformative)))
    ts_sim, maps_sim = ts.simplify(samples=important, map_nodes=True, keep_input_roots=False, keep_unary=False, update_sample_flags=False)
    return ts_sim, maps_sim

def count_bubbles(ts):
    """
    """

    node_table = ts.tables.nodes
    recomb_nodes = np.where(node_table.flags==131072)[0]
    num_bubbles = 0
    for i in range(0,len(recomb_nodes),2):
        parents1 = np.unique(ts.edges_parent[np.where(ts.edges_child==recomb_nodes[i])[0]])
        parents2 = np.unique(ts.edges_parent[np.where(ts.edges_child==recomb_nodes[i+1])[0]])
        if (len(parents1) == 1) and (len(parents2 == 1)) and (parents1[0] == parents2[0]):
            num_bubbles += 1
    return num_bubbles


#tct.create_trees_files(
#    demes_path="/Users/jameskitchens/Documents/GitHub/terracotta/devlog/20250630/assets/viewing_all_ancestors/dataset/demes.tsv",
#    samples_path="/Users/jameskitchens/Documents/GitHub/terracotta/devlog/20250630/assets/viewing_all_ancestors/dataset/samples.tsv",
#    number_of_trees=1,
#    pop_size=50,
#    migration_rate=0.01,
#    output_directory="/Users/jameskitchens/Documents/GitHub/terracotta/devlog/20250630/assets/viewing_all_ancestors/dataset"
#)

tct.create_arg_file(
    demes_path="/Users/jameskitchens/Documents/GitHub/terracotta/devlog/20250630/assets/viewing_all_ancestors/dataset/demes.tsv",
    samples_path="/Users/jameskitchens/Documents/GitHub/terracotta/devlog/20250630/assets/viewing_all_ancestors/dataset/samples_small.tsv",
    pop_size=50,
    sequence_length=100000,
    recombination_rate=1e-8,
    migration_rate=0.01,
    output_path="/Users/jameskitchens/Documents/GitHub/terracotta/devlog/20250630/assets/viewing_all_ancestors/dataset/arg.trees"
)

ts = tskit.load("dataset/arg.trees")
print(ts)
ts_sim, maps_sim = ts.simplify(map_nodes=True)
print(ts_sim)
ts_reduced, maps_reduced = simplify_with_recombination(ts=ts)
print(count_bubbles(ts=ts_reduced))

demes = pd.read_csv("dataset/demes.tsv", sep="\t")
samples = pd.read_csv("dataset/samples_small.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)

total_number_of_edges = ts_sim.num_edges+1
branch_lengths = np.zeros(total_number_of_edges, dtype="int64")
edge_counter = 0
for edge in ts_sim.edges():
    branch_lengths[edge_counter] = int(ts_sim.node(edge.parent).time - ts_sim.node(edge.child).time)
    edge_counter += 1
branch_lengths = np.unique(np.array(branch_lengths))

transition_matrix = world_map.build_transition_matrix(migration_rates={0:0.01, 1:0.01, 2:0.01})
trans, log = tct.precalculate_transitions(
    branch_lengths=branch_lengths,
    transition_matrix=transition_matrix
)

times = range(0, 2010, 10)
genome_position = 0

bp = BeliefPropagation(
    ts=ts_sim,
    world_map=world_map,
    branch_lengths=branch_lengths,
    precalculated_factors=trans
)

for j in range(200):
    bp.iterate()

colors = ["blue", "orange", "red", "green", "purple", "black"]


random_samples = random.sample(list(ts.samples()), 1)

for color_index, sample in enumerate(random_samples):
    print(sample)

    estimated_positions_tree = tct.track_lineage_over_time(
        sample=sample,
        times=times,
        tree=ts_sim.at(genome_position),
        world_map=world_map,
        migration_rates=np.array([0.01, 0.01, 0.01])
    )
    
    print("- tree positions")

    tree = ts.at(genome_position)
    ancestors = [sample] + list(tct.ancs(tree=tree, u=sample))

    true_positions = {}
    a = sample
    while a != -1:
        true_positions[ts.node(a).time] = int(ts.population(ts.node(a).population).metadata["name"].split("_")[-1])
        a = tree.parent(a)

    df = pd.DataFrame({"time":true_positions.keys(), "deme":true_positions.values()})

    print("- true positions")

    estimated_positions_arg = bp.locate_lineage_at_times(
        sample=sample,
        genome_position=genome_position,
        times=times,
        transition_matrix=transition_matrix
    )

    print("- ARG positions")

    single_sample = samples.loc[samples["id"]==sample,].reset_index(drop=True)
    single_sample["id"] = 0
    world_map_single = tct.WorldMap(demes, single_sample)
    uninformed_ts = ts.simplify(samples=[sample], keep_input_roots=True)

    estimated_positions_uninformed = tct.track_lineage_over_time(
        sample=0,
        times=times,
        tree=uninformed_ts.first(),
        world_map=world_map_single,
        migration_rates=np.array([0.01, 0.01, 0.01])
    )

    print("- uninformed positions")

    probs_tree = []
    probs_arg = []
    probs_uninformed = []
    for time in times:
        filtered = df.loc[df["time"]<=time,:]
        deme = filtered.loc[filtered["time"]==filtered["time"].max(),"deme"].iloc[0]
        prob_tree = estimated_positions_tree[time][0][np.where(world_map.demes["id"] == deme)[0][0]]
        prob_arg = estimated_positions_arg[time][0][np.where(world_map.demes["id"] == deme)[0][0]]
        prob_uninformed = estimated_positions_uninformed[time][0][np.where(world_map.demes["id"] == deme)[0][0]]
        probs_tree.append(prob_tree)
        probs_arg.append(prob_arg)
        probs_uninformed.append(prob_uninformed)

    plt.plot(times, probs_tree, linestyle="dashed", color=colors[color_index])
    plt.plot(times, probs_arg, color=colors[color_index])
    plt.plot(times, probs_uninformed, linestyle="dotted", color=colors[color_index])

plt.axhline(1/len(world_map.demes), linestyle="dashed", color="grey")
plt.yscale("log")
plt.legend()
plt.xlabel("Generations In Past")
plt.ylabel("Probability Of True Location")
plt.show()