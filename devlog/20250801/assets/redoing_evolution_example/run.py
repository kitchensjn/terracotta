import pandas as pd
import numpy as np
from glob import glob
import tskit
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
from scipy.optimize import minimize, shgo
import time
from collections import defaultdict
import matplotlib.pyplot as plt


def remove_unattached_nodes(ts):
    """Removes any nodes that are not attached to any other nodes from the tree sequence
    
    Parameters
    ----------
    ts : tskit.TreeSequence

    Returns
    -------
    ts_final : tskitTreeSequence
        A tree sequence with unattached nodes removed
    """

    edge_table = ts.tables.edges
    connected_nodes = np.sort(np.unique(np.concatenate((edge_table.parent,edge_table.child))))
    ts_final = ts.subset(nodes=connected_nodes)
    return ts_final

def merge_unnecessary_roots(ts):
    """Merges root node IDs that are referring to the same node

    This commonly occurs as a result of decapitate(). Combines the two nodes into one and then
    removes the unattached node that is no longer important. This does not merge all roots into
    one, just those that are referring to the same root.

    Parameters
    ----------
    ts : tskit.TreeSequence

    Returns
    -------
    ts_new : tskitTreeSequence
        A tree sequence with corresponding roots merged
    """

    ts_tables = ts.dump_tables()
    edge_table = ts_tables.edges 
    parent = edge_table.parent
    roots = np.where(ts_tables.nodes.time == ts.max_time)[0]
    children = defaultdict(list)
    for root in roots:
        root_children = edge_table.child[np.where(edge_table.parent == root)[0]]
        for child in root_children:
            children[child] += [root]
    for child in children:
        pts = children[child]
        if len(pts) > 1:
            for pt in pts:
                if len(np.unique(edge_table.child[np.where(edge_table.parent == pt)[0]])) > 1:
                    print(pt, "has multiple children! Merge roots with caution.")
                parent[np.where(ts.tables.edges.parent == pt)[0]] = pts[0]
    edge_table.parent = parent 
    ts_tables.sort() 
    ts_new = remove_unattached_nodes(ts=ts_tables.tree_sequence())
    return ts_new

def chop_arg(ts, time):
    """Chops the tree sequence at a time in the past

    Parameters
    ----------
    ts : tskit.TreeSequence
    time : int
        Chops at `time` generations in the past

    Returns
    -------
    merged : tskitTreeSequence
        A tree sequence that has been decapitated and subset
    """

    decap = ts.decapitate(time)
    subset = decap.subset(nodes=np.where(decap.tables.nodes.time <= time)[0])
    merged = merge_unnecessary_roots(ts=subset)
    return merged



demes = pd.read_csv("dataset/demes.tsv", sep="\t")
samples = pd.read_csv("dataset/samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)

#trees = []
#for tree_file in glob(f"dataset/trees/*"):
#    ts = chop_arg(tskit.load(tree_file).simplify(), 1000)
#    trees.append(tct.nx_bin_ts(ts, [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000]).first())


trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000]).first() for ts in glob(f"dataset/trees/*")]

cl = []
bal = []
r = []
for tree in trees:
    child_list, branch_above_list, roots = tct.convert_tree_to_tuple_list(tree)
    cl.append(child_list)
    bal.append(branch_above_list)
    r.append(roots)

total_number_of_edges = 0
for tree in trees:
    total_number_of_edges += tree.num_edges+1
branch_lengths = np.zeros(total_number_of_edges, dtype="int64")
edge_counter = 0
for tree in trees:
    for node in tree.nodes(order="timeasc"):
        branch_lengths[edge_counter] = int(tree.branch_length(node))
        edge_counter += 1
branch_lengths = np.unique(np.array(branch_lengths))


mr = np.array([-4, -4, -10])
migration_rates = np.exp(mr)

transition_matrix = world_map.build_transition_matrix(migration_rates=migration_rates)
start = time.time()
precomputed_transitions = tct.precalculate_transitions(
    branch_lengths=branch_lengths,
    transition_matrix=transition_matrix
)
print(time.time() - start)

res = shgo(
    tct.calc_log_migration_rate_log_likelihood,
    bounds=[(-20, 20), (-20, 20), (-20, 20)],
    n=200,
    iters=5,
    sampling_method="sobol",
    args=(world_map, cl, bal, r, branch_lengths)
)

print(res)

plt.plot(res)
plt.show()