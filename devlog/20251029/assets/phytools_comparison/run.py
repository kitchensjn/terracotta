import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import tskit
import pandas as pd



#ts = tskit.load("trees/0.trees")
#for node in tree.nodes(order="timeasc"):
#    print(node, node.time)

#demes = pd.read_csv("demes.tsv", sep="\t")
#samples = pd.read_csv("samples.tsv", sep="\t")
#world_map = tct.WorldMap(demes, samples, asymmetric=True)

#locs_0 = tct.locate_nodes_revell(
#    ts=ts,
#    world_map=world_map,
#    migration_rates=[0.01, 0.05]
#)


#locs_1 = tct.locate_nodes_revell_messages(
#    tree=ts.first(),
#    world_map=world_map,
#    migration_rates=[0.01, 0.05]
#)
#
#print(locs_0 - locs_1)


print(tct.run_on_trees(
    migration_rates=[0.01, 0.05],
    demes_path="demes.tsv",
    samples_path="samples.tsv",
    trees_dir_path="trees",
    asymmetric=True
))



#locs = tct.locate_nodes(
#    tree=ts.first(),
#    world_map=world_map,
#    migration_rates=[0.01, 0.05]
#)

#for node in tree.first().nodes(order="timeasc"):
#    print(node, locs[node])


#pos = tct.track_lineage_over_time(
#    sample=3,
#    times=[0, 85, 98],
#    tree=tree.first(),
#    world_map=world_map,
#    migration_rates=[0.01, 0.01]
#)

#for p in pos:
#    print(p, pos[p])


#tct.reconstruct_node_locations(
#    tree,
#    world_map,
#    [0.01562150, 0.01749247]
#)


#print(tree.tables.edges)
#print(tree.node(13))

#exit()

#print(tct.run_for_single(
#    migration_rates=[0.01, 0.05],
#    demes_path="demes.tsv",
#    samples_path="samples.tsv",
#    trees_dir_path="trees",
#    asymmetric=True
#))


