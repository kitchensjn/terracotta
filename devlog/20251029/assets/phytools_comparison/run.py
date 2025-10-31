import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import tskit



#tree = tskit.load("trees/0.trees")
#print(tree.tables.edges)
#print(tree.node(13))

#exit()

print(tct.run_for_single(
    migration_rates=[0.01562150, 0.01749247],
    demes_path="demes.tsv",
    samples_path="samples.tsv",
    trees_dir_path="trees",
    asymmetric=True
))