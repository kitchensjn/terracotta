import terracotta as tct
import pandas as pd
import tskit


demes = pd.read_csv("dataset/demes.tsv", sep="\t")
samples = pd.read_csv("dataset/samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)
tree = tskit.load("dataset/trees/0.trees").simplify()
#tree = tree.first()

print(tree.draw_text())

exit()

locations = tct.track_lineage_in_tree(
    node=13,
    times=range(0,30500,500),
    tree=tree,
    world_map=world_map,
    migration_rates=[0.13117145]
)

counter = 0
for t in range(0,30500,500):
    print(t)
    world_map.draw(figsize=(10,10), location_vector=locations[counter], title=f"{t} generations ago", save_to=f"figures/{t}.png", show=False)
    counter += 1