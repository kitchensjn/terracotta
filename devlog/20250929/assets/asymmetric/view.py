import tskit
from glob import glob
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct
import importlib
importlib.reload(tct)
import pandas as pd


demes = pd.read_csv("demes.tsv", sep="\t")

world_map = tct.WorldMap(demes)

world_map.draw((7,7), color_demes=True, color_connections=True)



trees = [tskit.load(tree).simplify().first() for tree in glob("symmetric/*")]

min = 0
max = 9

tmrcas = [[] for distance in range(1,(max-min+1))]

for sample0 in range(min, max+1):
    for sample1 in range(sample0+1, max+1):
        for tree in trees:
            tmrcas[sample1-sample0-1].append(tree.tmrca(sample0, sample1))

plt.violinplot(tmrcas, side="low")



trees = [tskit.load(tree).simplify().first() for tree in glob("fast_asymmetric/*")]

tmrcas = [[] for distance in range(1,(max-min+1))]

for sample0 in range(min, max+1):
    for sample1 in range(sample0+1, max+1):
        for tree in trees:
            tmrcas[sample1-sample0-1].append(tree.tmrca(sample0, sample1))

plt.violinplot(tmrcas, side="high")


plt.show()



trees = [tskit.load(tree).simplify().first() for tree in glob("symmetric/*")]
tmrcas = []
for tree in trees:
    tmrcas.append(tree.tmrca(4,5))

plt.violinplot(tmrcas, side="low")

trees = [tskit.load(tree).simplify().first() for tree in glob("fast_asymmetric/*")]
tmrcas = []
for tree in trees:
    tmrcas.append(tree.tmrca(4,5))

plt.violinplot(tmrcas, side="high")

plt.show()



tss = [tskit.load(tree).simplify() for tree in glob("fast_asymmetric/*")]
for ts in tss:
    tree = ts.first()
    root = tree.root
    #print(ts.population(ts.node(root).population).metadata["name"].split("_")[1])



trees = [tskit.load(tree).simplify().first() for tree in glob("symmetric/*")]
tmrcas = [[] for other_deme in range(min, max+1)]
for tree in trees:
    for other_deme in range(min, max+1):
        tmrcas[other_deme].append(tree.tmrca(4,other_deme))

plt.violinplot(tmrcas, showmeans=True, side="low")

trees = [tskit.load(tree).simplify().first() for tree in glob("asymmetric/*")]
tmrcas = [[] for other_deme in range(min, max+1)]
for tree in trees:
    for other_deme in range(min, max+1):
        tmrcas[other_deme].append(tree.tmrca(4,other_deme))

plt.violinplot(tmrcas, showmeans=True, side="high")

plt.show()



trees = [tskit.load(tree).simplify().first() for tree in glob("symmetric/*")]
tmrcas = [[] for other_deme in range(min, max+1)]
for tree in trees:
    for other_deme in range(min, max+1):
        tmrcas[other_deme].append(tree.tmrca(5,other_deme))

plt.violinplot(tmrcas, showmeans=True, side="low")

trees = [tskit.load(tree).simplify().first() for tree in glob("asymmetric/*")]
tmrcas = [[] for other_deme in range(min, max+1)]
for tree in trees:
    for other_deme in range(min, max+1):
        tmrcas[other_deme].append(tree.tmrca(5,other_deme))

plt.violinplot(tmrcas, showmeans=True, side="high")

plt.show()