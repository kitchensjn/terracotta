import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct


results = pd.read_csv("out_bug_corrected.tsv", sep="\t")

plt.scatter(results["m0"],results["m1"], c=results["loglikelihood"])
plt.scatter(0.05, 0.002, color="red", marker="*", label="Truth")
plt.scatter(results.loc[results["loglikelihood"]==results["loglikelihood"].max(), "m0"], results.loc[results["loglikelihood"]==results["loglikelihood"].max(), "m1"], color="black", marker="*", label="Estimate")
plt.xlabel("Migration Rate Between Same Deme Type")
plt.ylabel("Migration Rate Between Different Deme Type")
plt.xscale("log")
plt.yscale("log")
plt.axis("square")
plt.legend(framealpha=1, loc="upper left")
plt.savefig("new_multiple_migration_rates_bug_corrected.png")
plt.show()


max_like = results.loc[results["loglikelihood"]==results["loglikelihood"].max()].iloc[0]

migration_rates = {
    0: max_like["m0"],
    1: max_like["m1"],
    2: max_like["m0"]
}

demes = pd.read_csv("demes.tsv", sep="\t")
world_map = tct.WorldMap(demes)
world_map.draw(color_demes=True, migration_rates=migration_rates, save_to="world_map_bug_corrected.png")

