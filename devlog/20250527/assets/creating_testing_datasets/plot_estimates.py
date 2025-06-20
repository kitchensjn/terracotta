import pandas as pd
import matplotlib.pyplot as plt

estimates = pd.read_csv("estimate_outputs_updated.csv")

plt.plot([10**i for i in range(-5,0)], [10**j for j in range(-5,0)], linestyle="dashed", color="#0C2320")
plt.scatter(estimates.loc[estimates["sample_type"]==0, "migration_rate"], estimates.loc[estimates["sample_type"]==0, "estimate"], zorder=3, color="#CC733E", label="Max one sample per deme")
plt.scatter(estimates.loc[estimates["sample_type"]==1, "migration_rate"], estimates.loc[estimates["sample_type"]==1, "estimate"], zorder=3, color="#551309", label="All samples")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("True Migration Rate")
plt.ylabel("Estimated Migration Rate")
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.legend()
plt.savefig("estimate_outputs_updated.svg")
plt.show()
