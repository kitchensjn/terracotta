import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




with open("coarse_search_cleaned.txt", "w") as out:
    out.write("i\tj\tk\tl\tloglike\n")
    with open("coarse_search_farm.txt", "r") as file:
        for line in file:
            data = line.split()
            out.write("\t".join(data) + "\n")

df = pd.read_csv("coarse_search_cleaned.txt", sep="\t")

sorted_df = df.sort_values(by="loglike")
print("Sorted by loglike:\n", sorted_df)

#print(df.loc[df["loglike"]==min(df["loglike"]), :])


#i = np.log(df["i"])
#plt.scatter(i, -df["loglike"])
#plt.show()