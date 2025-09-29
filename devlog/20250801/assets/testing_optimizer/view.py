import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




with open("dataset_0/search_cleaned.txt", "w") as out:
    out.write("i\tj\tk\tloglike\n")
    with open("dataset_0/search.txt", "r") as file:
        for line in file:
            data = line.split()
            out.write("\t".join(data) + "\n")

df = pd.read_csv("dataset_0/search_cleaned.txt", sep="\t")


counter = 0
for _,row in df.sort_values("loglike").iterrows():
    if counter < 100:
        print(row)
    counter += 1


#print(df.loc[df["loglike"]==min(df["loglike"]), :])


#i = np.log(df["i"])
#plt.scatter(i, -df["loglike"])
#plt.show()