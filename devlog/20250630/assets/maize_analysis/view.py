import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("output.txt", sep=" ")
data = data.sort_values("tree")
plt.plot(data["tree"]*10000, data["rate"])
plt.yscale("log")
plt.ylabel("MLE Migration Rate")
plt.xlabel("Tree Index")
plt.show()