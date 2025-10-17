import tskit
from glob import glob

ts = tskit.load("symmetric/0.trees")

print(ts.draw_text())