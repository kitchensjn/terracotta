import tskit
from glob import glob



for ts in glob(f"dataset/sim/trees/*"):
    ts = tskit.load(ts).simplify()
    print(ts.tables.nodes.time[-3:])
    print(ts.draw_text())
    exit()