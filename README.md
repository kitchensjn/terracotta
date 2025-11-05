![Logo](devlog/20241202/assets/logo.png)


`terracotta` is a Python package for estimating migration surfaces and the locations of genetic ancestors from gene trees and/or small ancestral recombination graphs. **This package is in early development, so there may be rapid/breaking changes to the functionality and options going forward.**

### Creating environment

Download the `.zip` of the repository from GitHub. This tutorial uses `conda` to manage the environment, though there are many options if you prefer another.

```
conda env create -f environment.yml
conda activate terracotta
```

### Quickstart

Within the [example_dataset/](https://github.com/kitchensjn/terracotta/tree/main/example_dataset) folder, you can find an example `demes.tsv`, `samples.tsv`, and `trees/` folder which are the necessary inputs for this method. The following code can also be found in [example_dataset/run.py](https://github.com/kitchensjn/terracotta/tree/main/example_dataset/run.py).

#### Loading the inputs

```
import pandas as pd
import sys
sys.path.append("..")   # if you are running from `example_dataset/`, otherwise path to `terracotta/`
import terracotta as tct
import tskit
from glob import glob

demes = pd.read_csv("demes.tsv", sep="\t")
samples = pd.read_csv("samples.tsv", sep="\t")
world_map = tct.WorldMap(demes, samples)
trees = [tct.nx_bin_ts(tskit.load(ts).simplify(), [0]+[10**i for i in range(1,10)]).first() for ts in glob(f"trees/*")]
```

`trees/` is a preprocessed directory of `.trees` files in the `tskit` format. These files only contain a single tree each. By loading it in this way, you've created a list of `tskit.Trees` with times discretized.

#### View the world map

```
world_map.draw(
    figsize=(15,15),
    color_demes=True,
    show_samples=True
)
```

![Example World Map](example_dataset/readme_figures/world_map.png)

Here, demes have been colored according to their type set by `demes.tsv`. Orange circles mark demes with samples, where the size of the circle is proportional to the number of samples found in that deme.

#### Estimate the most likely migration surface

```
result, log_likelihood = tct.run(
    demes_path="demes.tsv",
    samples_path="samples.tsv",
    trees_dir_path="trees",
    asymmetric=True,
    output_file="results.txt"
)
```

`terracotta` uses a global optimization algorithm (SHGO) to locate the most likely migration surface given your set deme types.

#### Tracking a lineage over time

To track a lineage over time you need to estimate the location of the lineage at specified time points within the tree.

```
locations = tct.track_lineage_in_tree(
    node=0,
    times=range(0,100,10),
    tree=trees[0],
    world_map=world_map,
    migration_rates=result
)

for i,time in enumerate(range(0, 100, 10)):
    world_map.draw(figsize=(5,5), location_vector=locations[i], title=f"{time} generations ago")
```

### Map Builder

Within the [map_builder/](https://github.com/kitchensjn/terracotta/tree/main/map_builder) folder, there are two D3.js based tools for creating `demes.tsv` and `samples.tsv`. These can be useful for generating test cases, such as changing the type of demes interactively, though are not intended to replace more powerful GIS-based workflows. You will need to access these tools through a Live Server. See `tct.create_demes_file_from_world_builder()` and `tct.create_samples_file_from_world_builder()` for information about how to create your input files with these tool.
