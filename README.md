# terracotta

A belief propagation method for estimating the location of genetic ancestors from gene trees.


## Creating environment

Download the `.zip` of the repository from GitHub. This tutorial uses `conda` to manage the environment,
though there are many options if you prefer another.

```
conda env create -f environment.yml
conda activate terracotta
```

## Importing terracotta

```
import terracotta as tct
```

## Input Dataset

There are three main inputs to 

### `trees/`

A directory of `.trees` files in the `tskit` format. These files only contain a single tree each. In the future, this format may be
changed to a file of newick trees, but for now, many of the `tskit` functions are being used.

### `samples.tsv`



### `world_map.tsv`

### Simulating a dataset for a grid demography with `msprime`

![INSERT A PICTURE OF THE GRID]

`terrracotta` provides functionality for simulating a dataset under a grid demography using `msprime`. This has been useful for testing
the accuracy of migration rate estimates.

```
tct.create_grid_demography_dataset(
    side_length=10,
    number_of_deme_types=2,
    number_of_samples=25,
    allow_multiple_samples_per_deme=False,
    number_of_trees=100,
    ploidy=1,
    pop_size=500,
    migration_rates={0:0.01, 1:0.00001, 2:0.01},
    output_directory=directory
)
```
