import sys
sys.path.append("../..")
import terracotta as tct
import importlib
importlib.reload(tct)


tct.create_trees_files(
    demes_path="dataset_2/demes.tsv",
    samples_path="dataset_2/samples.tsv",
    number_of_trees=100,
    pop_size=50,
    migration_rates={
        0:0.001,
        1:0.010,
        2:0.010,
        3:0.100
    },
    output_directory="dataset_2"
)