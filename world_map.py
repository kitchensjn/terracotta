import terracotta as tct
import pandas as pd

#tct._create_grid_demes_file(
#    side_length=10,
#    number_of_deme_types=1,
#    output_path="demes.tsv"
#)

demes = pd.read_csv("datasets/mmr_3/demes.tsv", sep="\t")
samples = pd.read_csv("datasets/mmr_3/samples.tsv", sep="\t")

world_map = tct.WorldMap(demes=demes)

world_map.draw(samples=samples)

