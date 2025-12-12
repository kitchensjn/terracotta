import terracotta as tct

print(tct.run(
    demes_path="dataset/demes.tsv",
    samples_path="dataset/samples.tsv",
    trees_dir_path="dataset/trees",
    time_bins=[0]+[10**i for i in range(0,10)],
    output_file="results.tsv"
))