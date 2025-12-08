import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/terracotta")
import terracotta as tct


rate = 1e-5

result, log_likelihood = tct.run(
    demes_path=f"datasets/{rate}/demes.tsv",
    samples_path=f"datasets/{rate}/samples.tsv",
    trees_dir_path=f"datasets/{rate}/trees",
    asymmetric=False,
    time_bins=[0]+[10**t for t in range(0,10)]
)

print(result, log_likelihood)