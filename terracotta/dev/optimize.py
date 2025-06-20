import scipy
from terracotta.main import *

def run(migration_rates, world_map, trees):
    migration_rate_dict = {i: rate for i, rate in enumerate(migration_rates)}
    return calc_migration_rate_log_likelihood(world_map, trees, migration_rate_dict)[0]