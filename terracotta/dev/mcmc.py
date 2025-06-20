from terracotta.main import *
import numpy as np
import emcee


def lnprior(migration_rates):
    for mr in migration_rates:
        if not(0.0000001 < mr < 10):
            return -np.inf
    return 0.0

def lnprob(migration_rates, world_map, trees, branch_lengths):
    lp = lnprior(migration_rates)
    if not np.isfinite(lp):
        return -np.inf
    prob = lp + calc_migration_rate_log_likelihood(
        world_map=world_map,
        trees=trees,
        migration_rates={i:mr for i,mr in enumerate(migration_rates)},
        branch_lengths=branch_lengths
    )[0]
    print(migration_rates, prob)
    return prob

def run(p0, nwalkers, niter, world_map, trees, save_to):
    ndim = len(p0[0])

    filename = save_to
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    total_number_of_edges = 0
    for tree in trees:
        total_number_of_edges += tree.num_edges+1
    branch_lengths = np.zeros(total_number_of_edges, dtype="int64")
    edge_counter = 0
    for tree in trees:
        for node in tree.nodes(order="timeasc"):
            branch_lengths[edge_counter] = int(tree.branch_length(node))
            edge_counter += 1
    branch_lengths = np.unique(np.array(branch_lengths))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, args=[world_map, trees, branch_lengths])
    #print("Running burn-in...")
    #p0, _, _ = sampler.run_mcmc(p0, 50, progress=True)
    #print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)
    return sampler, pos, prob, state