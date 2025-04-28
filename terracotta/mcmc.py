from terracotta.main import *
import numpy as np
import emcee


def lnprior(migration_rates):
    for mr in migration_rates:
        if not(0.0001 < mr < 1):
            return -np.inf
    return 0.0

def lnprob(migration_rates, world_map, trees):
    lp = lnprior(migration_rates)
    if not np.isfinite(lp):
        return -np.inf
    return lp + calc_migration_rate_log_likelihood(
        world_map=world_map,
        trees=trees,
        migration_rates={i:mr for i,mr in enumerate(migration_rates)}
    )[0]

def run(p0, nwalkers, niter, world_map, trees):
    ndim = len(p0[0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[world_map, trees])
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 50)
    sampler.reset()
    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)
    return sampler, pos, prob, state