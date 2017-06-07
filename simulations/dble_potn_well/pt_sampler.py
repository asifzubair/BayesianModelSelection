from math import log
from math import pi
from scipy.stats import uniform
from scipy.stats import multivariate_normal
from collections import defaultdict
from pymc.Matplot import plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
import sys
import yaml

""" 
MCMC Parameters
"""
dump_all = True
"""
Model Parameters
"""
prior_min = -2
prior_max = 2

def reflect_proposal(proposed, lower, upper):
    """reflectProposal: We use reflecting boundaries"""
    while (proposed < lower or upper < proposed):
        if(proposed < lower):
            proposed = (2*lower - proposed)
        else:
            proposed = (2*upper - proposed)
    return proposed

def exchange(chain, num_chains):
    """Exchange scheme"""
    if not chain:
        return 1
    if chain == num_chains-1:
        return num_chains-2
    return np.random.choice([chain-1, chain+1],1)[0]

def pt_sampler(logTarget, num_steps = 100000, num_chains = 10, num_pairs = None, do_exchange = True, 
    init = 1., var = 0.1, cols=["samples"], dir_name="target_density", suffix = "dble_potn"):
    """Parallel Tempering Sampler"""
    temps = np.array([1.*(i+1)/(num_chains) for i in range(num_chains)])**5
    if not num_pairs:
        num_pairs = num_chains
    x0 = np.array([init] * num_chains)
    log_likelihood0 = np.array([logTarget(x) for x in x0])
    log_likelihood = temps * log_likelihood0
    os.system("mkdir -p " + dir_name)
    files = [os.path.join(dir_name, "samples" + "_" + suffix + "_" + str(temp + 1) + ".csv") for temp in range(num_chains)]
    handles = [open(dump_name, "w") for dump_name in files]
    tmp = [f.write("\t".join(cols) + "\t" + "accept" + "\t" + "LnHR" + "\n") for f in handles]

    for i in range(num_steps):
        y = [multivariate_normal.rvs(t, var, 1) for t in x0]
        y = np.array([reflect_proposal(t, prior_min, prior_max) for t in y])
        proposed_log_likelihood0 = np.array([logTarget(x) for x in y])
        proposed_log_likelihood = temps * proposed_log_likelihood0
        log_hr = proposed_log_likelihood - log_likelihood
        accept = np.array([u < hr for u,hr in zip(np.log(uniform.rvs(size=num_chains)),log_hr)])
        x0[accept] = y[accept]
        log_likelihood0[accept] = proposed_log_likelihood0[accept]
        log_likelihood[accept] = proposed_log_likelihood[accept]

        if do_exchange:
            for ii in np.random.choice(range(num_chains),num_pairs):
                jj = exchange(ii, num_chains)
                proposed_log_likelihood_ii = temps[ii] * log_likelihood0[jj]
                proposed_log_likelihood_jj = temps[jj] * log_likelihood0[ii]
                exchange_log_hr =  proposed_log_likelihood_ii + proposed_log_likelihood_jj  - log_likelihood[ii] - log_likelihood[jj]
                if (log(uniform.rvs()) < exchange_log_hr):
                    x0[ii], x0[jj] = x0[jj].copy(), x0[ii].copy()
                    log_likelihood0[ii], log_likelihood0[jj] = log_likelihood0[jj], log_likelihood0[ii]
                    log_likelihood[ii], log_likelihood[jj] = proposed_log_likelihood_ii, proposed_log_likelihood_jj
        posts = [np.append(x0.copy()[temp], [accept[temp], log_hr.copy()[temp]]) for temp in range(num_chains)]
        tmp = [f.write("\t".join(str(x) for x in post) + "\n") for post, f in zip(posts, handles)]

def sub_sample(df, burn_in = 5000, thin = 10):
    """Remove burn-in and subsample"""
    return df[burn_in::thin]

def trace_plot(file_name, suffix='', burn_in=5000, thin=10, likelihood_only=False, verbose=1, last=False):
    """Plots traces of various kinds"""
    df = pd.read_table(file_name)
    if verbose:
        print "*" * 40
        print "*" + "\t" + "Iterations: " + str(df.shape[0])
        print "*" + "\t" + "Burn In: " + str(burn_in)
        print "*" + "\t" + "Thin: " + str(thin)
        print "*" * 40
    df = sub_sample(df, burn_in, thin)
    if likelihood_only:
        df["LnHR"].plot(figsize=(10,6), title = "Likelihood " + suffix, use_index = False, fontsize=15)
        return
    df["LnHR"].plot(figsize=(10,6), title = "Likelihood " + suffix, use_index = False, fontsize=15)
    if not likelihood_only:
        for col in set(df.columns) - set(["LnHR", "accept"]):
            plot(df[col], col, suffix = "_" + suffix, last = last, verbose=0)
    return df