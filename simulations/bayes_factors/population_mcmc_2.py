from mcmc_utils import *
from math import log
from math import pi
from scipy.stats import uniform
from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import time
import os
import sys

if __name__ == "__main__":
    seed = int((time.time()*1000)%10000000)
    np.random.seed(seed)
    """ 
    MCMC parameters
    """
    num_steps = 500000
    num_chains = 2  ## greater than 4
    num_pairs = 20
    do_exchange = False
    dump_all = True
    temps = np.array([1.*(i+1)/(num_chains) for i in range(num_chains)])**5
    """
    Model parameters
    """
    suffix = "m2_mh500k"
    dir_name = "m2_mh"
    cols = ["mu1", "mu2"]
    initial_value = [-1., 1.]
    prior_min = [-5., -5.]
    prior_max = [5., 5.]
    initial_variance = 0.01
    ##data = np.array([-1.29641906, -3.12319291, -1.73057062,  1.56234748, -0.18096356,
    ##    -1.03496617,  3.83181641, -3.47513347,  1.32579626,  3.43152853]) 
    data = np.array([-4.9, -2.25, -0.3, -0.3, 1.9, 2.1, 1.95, 5., 2.9, 3.43152853]) 
    os.system("mkdir -p " + dir_name)
    """
    Chain initialisation
    """
    t0 = time.time()
    theta0 = np.array([initial_value] * num_chains)
    log_likelihood0 = np.array([calculate_log_likelihood_2(data, param) for param in theta0])
    log_likelihood = temps * log_likelihood0
    log_prior = np.array([calculate_log_prior_2(param) for param in theta0])
    files = [os.path.join(dir_name, "posteriors" + "_" + suffix + "_" + str(temp + 1) + ".csv") for temp in range(num_chains)]
    handles = [open(dump_name, "w") for dump_name in files]
    tmp = [f.write("\t".join(cols) + "\t" + "accept" + "\t" + "LnLike" + "\n") for f in handles]
    for i in range(num_steps):
        theta = [multivariate_normal.rvs(t, initial_variance) for t in theta0]
        theta = np.array([reflect_proposal_2(t, prior_min, prior_max) for t in theta])
        proposed_log_likelihood0 = np.array([calculate_log_likelihood_2(data, param) for param in theta])
        proposed_log_likelihood = temps * proposed_log_likelihood0
        proposed_log_prior = np.array([calculate_log_prior_2(param) for param in theta])
        log_hr = proposed_log_likelihood + proposed_log_prior - log_likelihood - log_prior
        accept = np.array([u < hr for u,hr in zip(np.log(uniform.rvs(size=num_chains)),log_hr)])
        theta0[accept] = theta[accept]
        log_likelihood0[accept] = proposed_log_likelihood0[accept]
        log_likelihood[accept] = proposed_log_likelihood[accept]
        log_prior[accept] = proposed_log_prior[accept]
        if do_exchange:
            for ii in np.random.choice(range(num_chains),num_pairs):
                jj = exchange(ii, num_chains)
                proposed_log_likelihood_ii = temps[ii] * log_likelihood0[jj]
                proposed_log_likelihood_jj = temps[jj] * log_likelihood0[ii]
                trans_prob = 0.
                exchange_log_hr =  proposed_log_likelihood_ii + proposed_log_likelihood_jj - log_likelihood[ii] - log_likelihood[jj]
                if (log(uniform.rvs()) < exchange_log_hr):
                    theta0[ii], theta0[jj] = theta0[jj].copy(), theta0[ii].copy()
                    log_prior[ii], log_prior[jj] = log_prior[jj], log_prior[ii]
                    log_likelihood0[ii], log_likelihood0[jj] = log_likelihood0[jj], log_likelihood0[ii]
                    log_likelihood[ii], log_likelihood[jj] = proposed_log_likelihood_ii, proposed_log_likelihood_jj
        posts = [np.append(theta0.copy()[temp,:], [accept[temp], log_likelihood.copy()[temp]]) for temp in range(num_chains)]
        tmp = [f.write("\t".join(str(x) for x in post) + "\n") for post, f in zip(posts, handles)]
        if not (i%10000):
            f.flush()
    print str(seed)
    print str((time.time() - t0)/60.) + " mins. to run " + str(num_steps) + " iters. with " \
    + str(num_chains) + " chains and exchange is " + str(do_exchange) + "!"
