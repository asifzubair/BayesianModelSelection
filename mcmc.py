from mcmc_utils import *
from math import log, pi
from scipy.stats import uniform, multivariate_normal
from pymc.Matplot import plot
import numpy as np
import pandas as pd
import time
import os
import sys
import yaml

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print "give me a config file !"
        sys.exit()

    with open(sys.argv[1], 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    exec('import ' + cfg['model']['class_file'])

    if cfg['mcmc']['seed']:
        np.random.seed(cfg['mcmc']['seed'])    
    else:
        seed = int((time.time()*1000)%10000000)
        np.random.seed(seed)
        cfg['mcmc']['seed'] = seed

    """ 
    MCMC parameters
    """
    num_steps = cfg['mcmc']['num_steps']
    num_chains = cfg['mcmc']['num_chains']  ## greater than 4
    num_pairs = cfg['mcmc']['num_pairs']
    do_exchange = cfg['mcmc']['do_exchange']
    dump_all = cfg['mcmc']['dump_all']
    suffix = cfg['mcmc']['suffix']
    dir_name = cfg['mcmc']['dir_name']
    temps = np.array([1.*(i+1)/(num_chains) for i in range(num_chains)])**5
    error = cfg['mcmc']['error']

    """
    Model parameters
    """
    cols = cfg['model']['param_names']
    prior_min = cfg['model']['prior_min']
    prior_max = cfg['model']['prior_max']

    if cfg['model']['initial_values']:
        initial_value = cfg['model']['initial_values']
    else:
        initial_value = [uniform.rvs(lower, upper-lower) for lower,upper in zip(prior_min,prior_max)]
        cfg['model']['initial_values'] = initial_value

    initial_variance = (1./6)*np.diag(cfg['mcmc']['initial_variance'])
    m = eval(cfg['model']['object'])
    weight_vector = m.domains
    data = m.y 

    """
    Dump the config file for posterity
    """
    os.system("mkdir -p " + dir_name)
    with open(os.path.join(dir_name, 'config.yml'), 'w') as outfile:
        outfile.write( yaml.dump(cfg) )

    """
    Chain initialisation
    """
    t0 = time.time()
    theta0 = np.array([initial_value] * num_chains)

    log_likelihood0 = np.array([calculate_log_likelihood(data, error, weight_vector, param, m.predict) for param in theta0])
    log_likelihood = temps * log_likelihood0

    # theta_matrix = []
    # accepted = []
    # LnLike = []
    # with open(os.path.join(dir_name, "posteriors" + "_" + suffix + ".csv"), 'w') as f:

    files = [os.path.join(dir_name, "posteriors" + "_" + suffix + "_" + str(temp + 1) + ".csv") for temp in range(num_chains)]
    handles = [open(dump_name, "w") for dump_name in files]
    tmp = [f.write("\t".join(cols) + "\t" + "accept" + "\t" + "LnLike" + "\n") for f in handles]

    for i in range(num_steps):
        theta = [multivariate_normal.rvs(t, initial_variance, 1) for t in theta0]
        theta = np.array([reflect_proposal(t, prior_min, prior_max) for t in theta])
        proposed_log_likelihood0 = np.array([calculate_log_likelihood(data, error, weight_vector, param, m.predict) for param in theta])
        proposed_log_likelihood = temps * proposed_log_likelihood0
        log_hr = proposed_log_likelihood - log_likelihood

        """ 
        Crossover could be included as well.
        """

        accept = np.array([u < hr for u,hr in zip(np.log(uniform.rvs(size=num_chains)),log_hr)])
        theta0[accept] = theta[accept]
        log_likelihood0[accept] = proposed_log_likelihood0[accept]
        log_likelihood[accept] = proposed_log_likelihood[accept]

        if do_exchange:
            for ii in np.random.choice(range(num_chains),num_pairs):
                jj = exchange(ii, num_chains)
                proposed_log_likelihood_ii = temps[ii] * log_likelihood0[jj]
                proposed_log_likelihood_jj = temps[jj] * log_likelihood0[ii]
                trans_prob = 0.
                # if (not ii) or (ii == num_chains-1):
                #     trans_prob = log(2)
                # if (not jj) or (jj == num_chains-1):
                #     trans_prob = log(2)

                exchange_log_hr =  proposed_log_likelihood_ii + proposed_log_likelihood_jj  - log_likelihood[ii] - log_likelihood[jj] + trans_prob
                if (log(uniform.rvs()) < exchange_log_hr):
                    theta0[ii], theta0[jj] = theta0[jj].copy(), theta0[ii].copy()
                    log_likelihood0[ii], log_likelihood0[jj] = log_likelihood0[jj], log_likelihood0[ii]
                    log_likelihood[ii], log_likelihood[jj] = proposed_log_likelihood_ii, proposed_log_likelihood_jj
        # theta_matrix.append(theta0.copy())
        # accepted.append(accept)
        # LnLike.append(log_likelihood.copy())

        posts = [np.append(theta0.copy()[temp, :], [accept[temp], log_likelihood.copy()[temp]]) for temp in range(num_chains)]
        tmp = [f.write("\t".join(str(x) for x in post) + "\n") for post, f in zip(posts, handles)]

        if not (i%10000):
            f.flush()

    print str((time.time() - t0)/60.) + " mins. to run " + str(num_steps) + " iters. with " + str(num_chains) + " chains and exchange is " + str(do_exchange) + "!"
   
    # theta_matrix = np.array(theta_matrix)
    # accepted = np.array(accepted)
    # LnLike = np.array(LnLike)
    # posteriors = theta_matrix[:,-1,:]
    # posteriors = pd.DataFrame(posteriors, columns=cols)
    # posteriors["accepted"] = accepted[:,-1]
    # posteriors["likelihood"] = LnLike[:,-1]
    # posteriors.to_csv(os.path.join(dir_name, "posteriors" + "_" + suffix + ".csv"), sep="\t", index=False)
    # if dump_all:
    #     for temp in range(len(temps)):
    #         post = theta_matrix[:,temp,:]
    #         post = pd.DataFrame(post, columns=cols)
    #         post["accepted"] = accepted[:,temp]
    #         post["likelihood"] = LnLike[:,temp]
    #         post.to_csv(os.path.join(dir_name, "posteriors" + "_" + suffix + "_" + str(temp + 1) + ".csv"), sep="\t", index=False)
    # for temp in range(len(temps)):
    #     post = np.append(theta0.copy[temp, :], [accept, log_likelihood.copy()])
    #     post = pd.DataFrame(post, columns=cols)
    #     post["accepted"] = accepted[:,temp]
    #     post["likelihood"] = LnLike[:,temp]
    #     post.to_csv(os.path.join(dir_name, "posteriors" + "_" + suffix + "_" + str(temp + 1) + ".csv"), sep="\t", index=False)
