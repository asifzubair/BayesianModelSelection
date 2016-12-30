from math import log
import numpy as np
from scipy.stats import uniform, multivariate_normal
import time
from pymc.Matplot import plot
import pandas as pd
import model
from mcmc_utils import *

class PtMcmc:
    def __init__(self, model, initial_value, prior_min, prior_max, initial_variance, cols, num_steps = 50000, num_chains = 10, error = 0.1, num_pairs = 10, do_exchange = True, dump_all = False, suffix="pt"):
        if not prior_min:
            self.prior_min = [0.0001, 6, 6, 6, 0.0001, 0.0001]

        if not prior_max:
            self.prior_max = [1, 9, 9, 9, 2, 2]

        if not initial_variance:
            self.initial_variance = (1./6)*np.diag([1./100,1./100,1./100,1./100,1./100,1./100])**2

        if not cols:
            self.cols = ["alpha", "K", "K1", "K2", "Co", "D"]

        self.initial_value = initial_value
        self.num_steps = num_steps
        self.num_chains = num_chains
        self.temps = np.array([1.*i/(num_chains - 1) for i in range(num_chains)])**5
        self.error = error
        self.do_exchange = do_exchange
        self.dump_all = dump_all
        self.model = model
    
    def sampler(self):
        t0 = time.time()
        theta0 = np.array([self.initial_value] * self.num_chains)
        log_likelihood = np.array([calculate_log_likelihood(self.model.y, error, self.model.domains, param, temp, self.model.predict) for param,temp in zip(theta0, self.temps)])

        theta_matrix = []
        accepted = []
        LnLike = []

        for i in range(num_steps):
            theta = [multivariate_normal.rvs(t, self.initial_variance, 1) for t in theta0]
            theta = np.array([reflect_proposal(t, self.prior_min, self.prior_max) for t in theta])
            proposed_log_likelihood = np.array([calculate_log_likelihood(self.model.y, self.error, self.model.domains, param, temp, self.model.predict) for param, temp in zip(theta, self.temps)])
            log_hr = proposed_log_likelihood - log_likelihood

            """ 
            Crossover could be included as well.
            """
            accept = np.array([u < hr for u,hr in zip(np.log(uniform.rvs(size=num_chains)),log_hr)])
            theta0[accept] = theta[accept]
            log_likelihood[accept] = proposed_log_likelihood[accept]

            if do_exchange:
                for ii in np.random.choice(range(_.num_chains),_.num_pairs):
                    jj = exchange(ii, self.num_chains)
                    if jj:
                        proposed_log_likelihood_ii = 1.*(temps[ii]/temps[jj])*log_likelihood[jj]
                    else:
                        proposed_log_likelihood_ii = calculate_log_likelihood(self.model.y, self.error, self.model.domains, theta0[ii], temps[ii], self.model.predict)
                    if ii:          
                        proposed_log_likelihood_jj = 1.*(temps[jj]/temps[ii])*log_likelihood[ii]
                    else:
                        proposed_log_likelihood_jj = calculate_log_likelihood(self.model.y, self.error, self.model.domains, theta0[jj], temps[jj], self.model.predict)
                    exchange_log_hr =  proposed_log_likelihood_ii + proposed_log_likelihood_jj  - log_likelihood[ii] - log_likelihood[jj]
                    if (log(uniform.rvs()) < exchange_log_hr):
                        theta0[ii], theta0[jj] = theta0[jj].copy(), theta0[ii].copy()
                        log_likelihood[ii], log_likelihood[jj] = proposed_log_likelihood_ii, proposed_log_likelihood_jj
            theta_matrix.append(theta0.copy())
            accepted.append(accept)
            LnLike.append(log_likelihood)

        print str((time.time() - t0)/60.) + " mins. to run " + str(self.num_steps) + " iters. with " + str(self.num_chains) + " chains and exchange is " + str(self.do_exchange) + "!"

        theta_matrix = np.array(theta_matrix)
        accepted = np.array(accepted)
        LnLike = np.array(LnLike)

        return [theta_matrix, accepted, LnLike]
