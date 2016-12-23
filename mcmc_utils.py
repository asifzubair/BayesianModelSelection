import numpy as np
from math import log, pi
from scipy.stats import multivariate_normal
from pymc.Matplot import plot
import pandas as pd
import matplotlib.pyplot as plt


def proposal_step(theta, initial_variance):
    if np.uniform.rvs() > 0.9:
        t =  theta.copy()
        t[-1] = t[-1] + np.random.choice([-1,1],1)[0]
        return t
    return multivariate_normal.rvs(theta, initial_variance, 1)

def reflect_proposal(parametersProposed, parametersLower, parametersUpper):
    """reflectProposal: We use reflecting boundaries"""
    for ii in range(np.size(parametersProposed)):
        while (parametersProposed[ii] < parametersLower[ii] or parametersUpper[ii] < parametersProposed[ii]):
            if(parametersProposed[ii] < parametersLower[ii]):
                parametersProposed[ii] = (2*parametersLower[ii] - parametersProposed[ii])
            else:
                parametersProposed[ii] = (2*parametersUpper[ii] - parametersProposed[ii])
    return parametersProposed

def calculate_log_likelihood(target_data, exp_error, weight_vector, params, model):
    """Log likelihood of the data given the parameters"""
    exact_solution = model(params)
    LL = -(0.5)*(log(2*pi) + 2*log(exp_error) + ((exact_solution - target_data)**2)/(exp_error**2))
    return weight_vector.dot(LL)

def exchange(chain, num_chains):
    """Exchange scheme"""
    if not chain:
        return 1
    if chain == num_chains-1:
        return num_chains-2
    return np.random.choice([chain-1, chain+1],1)[0]

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
        df["LnLike"].plot(figsize=(10,6), title = "Likelihood " + suffix, use_index = False, fontsize=15)
        return
    df["LnLike"].plot(figsize=(10,6), title = "Likelihood " + suffix, use_index = False, fontsize=15)
    if not likelihood_only:
        for col in set(df.columns) - set(["LnLike", "accept"]):
            plot(df[col], col, suffix = "_" + suffix, last = last, verbose=0)
    return df

def marginal_likelihood(likelihood, temps):
    """Gives the estimate for the marginal likelihood"""
    delta = np.diff(temps)    
    delta = np.append(np.append(0,delta),0)
    deltas = delta[:-1] + delta[1:]
    means = likelihood.mean(axis=0)
    return 0.5*deltas.dot(means)

def posterior_sample(y, df, predict, name="sample_posteriors", sample_every=10):
    gap = ["Hb","Kni","Kr","Gt"]
    post = sub_sample(df, 0, sample_every)
    fig, ax = plt.subplots(2,2)
    for i, a in enumerate(ax.flatten()):
        a.plot(y[i*100:100+i*100],"black")
        a.set_title(gap[i])
    for parms in post.iterrows():
        Ji = predict(parms[1][:-2])
        for i, a in enumerate(ax.flatten()):
            a.plot(Ji[i*100:100+i*100],'b-')
            a.set_title(gap[i])
    plt.show()

def plot_expression(y, predict, parms, name="expression profile"):
    gap = ["Hb","Kni","Kr","Gt"]
    fig, ax = plt.subplots(2,2)
    for i, a in enumerate(ax.flatten()):
        a.plot(y[i*100:100+i*100],"black")
        a.set_title(gap[i])
    Ji = predict(parms)
    for i, a in enumerate(ax.flatten()):
        a.plot(Ji[i*100:100+i*100],'b-')
        a.set_title(gap[i])
    plt.show()
