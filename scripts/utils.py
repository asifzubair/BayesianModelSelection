from math import log
from math import pi
from scipy.stats import uniform
from scipy.stats import multivariate_normal
from pymc.Matplot import plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sub_sample(df, burn_in = 5000, thin = 10):
    """remove burn-in and subsample"""
    return df[burn_in::thin]

def open_run_subsample(file_name, burn_in = 5000, thin = 10):
    df = pd.read_table(file_name)
    return sub_sample(df, burn_in, thin)

def trace_plot(file_name, suffix='', burn_in=5000, thin=10, name = "trace", verbose=1, last=False):
    """plots traces of various kinds"""
    df = pd.read_table(file_name)
    if verbose:
        print "*" * 40
        print "*" + "\t" + "Iterations: " + str(df.shape[0])
        print "*" + "\t" + "Burn In: " + str(burn_in)
        print "*" + "\t" + "Thin: " + str(thin)
        print "*" * 40
    df = sub_sample(df, burn_in, thin)
    ax = df["LnLike"].plot(figsize=(10,6), title = "Likelihood " + suffix, use_index = False, fontsize=15)
    fig = ax.get_figure()
    fig.savefig(name + "_trace.pdf", format='pdf', dpi=1200)
    return df
        
def marginal_likelihood(likelihood, temps):
    """estimate for the marginal likelihood"""
    delta = np.diff(temps)    
    delta = np.append(np.append(0,delta),0)
    deltas = delta[:-1] + delta[1:]
    means = likelihood.mean(axis=1)
    return 0.5*deltas.dot(means)

def posterior_sample(y, df, predict, name="sample_posteriors", sample_every=10):
    """samples from the posterior"""
    gap = ["Hb","Kni","Kr","Gt"]
    dom_antr = [30, 40, 20, 10]
    dom_post = [70, 90, 80, 90]
    post = sub_sample(df, 0, sample_every)
    fig, ax = plt.subplots(2,2)
    for i, a in enumerate(ax.flatten()):
        a.plot(y[i*100:100+i*100],"black")
        a.plot((dom_antr[i], dom_antr[i]), (0,1), 'k-')
        a.plot((dom_post[i], dom_post[i]), (0,1), 'k-')
        a.set_title(gap[i])
    for parms in post.iterrows():
        Ji = predict(parms[1][:-2])
        for i, a in enumerate(ax.flatten()):
            a.plot(Ji[i*100:100+i*100],'b-')
            a.set_title(gap[i])
    plt.show()
    fig.savefig(name, format='pdf', dpi=1200)
