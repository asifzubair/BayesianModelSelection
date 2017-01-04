import numpy as np
import pandas as pd
import sys, os

from utils import *
sys.path.append("..")
import models

num_chains = 10
temps = np.array([1.*(i+1)/(num_chains) for i in range(num_chains)])**5
runs_dir = sys.argv[1]
BURNIN = 300000
THIN = 100

runs = ["run_" + str(i).zfill(2) for i in range(1, 11)]
runs = [os.path.join(runs_dir, r) for r in runs]

for dir in runs:
    post = [os.path.join(dir, "posteriors_pt_1M_" + str(i) + ".csv") for i in range(1, 11)]
    likelihoods = np.array([np.array(open_run_subsample(file, BURNIN, THIN)["LnLike"]) for file in post])
    print marginal_likelihood(likelihoods, temps)

"""    
if model == "ma6":
    m = PapaModelA6()
elif model == "mb7":
    m = PapaModelB7()
elif model == "mb7r":
    m = PapaModelB7r()
elif model == "mc8":
    m = PapaModelC8()
elif model == "mB_Kr7":
    m = PapaModel_B_Kr7()
elif model == "mB_Kr7r":
    m = PapaModel_B_Kr7r()
elif model == "mB_Kr8":
    m = PapaModel_B_Kr8()
predict = m.predict()
data = m.y
"""
