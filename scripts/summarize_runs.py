from math import log
from math import pi
from scipy.stats import uniform
from scipy.stats import multivariate_normal
from pymc.Matplot import plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

from utils import *
sys.path.append("..")
import models

num_chains = 10
temps = np.array([1.*(i+1)/(num_chains) for i in range(num_chains)])**5

model = sys.argv[1]
runs = sys.argv[2]

BURNIN = 300000
THIN = 100

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


df = trace_plot(os.path.join("run_1", "posteriors_pt_1M_10.csv"), burn_in = BURNIN, thin = THIN, name = )
df1 = sub_sample(pd.read_table(""), burn_in = BURNIN, thin = THIN)


