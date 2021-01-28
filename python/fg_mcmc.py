import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from likelihoods import LikelihoodFg
from gibbs import GibbsSampler
from corr_coeff_utils import compute_spectra, set_multipole_range
import os
import yaml
import sys

yaml_file = sys.argv[1]
with open(yaml_file) as yml:
    cdict = yaml.load(yml, Loader = yaml.FullLoader)

Ldict = {key : cdict[key] for key in cdict if (key in LikelihoodFg.__init__.__code__.co_varnames)}
Lclass = LikelihoodFg(**Ldict)

Gdict = {key : cdict[key] for key in cdict if (key in GibbsSampler.__init__.__code__.co_varnames)}
gibbs_sampler = GibbsSampler(3000, Lclass, **cdict)
gibbs_sampler.run()


def get_lag(id, kmax, accepted):
    a = accepted[:, id]
    a_avg = np.mean(a)
    N = len(a)
    denom = np.sum((a-a_avg)**2)
    lag = []
    for k in range(1, kmax + 1):
        num = 0
        for i in range(0, N - k):
            num += (a[i] - a_avg) * (a[i+k]-a_avg)
        lag.append(num)
    lag = np.array(lag)

    return(lag / denom)
