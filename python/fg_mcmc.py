import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from likelihoods import LikelihoodFg
from gibbs import GibbsSampler
from corr_coeff_utils import compute_spectra, set_multipole_range
import os


data_path = "../../hillipop/modules/data/Hillipop/"
spectra_path = data_path + "Data/NPIPE/spectra/"
output_path = "../python_products/"
gibbs_path = output_path + "gibbs_inputs/"


multipole_range_file = "../hillipop/modules/data/planck_2020/hillipop/data/binning_lowl_extended.fits"
invcov_file = output_path + "cov/invfll_NPIPE_detset_extl_TTTEEE_bin.fits"
binning_file = "../python_products/binning_corr_coeff.fits"

invcov = fits.getdata(invcov_file, hdu=0).field(0)
N = int(np.sqrt(len(invcov)))
invcov = invcov.reshape(N, N)
invcov *= 1e-24 # K to muK

binning = fits.getdata(binning_file ,hdu = 0).field(0)

multipole_range = set_multipole_range(multipole_range_file)

nmap = 6
nfreq = 3
frequencies = [100,100,143,143,217,217]

ellTT, CellTT, limsTT = compute_spectra(nmap, nfreq, frequencies, 0,
                                        multipole_range, spectra_path, binning)
ellEE, CellEE, limsEE = compute_spectra(nmap, nfreq, frequencies, 1,
                                        multipole_range, spectra_path, binning)
ellTE, CellTE, limsTE = compute_spectra(nmap, nfreq, frequencies, 2,
                                        multipole_range, spectra_path, binning)

vec_data = np.concatenate((np.concatenate(CellTT),
                           np.concatenate(CellEE),
                           np.concatenate(CellTE)))

init_cmb = np.loadtxt(gibbs_path + "Cb_example.dat")
A = np.loadtxt(gibbs_path + "A.dat")
Lclass = LikelihoodFg(vec_data, invcov)
logL = Lclass.logprob
logP = Lclass.logprior

parameters = {"Aplanck": {"init": 1., "proposal": 0.0013/2},# 0.0083},
              "c0": {"init": 0.0, "proposal": 0.0009/2}, #0.0055},
              "c1": {"init": 0.0, "proposal": 0.0009/2}, #0.0055},
              "c3": {"init": 0.0, "proposal": 0.0008/2}, #0.0048},
              "c4": {"init": 0.0, "proposal": 0.0009/2}, #0.0055},
              "c5": {"init": 0.0, "proposal": 0.0009/2}, #0.0055},
              "Aradio": {"init": 1.61984, "proposal": 0.04/2}, #0.2808},#1.61984
              "Adusty": {"init": 0.781192, "proposal": 0.03/2}, #0.2427},#0.781192
              "AdustTT": {"init": 1, "proposal": 0.03/2},#0.2309},#1
              "AdustPP": {"init": 1., "proposal": 0.05/2},#0.3277},#1
              "AdustTP": {"init": 1., "proposal": 0.05/2},#0.3501},#1
              "Asz": {"init": 1., "proposal": 0.08/2},#0.5448},#1
              "Acib": {"init": 1., "proposal": 0.05/2},#0.3377},#1
              "Aksz": {"init": 0.6, "proposal": 0.03/2},
              "Aszxcib": {"init": 1, "proposal": 0.03/2}}

output_chains = "../chains/"
if not os.path.exists(output_chains):
    os.makedirs(output_chains)
gibbs_sampler = GibbsSampler(3000, output_chains, multipole_range,
                             binning, parameters, init_cmb,
                             vec_data, invcov, A, logL, logP)
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
