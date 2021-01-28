import astropy.io.fits as fits
import numpy as np
from corr_coeff_utils import set_multipole_range, compute_spectra
from corr_coeff_utils import bin_spectra, get_foregrounds
from corr_coeff_utils import get_fg_xfreq_vec
import os

def get_ell_cmb(lims, binning):

    ellmin_cmb = np.min(np.array(lims)[:,0])
    ellmax_cmb = np.max(np.array(lims)[:,1])
    ell_cmb = np.arange(ellmin_cmb, ellmax_cmb+1)
    id_binning = np.where(binning < ellmax_cmb)
    binning_cmb = binning[id_binning]
    ell_cmb = bin_spectra(ell_cmb, ell_cmb, binning_cmb)[0]
    id = np.where((ell_cmb >= ellmin_cmb) & (ell_cmb <= ellmax_cmb))
    ell_cmb = ell_cmb[id]

    return(ell_cmb)

def get_A_matrix(ell_cmb, ell_data):

    ell_vec = np.concatenate((ell_data))
    A = []
    for i, ell in enumerate(ell_vec):
        bool_array = (ell_cmb == ell)
        line = list(bool_array.astype(float))
        A.append(line)
    A = np.array(A)

    return(A)

def get_full_A_matrix(ells, lims, binning):

    ellTT, ellEE, ellTE = ells
    limsTT, limsEE, limsTE = lims

    ell_cmbTT = get_ell_cmb(limsTT, binning)
    ell_cmbEE = get_ell_cmb(limsEE, binning)
    ell_cmbTE = get_ell_cmb(limsTE, binning)

    ell_cmb = np.concatenate((ell_cmbTT,
                              ell_cmbEE,
                              ell_cmbTE))

    ATT = get_A_matrix(ell_cmbTT, ellTT)
    AEE = get_A_matrix(ell_cmbEE, ellEE)
    ATE = get_A_matrix(ell_cmbTE, ellTE)

    from scipy.linalg import block_diag
    A = block_diag(ATT, AEE, ATE)

    return(ell_cmb, A)

def get_full_fg_vec(pars, fgs, spectra_path, multipole_range,
                    nmap, nfreq, frequencies, binning):

    _, specfgTT, _ = get_fg_xfreq_vec(pars, fgs, spectra_path, multipole_range,
                                      nmap, nfreq, frequencies, 0, binning)
    _, specfgEE, _ = get_fg_xfreq_vec(pars, fgs, spectra_path, multipole_range,
                                      nmap, nfreq, frequencies, 1, binning)
    _, specfgTE, _ = get_fg_xfreq_vec(pars, fgs, spectra_path, multipole_range,
                                      nmap, nfreq, frequencies, 2, binning)

    return(np.concatenate((np.concatenate(specfgTT),
                           np.concatenate(specfgEE),
                           np.concatenate(specfgTE))))

def get_Cb_and_Q(A, vec_data, vec_fg, invcov):

    invQ = (A.T).dot(invcov).dot(A)
    Q = np.linalg.inv(invQ)
    Cb = Q.dot(A.T).dot(invcov).dot(vec_data - vec_fg)

    return(Cb, Q)

    
a="""
# PATHS
data_path = "../../hillipop/modules/data/Hillipop/"
spectra_path = data_path + "Data/NPIPE/spectra/"
output_path = "../python_products/"

# FILES
multipole_range_file = "../hillipop/modules/data/planck_2020/hillipop/data/binning_lowl_extended.fits"
invcov_file = output_path + "cov/invfll_NPIPE_detset_extl_TTTEEE_bin.fits"
binning_file = "../python_products/binning_corr_coeff.fits"

# USEFUL PARAMETERS
nmap = 6
nfreq = 3
lmax = 2500
frequencies = [100, 100, 143, 143, 217, 217]

# CL COVARIANCE
invcov = fits.getdata(invcov_file, hdu=0).field(0)
N = int(np.sqrt(len(invcov)))
invcov = invcov.reshape(N, N)
invcov *= 1e-24 # K to muK

# BINNING FILE
binning = fits.getdata(binning_file ,hdu = 0).field(0)

# MULTIPOLE RANGE
multipole_range = set_multipole_range(multipole_range_file)

# READ BINNED PS
print("Reading PS from file ...")
ellTT, CellTT, limsTT = compute_spectra(nmap, nfreq, frequencies, 0,
                                        multipole_range, spectra_path, binning)
ellEE, CellEE, limsEE = compute_spectra(nmap, nfreq, frequencies, 1,
                                        multipole_range, spectra_path, binning)
ellTE, CellTE, limsTE = compute_spectra(nmap, nfreq, frequencies, 2,
                                        multipole_range, spectra_path, binning)

# EXAMPLE NUISANCE PARAMETERS
pars = {"Aplanck": 9.997e-1,
        "c0": 7.064e-4,
        "c1": 4.000e-3,
        "c2": 0,
        "c3": -6.344e-4,
        "c4": -1.432e-3,
        "c5": -2.863e-3,
        "Aradio": 1.698,
        "Adusty": 6.258e-1,
        "AdustTT": 8.649e-1,
        "AdustPP": 1.156,
        "AdustTP": 1.092,
        "Asz": 1.155,
        "Acib": 1.070,
        "Aksz": 1.24e-2,
        "Aszxcib": 1.264}

# A MATRIX
print("Get A matrix ...")
ell_cmb, A = get_full_A_matrix([ellTT, ellEE, ellTE],
                               [limsTT, limsEE, limsTE],
                               binning)

# SAVE INPUTS FOR GIBBS
gibbs_path = output_path + "gibbs_inputs/"
if not os.path.exists(gibbs_path):
    os.makedirs(gibbs_path)
np.savetxt(gibbs_path + "ell_cmb.dat", ell_cmb)
np.savetxt(gibbs_path + "A.dat", A)



fgs = get_foregrounds(data_path, lmax, frequencies)
vec_fg = get_full_fg_vec(pars, fgs, spectra_path, multipole_range,
                         nmap, nfreq, frequencies, binning)
vec_data = np.concatenate((np.concatenate(CellTT),
                           np.concatenate(CellEE),
                           np.concatenate(CellTE)))

# MEAN AND COVARIANCE COMPUTATION
print("Compute mean and covariance")
Cb, Q = get_Cb_and_Q(A, vec_data, vec_fg, invcov)
np.savetxt(gibbs_path + "Cb_example.dat", Cb)
np.savetxt(gibbs_path + "Q_example.dat", Q)

# SIMULATION TESTS
from corr_coeff_utils import svd_pow
NSIMS = 1000
sims = []
sqQ = svd_pow(Q, 0.5)
for i in range(NSIMS):
    sim = Cb + sqQ.dot(np.random.randn(len(Cb)))
    sims.append(sim)
sims = np.array(sims)
mean_sim = np.mean(sims, axis = 0)
std_sim = np.std(sims, axis = 0)

import matplotlib.pyplot as plt
plt.figure()


for i in range(NSIMS):
    plt.plot(ell_cmb, sims[i], color = "gray", alpha = 0.5)
plt.plot(ell_cmb, mean_sim, color = "tab:red", lw = 2)
plt.yscale('log')
plt.show()
"""
