from corr_coeff_utils import read_covmat, set_multipole_range
import astropy.io.fits as fits
import numpy as np
import os

def produce_cov_pattern(nfreq, modes):

    nxfreq = nfreq * (nfreq + 1) // 2
    vec = []
    for mode in modes:

        for i in range(nxfreq):

            vec.append((mode,str(i)))

    cov = []
    for doublet1 in vec:

        line = []
        for doublet2 in vec:

            m1, xf1 = doublet1
            m2, xf2 = doublet2
            line.append((m1+m2, xf1+xf2))

        cov.append(line)

    return(cov)

def compute_full_covmat(modes, binning, nmap, nfreq, frequencies,
                        lmax, cov_path, multipole_range):

    print("Creating pattern for the full covariance matrix ...")
    pattern_cov = produce_cov_pattern(nfreq, modes)
    full_cov = []
    print("Starting to fill the matrix ...")
    for i in range(len(pattern_cov)):

        line = []
        for j in range(len(pattern_cov)):
            print(pattern_cov[i][j])
            mode, xfreq_couple = pattern_cov[i][j]
            xf1, xf2 = xfreq_couple
            xf1, xf2 = int(xf1), int(xf2)
            m1, m2 = mode[:2], mode[2:]
            if xf1 <= xf2:
                import time
                stime = time.time()
                line.append(
                    read_covmat(xf1, xf2, binning, mode,
                                nmap, nfreq, frequencies, lmax,
                                cov_path, multipole_range))
                print(time.time()-stime)
            else:
                import time
                stime = time.time()
                line.append(
                    read_covmat(xf2, xf1, binning, m2 + m1,
                                nmap, nfreq, frequencies, lmax,
                                cov_path, multipole_range).T)
                print(time.time()-stime)
        full_cov.append(line)

    full_cov = np.block(full_cov)
    print("End of computation.")

    return(full_cov)


data_path = '../../hillipop/modules/data/Hillipop/'
binning_path = '../hillipop/modules/data/planck_2020/hillipop/data/binning_lowl_extended.fits'
spectra_path = data_path + 'Data/NPIPE/spectra/'
cov_path = data_path + 'Data/NPIPE/pclcvm/fll/'

output_path = "../python_products/cov/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# save paths for binning & covariance matrix
save_bin_invcov = output_path + 'invfll_NPIPE_detset_extl_TT_bin.fits'
save_bin_cov = output_path + 'fll_NPIPE_detset_extl_TT_bin.fits'
save_binning = output_path + '../binning_corr_coeff.fits'


nmap = 6
nfreq = 3
frequencies = [100, 100, 143, 143, 217, 217]

#b1 = np.arange(30, 800, step = 30)
#b2 = np.arange(np.max(b1)+60, 1200, step = 60)
#b3 = np.arange(np.max(b2)+100, 9000, step = 100)
b1 = np.arange(30,1500,step=30)
b2 = np.arange(np.max(b1)+60,9000, step = 60)
binning = np.concatenate((b1, b2))
col = [fits.Column(name = 'binning', format = 'D', array = binning)]
hdulist=fits.BinTableHDU.from_columns(col)
hdulist.writeto(save_binning, overwrite=True)

modes = ["TT"]
lmax = 2500

multipole_range = set_multipole_range(binning_path)
fcov = compute_full_covmat(modes, binning, nmap, nfreq, frequencies,
                           lmax, cov_path, multipole_range)
col = [fits.Column(name = 'fullcov', format = 'D', array = fcov.reshape(len(fcov) * len(fcov)))]
hdulist=fits.BinTableHDU.from_columns(col)
hdulist.writeto(save_bin_cov, overwrite=True)

invfcov = np.linalg.inv(fcov) * 1e24 ## Input of likelihood
col = [fits.Column(name = 'invfullcov', format = 'D', array = invfcov.reshape(len(invfcov) * len(invfcov)))]
hdulist=fits.BinTableHDU.from_columns(col)
hdulist.writeto(save_bin_invcov, overwrite=True)
