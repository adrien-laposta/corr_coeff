import numpy as np
from get_corr_coeff_cmb_only import get_full_fg_vec
from corr_coeff_utils import *

class LikelihoodFg:

    def __init__(self, data_path, invcov_file, output_path, spectra_path,
                 multipole_range_file, binning_file, parameters):
        
        print("[likelihood] Initializing ...")
        # Variables
        self.nmap = 6
        self.nfreq = 3
        self.frequencies = [100,100,143,143,217,217]
        self.lmax = 2500

        # Binning
        self.binning = fits.getdata(output_path + binning_file, hdu = 0).field(0)

        # Multipole ranges
        self.multipole_range = set_multipole_range(multipole_range_file)

        # Data
        _, CellTT, _ = compute_spectra(self.nmap, self.nfreq,
                                                self.frequencies, 0,
                                                self.multipole_range,
                                                data_path + spectra_path,
                                                self.binning)

        _, CellEE, _ = compute_spectra(self.nmap, self.nfreq,
                                                self.frequencies, 1,
                                                self.multipole_range,
                                                data_path + spectra_path,
                                                self.binning)

        _, CellTE, _ = compute_spectra(self.nmap, self.nfreq,
                                                self.frequencies, 2,
                                                self.multipole_range,
                                                data_path + spectra_path,
                                                self.binning)

        self.data = np.concatenate((np.concatenate(CellTT),
                                   np.concatenate(CellEE),
                                   np.concatenate(CellTE)))

        # Load invcovmat
        self.invcov = fits.getdata(output_path + invcov_file, hdu=0).field(0)
        N = int(np.sqrt(len(self.invcov)))
        self.invcov = self.invcov.reshape(N, N) * 1e-24


        # Foregrounds computation
        self.fgs = get_foregrounds(data_path, self.lmax, self.frequencies)
        self.dlweight = read_dl_xspectra(data_path + spectra_path, self.nmap,
                                         self.multipole_range, field = 2)
        self.dlweight[self.dlweight == 0] = np.inf
        self.dlweight = 1.0 / self.dlweight ** 2

        # Priors
        self.priors = {key : [float(parameters[key]["prior"]["min"]),
                              float(parameters[key]["prior"]["max"])
                             ] for key in parameters}

    def logprior(self, **pars):

        for key in pars:
            if (pars[key] < self.priors[key][0]) or (
                pars[key] > self.priors[key][1]):

                return(-np.inf)

        return(1)

    def logprob(self, fg_vec, C_CMB, **pars):
        
        if fg_vec is None:
            fgpars = pars.copy()
            fgpars["c2"] = 0
            fg_vec = get_full_fg_vec(fgpars, self.fgs, self.dlweight,
                                     self.multipole_range, self.nmap,
                                     self.nfreq, self.frequencies, self.binning)

        res = (self.data - (fg_vec + C_CMB))

        return(-0.5 * res.dot(self.invcov).dot(res), fg_vec)
