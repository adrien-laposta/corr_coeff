import numpy as np
from get_corr_coeff_cmb_only import get_full_fg_vec
from corr_coeff_utils import *

class LikelihoodFg:

    def __init__(self, data_path, invcov_file, output_path, spectra_path,
                 multipole_range_file, binning_file, mode, parameters):
        
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
        
        self.mode = mode
        # Data
        if self.mode == "tt" or self.mode == "all":
            _, CellTT, _ = compute_spectra(self.nmap, self.nfreq,
                                           self.frequencies, 0,
                                           self.multipole_range,
                                           data_path + spectra_path,
                                           self.binning)
        if self.mode == "ee" or self.mode == "all":
            _, CellEE, _ = compute_spectra(self.nmap, self.nfreq,
                                           self.frequencies, 1,
                                           self.multipole_range,
                                           data_path + spectra_path,
                                           self.binning)
        if self.mode == "te" or self.mode == "all":
            _, CellTE, _ = compute_spectra(self.nmap, self.nfreq,
                                           self.frequencies, 2,
                                           self.multipole_range,
                                           data_path + spectra_path,
                                           self.binning)
        #if self.mode == "tt":
            #self.data = np.concatenate(CellTT)
        #elif self.mode == "ee":
            #self.data = np.concatenate(CellEE)
        #elif self.mode == "te":
            #self.data = np.concatenate(CellTE)
        #elif self.mode == "all":
            #self.data = np.concatenate((np.concatenate(CellTT),
            #                           np.concatenate(CellEE),
            #                           np.concatenate(CellTE)))
        self.data = np.loadtxt("/sps/litebird/Users/alaposta/development/corr_coeff/python_products/sims/sim_0.dat")[:318]
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
        self.sampled_par_keys = []
        for key in parameters:
            if not "value" in parameters[key]:
                self.sampled_par_keys.append(key)
        self.prior_type = {key : parameters[key]["prior"]["type"]
                           for key in self.sampled_par_keys}
        self.priors = {}
        for key in self.sampled_par_keys:
            if self.prior_type[key] == "flat":
                self.priors[key] = [float(parameters[key]["prior"]["min"]),
                                    float(parameters[key]["prior"]["max"])]
            elif self.prior_type[key] == "norm":
                self.priors[key] = [float(parameters[key]["prior"]["mean"]),
                                    float(parameters[key]["prior"]["std"])]

        #self.priors = {key : [float(parameters[key]["prior"]["min"]),
                              #float(parameters[key]["prior"]["max"])
                             #] for key in parameters}

    def logprior(self, **pars):
        
        def flat_logp(param, pmin, pmax):
            if (param < pmin) or (param > pmax):
                return(-np.inf)
            else:
                return(1)

        def norm_logp(param, mean, std):
            #return(-0.5*np.log(2*np.pi*pow(std,2)) - 0.5 * pow((param - mean)/std, 2))
            return(-0.5 * pow((param-mean)/std,2))
        log_prior = 0
        for key in pars:
            if self.prior_type[key] == "norm":
                log_prior += norm_logp(pars[key], self.priors[key][0], self.priors[key][1])
            elif self.prior_type[key] == "flat":
                log_prior += flat_logp(pars[key], self.priors[key][0], self.priors[key][1])

        return(log_prior)

    def logprob(self, fg_vec, C_CMB, **pars):
        
        if fg_vec is None:
            fg_vec = get_full_fg_vec(pars, self.fgs, self.dlweight,
                                     self.multipole_range, self.nmap,
                                     self.nfreq, self.frequencies, self.binning,
                                     self.mode)

        res = (self.data - (fg_vec + C_CMB))

        return(-0.5 * res.dot(self.invcov).dot(res), fg_vec)
