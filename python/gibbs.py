import numpy as np
import csv
from get_corr_coeff_cmb_only import get_full_fg_vec, get_Cb_and_Q
from corr_coeff_utils import *
class GibbsSampler:

    def __init__(self, Nsteps, savepath, multipole_range, binning,
                 pars, init_cmb, data, invcov, A, logprob, logprior):

        ## MCMC
        self.Nsteps = Nsteps
        self.savepath = savepath

        # DATA
        self.data = data
        self.invcov = invcov
        self.A = A

        # PARAMETERS
        self.par_keys = list(pars.keys())
        self.init_pars = {}
        self.proposals = {}
        for parname in self.par_keys:

            self.init_pars[parname] = pars[parname]['init']
            self.proposals[parname] = pars[parname]['proposal']

        self.init_cmb = init_cmb

        # LIKELIHOOD FUNCTION
        self.logprob = logprob
        self.logprior = logprior

        # FOREGROUNDS COMPUTATION
        self.nmap = 6
        self.nfreq = 3
        self.frequencies = [100,100,143,143,217,217]
        self.lmax = 2500
        self.data_path = "../../hillipop/modules/data/Hillipop/"
        self.spectra_path = self.data_path + "Data/NPIPE/spectra/"
        self.multipole_range = multipole_range
        self.binning = binning
        self.fgs = get_foregrounds(self.data_path, self.lmax, self.frequencies)
        self.dlweight = read_dl_xspectra(self.spectra_path, self.nmap, 
                                         self.multipole_range, field = 2)
        self.dlweight[self.dlweight == 0] = np.inf
        self.dlweight = 1.0 / self.dlweight ** 2
    
        with open(self.savepath + "chains3.dat", "w") as file:
            writer = csv.writer(file)
            writer.writerow(self.par_keys + ["logp"])


    def acceptance(self, logp, logp_new):

        if logp_new > logp:

            return True
        else:

            accept = np.random.uniform(0, 1)
            #print(accept<(np.exp(logp_new-logp)))
            return(accept < (np.exp(logp_new - logp)))

    def proposal(self, last_point):

        prop_dict = {}
        for parname in self.par_keys:

            mean = last_point[parname]
            std = self.proposals[parname]
            prop_dict[parname] = np.random.normal(mean, std)

        return(prop_dict)

    def run(self):
       
        accepted = []
        cmb_accepted = []
        current_point = self.init_pars
        current_fgpars = current_point.copy()
        current_fgpars["c2"] = 0
        current_fg  = get_full_fg_vec(current_fgpars, self.fgs, self.dlweight,
                                      self.multipole_range, self.nmap, 
                                      self.nfreq,self.frequencies, self.binning)
        current_cmb = self.init_cmb
        accep_rate = 0
        print("Running sampling for %d steps ..." % self.Nsteps)
        for i in range(self.Nsteps):
            #import time
            #st=time.time()
            if i != 0 and (i%(self.Nsteps/10) == 0 or i == self.Nsteps - 1):
                print("Accepted samples : %d" % accep_rate)
                print("Acceptance rate : %.03f" % (accep_rate / i))
                print("Current logL : %.05f" % (current_like))
            #st=time.time()

            new_point = self.proposal(current_point)
            new_fgpars = new_point.copy()
            new_fgpars["c2"] = 0
            #st=time.time()
            new_fg = get_full_fg_vec(new_fgpars, self.fgs, self.dlweight,
                                     self.multipole_range, self.nmap,
                                     self.nfreq, self.frequencies, self.binning)
            #print("new fg : {}".format(time.time()-st))
            
            current_like = self.logprob(self.A.dot(current_cmb), current_fg, self.invcov)
            current_prior = self.logprior(**current_point)
            #new_like = self.logprob(self.A.dot(new_cmb), new_fg, self.invcov)
            new_like = self.logprob(self.A.dot(current_cmb), new_fg, self.invcov)
            Cb, Q = get_Cb_and_Q(self.A, self.data, new_fg, self.invcov)
            #print("CbQ : {}".format(time.time()-st))
            #st=time.time()
            sqQ = svd_pow(Q, 0.5)
            #print("sqQ : {}".format(time.time()-st))
            new_cmb = Cb + sqQ.dot(np.random.randn(len(Cb)))
            new_prior = self.logprior(**new_point)
            #print(new_point['Aksz'], new_prior)
            #print(current_like+current_prior, new_like+new_prior)
            if (self.acceptance(current_like + current_prior,
                                new_like + new_prior)):

                current_point = new_point
                current_fg = new_fg
                current_cmb = new_cmb
                vec = [new_point[key] for key in new_point.keys()]
                accepted.append(vec + [new_like])
                cmb_accepted.append(new_cmb)
                accep_rate += 1
            
            else:
                vec = [current_point[key] for key in current_point.keys()]
                accepted.append(vec + [current_like])
                cmb_accepted.append(current_cmb)
            #print(time.time() - st)

        with open(self.savepath + "chains3.dat", "a") as file:
            writer = csv.writer(file)
            for line in accepted:
                writer.writerow(line)
        with open(self.savepath + "cmb3.dat", "w") as file:
            writer = csv.writer(file)
            for line in cmb_accepted:
                writer.writerow(line)
