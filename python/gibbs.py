import numpy as np
import csv
from get_corr_coeff_cmb_only import get_Cb_and_Q
from corr_coeff_utils import *

class GibbsSampler:

    def __init__(self, Nsteps, Likelihood, chains_path, gibbs_path, output_path, A_file,
                 init_cmb_file, parameters):

        ## MCMC
        self.Nsteps = Nsteps
        self.savepath = chains_path
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        # Data
        self.A = A
        self.init_cmb = np.loadtxt(output_path + gibbs_path + init_cmb_file)
        A = np.loadtxt(output_path + gibbs_path + A_file)
        # Parameters
        self.init_pars = {key : parameters[key]["init"] for key in parameters}
        self.proposals = {key : parameters[key]["proposal"] for key in parameters}
        self.par_keys = list(parameters.keys())


        self.init_cmb = init_cmb

        # Likelihoods and priors
        self.Likelihood = Likelihood
        self.logprob = Likelihood.logprob
        self.logprior = Likelihood.logprior

        with open(self.savepath + "chains.dat", "w") as file:
            writer = csv.writer(file)
            writer.writerow(self.par_keys + ["logp"])


    def acceptance(self, logp, logp_new):

        if logp_new > logp:

            return True
        else:

            accept = np.random.uniform(0, 1)
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
        accept_rates = []

        current_point = self.init_pars
        current_fgpars = current_point.copy()
        current_fgpars["c2"] = 0
        current_fg  = get_full_fg_vec(current_fgpars, self.Likelihood.fgs,
                                      self.Likelihood.dlweight,
                                      self.Likelihood.multipole_range,
                                      self.Likelihood.nmap,
                                      self.Likelihood.nfreq,
                                      self.Likelihood.frequencies,
                                      self.Likelihood.binning)
        current_cmb = self.init_cmb

        accep_count = 0

        print("Running sampling for %d steps ..." % self.Nsteps)
        for i in range(self.Nsteps):

            if i != 0 and (i%(self.Nsteps/10) == 0 or i == self.Nsteps - 1):

                print("Accepted samples : %d" % accep_rate)
                print("Acceptance rate : %.03f" % (accep_rate / i))
                print("Current logL : %.05f" % (current_like))

            new_point = self.proposal(current_point)

            # Current state (only for i == 0)
            if i == 0:

                current_like, _ = self.logprob(self.A.dot(current_cmb),
                                               **current_point)
                current_prior = self.logprior(**current_point)

            # New state (from current state CMB)
            new_like, new_fg = self.logprob(self.A.dot(current_cmb), **new_point)
            new_prior = self.logprior(**new_point)

            if (self.acceptance(current_like + current_prior,
                                new_like + new_prior)):

                # Compute new cmb state only if the new fg sample was accepted
                # (otherwise we already have the current cmb)
                Cb, Q = get_Cb_and_Q(self.A, self.Likelihood.data,
                                     new_fg, self.Likelihood.invcov)
                sqQ = svd_pow(Q, 0.5)
                new_cmb = Cb + sqQ.dot(np.random.randn(len(Cb)))

                # Set accepted point to current point
                current_point = new_point
                current_fg = new_fg
                current_cmb = new_cmb

                # Store accepted samples
                vec = [new_point[key] for key in new_point.keys()]
                accepted.append(vec + [new_like])
                cmb_accepted.append(new_cmb)
                accep_count += 1
            else:

                # Store current sample if new was rejected
                vec = [current_point[key] for key in current_point.keys()]
                accepted.append(vec + [current_like])
                cmb_accepted.append(current_cmb)

            accept_rates.append(accep_count / (i + 1))

        # Saving accepted samples in files
        with open(self.savepath + "chains.dat", "a") as file:
            writer = csv.writer(file)
            for line in accepted:
                writer.writerow(line)

        with open(self.savepath + "cmb.dat", "w") as file:
            writer = csv.writer(file)
            for line in cmb_accepted:
                writer.writerow(line)

        np.savetxt(self.savepath + "acceptance.dat", accept_rates)
