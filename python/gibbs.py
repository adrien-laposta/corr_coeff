import numpy as np
import csv
from get_corr_coeff_cmb_only import get_Cb_and_Q, get_full_fg_vec
from corr_coeff_utils import *

class GibbsSampler:

    def __init__(self, Nsteps, Likelihood, resume, chains_path,
                 gibbs_path, output_path, A_file, init_cmb_file,
                 cov_fg_file, mode, parameters):
        
        print("[gibbs] Initializing ...")
        ## MCMC
        self.Nsteps = Nsteps
        self.savepath = chains_path
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        self.resume = resume
        if self.resume:
            print("[gibbs] Resuming from old samples ...")

        # Parameters
        self.sampled_par_keys = []
        self.fixed_par_keys = []
        for key in parameters:
            if "value" in parameters[key]:
                self.fixed_par_keys.append(key)
            else:
                self.sampled_par_keys.append(key)
        print("[gibbs] Sampled parameters : {}".format(self.sampled_par_keys))
        print("[gibbs] Fixed parameters : {}".format(self.fixed_par_keys))
        if self.resume:
            with open(self.savepath + "chains.dat", "r") as file:
                reader = csv.reader(file, delimiter = ',')
                data_read = [row for row in reader]
            labs = np.array(data_read[0]).reshape(len(data_read[0]),)[:-1]
            data_read = data_read[1:]
            data = np.array([np.array(line, dtype="float") for line in data_read])[-1]
            cdict = {}
            for i, lab in enumerate(labs):
                cdict[lab] = data[i]
            self.init_pars = cdict
        else:
            self.init_pars = {key : float(parameters[key]["init"]) for key in self.sampled_par_keys}

        self.proposals = {key : float(parameters[key]["proposal"]) for key in self.sampled_par_keys}
        self.fixed = {key: float(parameters[key]["value"]) for key in self.fixed_par_keys}

        # Likelihoods and priors
        self.Likelihood = Likelihood
        self.logprob = Likelihood.logprob
        self.logprior = Likelihood.logprior

        # Data
        self.mode = mode
        self.A = np.loadtxt(output_path + gibbs_path + A_file)
        if self.resume:
            with open(self.savepath + "cmb.dat", "r") as file:
                reader = csv.reader(file, delimiter = ",")
                data_read = [row for row in reader][-1]
            data = np.array(data_read, dtype = "float")
            self.init_cmb = data
        else:
            self.init_cmb = np.loadtxt(output_path + gibbs_path + init_cmb_file)
        self.invQ = (self.A.T).dot(self.Likelihood.invcov).dot(self.A)
        self.Q = np.linalg.inv(self.invQ)
        self.sqQ = svd_pow(self.Q, 0.5)
        try:
            cov_pars = ["Aplanck", "c0", "c1", "c3", "c4",
                        "c5", "Aradio", "Adusty", "AdustTT", 
                        "AdustPP", "AdustTP", "Asz", "Acib",
                        "Aksz", "Aszxcib"]
            cov = np.loadtxt(output_path + gibbs_path + cov_fg_file)
            cov_dict = {}
            for i, k1 in enumerate(cov_pars):
                for j, k2 in enumerate(cov_pars):
                    
                    cov_dict[(k1, k2)] = cov[i, j]
                        
            self.cov_fg = []
            for i, k1 in enumerate(self.sampled_par_keys):
                line = []
                for j, k2 in enumerate(self.sampled_par_keys):
                    line.append(cov_dict[k1, k2])
                self.cov_fg.append(line)
            self.cov_fg = np.array(self.cov_fg)
            #self.cov_fg = np.loadtxt(output_path + gibbs_path + cov_fg_file)
            print("[gibbs] Proposals will be drawn from given parameter covariance matrix")
        except:
            self.cov_fg = 'None'
            print("[gibbs] Proposals will be drawn independantly")

        if not self.resume:
            with open(self.savepath + "chains.dat", "w") as file:
                writer = csv.writer(file)
                writer.writerow(self.sampled_par_keys + ["logp"])
            with open(self.savepath + "cmb.dat", "w") as file:
                pass


    def acceptance(self, logp, logp_new):

        if logp_new > logp:

            return True
        else:

            accept = np.random.uniform(0, 1)
            return(accept < (np.exp(logp_new - logp)))

    def proposal(self, last_point):

        prop_dict = {}
        
        if self.cov_fg == "None":
            for parname in self.sampled_par_keys:

                mean = last_point[parname]
                std = self.proposals[parname]
                prop_dict[parname] = np.random.normal(mean, std)

        else:
            mean = [last_point[parname] for parname in self.sampled_par_keys]
            rcov = pow(2.38, 2) * self.cov_fg / len(mean)
            prop_vec = mean + svd_pow(rcov, 0.5).dot(np.random.randn(len(mean)))
            for i, parname in enumerate(self.sampled_par_keys):

                prop_dict[parname] = prop_vec[i]

        return(prop_dict)

    def run(self):

        accepted = []
        cmb_accepted = []
        if self.resume:
            accept_rates = list(np.loadtxt(self.savepath + "acceptance.dat"))
            Nresume = len(accept_rates)
        else:
            accept_rates = []

        current_point = self.init_pars
        current_fgpars = current_point.copy()
        current_fgpars = {**current_fgpars, **self.fixed}
        current_fg, current_gamma = get_full_fg_vec(current_fgpars, self.Likelihood.fgs,
                                                    self.Likelihood.dlweight,
                                                    self.Likelihood.multipole_range,
                                                    self.Likelihood.nmap,
                                                    self.Likelihood.nfreq,
                                                    self.Likelihood.frequencies,
                                                    self.Likelihood.binning,
                                                    self.mode)
        np.savetxt("/sps/litebird/Users/alaposta/development/corr_coeff/python_products/gibbs_inputs/fg_init_vec.dat", current_fg)
        current_cmb = self.init_cmb
        import time
        st=time.time()
        if self.resume:
            accep_count = int(Nresume * accept_rates[-1])
        else:
            accep_count = 0
        print("[gibbs] Running sampling for %d steps ..." % self.Nsteps)
        for i in range(self.Nsteps):

            if i != 0 and (i%(self.Nsteps/10) == 0 or i == self.Nsteps - 1):

                print("[gibbs] Accepted samples : %d" % accep_count)
                if self.resume:
                    print("[gibbs] Acceptance rate : %.03f" % (accep_count / int(i + Nresume)))
                else:
                    print("[gibbs] Acceptance rate : %.03f" % (accep_count / i))
                print("[gibbs] Current logL : %.05f" % (current_like))

            new_point = self.proposal(current_point)
       
            # Current state 
            current_like, _, _ = self.logprob(fg_vec = current_fg, 
                                              C_CMB = self.A.dot(current_cmb),
                                              **current_point)
            print(current_like)
            current_prior = self.logprior(**current_point)

            # New state (from current state CMB)
            new_like, new_fg, new_gamma = self.logprob(fg_vec = None,
                                            C_CMB = self.A.dot(current_cmb), 
                                            **{**new_point, **self.fixed})
            new_prior = self.logprior(**new_point)

            #print(new_like, current_like,self.acceptance(current_like + current_prior,new_like + new_prior))
            if (self.acceptance(current_like + current_prior,
                                new_like + new_prior)):
                
                # Compute new cmb state using current fg
                Cb, _ = get_Cb_and_Q(self.A, self.Likelihood.data,
                                     new_fg, self.Likelihood.invcov,
                                     get_Q=False, Q=self.Q)
                
                new_cmb = Cb + self.sqQ.dot(np.random.randn(len(Cb)))
                #new_cmb = current_cmb

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
                if i ==0:
                    Cb, _ = get_Cb_and_Q(self.A, self.Likelihood.data,
                                         current_fg, self.Likelihood.invcov,
                                         get_Q=False, Q=self.Q)
                new_cmb = Cb + self.sqQ.dot(np.random.randn(len(Cb)))
                #new_cmb = current_cmb
                current_cmb = new_cmb

                # Store current sample if new was rejected
                vec = [current_point[key] for key in current_point.keys()]
                accepted.append(vec + [current_like])
                #cmb_accepted.append(current_cmb)
                cmb_accepted.append(new_cmb)
            
            if self.resume:
                accept_rates.append(accep_count / (i + 1 + Nresume))
            else:
                accept_rates.append(accep_count / (i + 1))
        
            # Saving accepted samples in files every 500 steps
            # or at the end of computation
            if i!=0 and (i%500 == 0 or i == self.Nsteps-1):

                with open(self.savepath + "chains.dat", "a") as file:
                    writer = csv.writer(file)
                    for line in accepted:
                        writer.writerow(line)

                with open(self.savepath + "cmb.dat", "a") as file:
                    writer = csv.writer(file)
                    for line in cmb_accepted:
                        writer.writerow(line)
        

                np.savetxt(self.savepath + "acceptance.dat", accept_rates)
                accepted = []
                cmb_accepted = []

        print("TIME/NSTEPS = {}/{}".format(time.time()-st, self.Nsteps))
