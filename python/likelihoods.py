import numpy as np

class Likelihood:

    def __init__(self, data):

        self.data = data

    def logprior(self, **pars):

        for key in pars.keys():
            if pars[key] <= 0:
                return(-np.inf)

        return(1)

    def logprob(self, **pars):

        return(np.sum(-np.log(pars["sig"] * np.sqrt(2 * np.pi)) - ((self.data - pars["mu"]) ** 2) / (2 * pars["sig"] ** 2)))

class LikelihoodFg:

    def __init__(self, data, invcov):

        self.data = data
        self.invcov = invcov

    def logprior(self, **pars):

        if pars["Aplanck"] < 0.958 or pars["Aplanck"] > 1.041:
            return(-np.inf)
        if np.abs(pars["c0"]) > 0.028:
            return(-np.inf)
        if np.abs(pars["c1"]) > 0.031:
            return(-np.inf)
        if np.abs(pars["c3"]) > 0.024:
            return(-np.inf)
        if np.abs(pars["c4"]) > 0.028:
            return(-np.inf)
        if np.abs(pars["c5"]) > 0.028:
            return(-np.inf)
        if pars["Aradio"] < 0.29 or pars["Aradio"] > 3.10:
            return(-np.inf)
        if pars["Adusty"] < 0 or pars["Adusty"] > 2:
            return(-np.inf)
        if pars["AdustTT"] < 0 or pars["AdustTT"] > 2:
            return(-np.inf)
        if pars["AdustPP"] < 0 or pars["AdustPP"] > 2.5:
            return(-np.inf)
        if pars["AdustTP"] < 0 or pars["AdustTP"] > 2.5:
            return(-np.inf)
        if pars["Asz"] < 0 or pars["Asz"] > 3.88:
            return(-np.inf)
        if pars["Acib"] < 0 or pars["Acib"] > 3:
            return(-np.inf)
        if pars["Aksz"] < 0 or pars["Aksz"] > 10:
            return(-np.inf)
        if pars["Aszxcib"] < 0 or pars["Aszxcib"] > 10:
            return(-np.inf)
            
        return(1)

    def logprob(self, C_CMB, vec_fg, invcov):
        res = (self.data - (vec_fg + C_CMB))

        return(-0.5 * res.dot(invcov).dot(res))
