import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from corr_coeff_utils import compute_spectra, set_multipole_range, read_covmat, set_lists, select_covblock
import os
def select_spectra_unbin(cl, nmap, nfreq,
                         frequencies,
                         multipole_range, mode):

    acl = np.asarray(cl)
    xl = []
    elll = []
    nxfreq = nfreq * (nfreq + 1) // 2
    for xf in range(nxfreq):
        lmin = multipole_range[1][mode][set_lists(nmap, nfreq, frequencies).index(xf)]
        lmax = multipole_range[2][mode][set_lists(nmap, nfreq, frequencies).index(xf)]
        xl_temp = acl[xf, :]
        ell = np.arange(len(xl_temp))
        xl.append(xl_temp[lmin: lmax + 1])
        elll.append(ell[lmin: lmax + 1])
    return(elll, xl)

def compute_spectra_unbin(nmap, nfreq, frequencies, mode,
                          multipole_range, spectra_path):

    from corr_coeff_utils import read_dl_xspectra, get_spectra, xspectra2xfreq
    nxfreq = nfreq * (nfreq + 1) // 2
    nxspec = nmap * (nmap - 1) // 2

    dlsig = read_dl_xspectra(spectra_path, nmap, multipole_range, field = 2)
    dlsig[dlsig == 0] = np.inf
    dlweight = 1.0 / dlsig ** 2
    data = read_dl_xspectra(spectra_path, nmap, multipole_range)
    if mode != 2:
        data = get_spectra(data, nxspec, mode = mode)
        data = xspectra2xfreq(data, dlweight[mode], nmap,
                              nfreq, frequencies,
                              multipole_range, normed = True)
        ell, spectra = select_spectra_unbin(data, nmap,
                                            nfreq, frequencies,
                                            multipole_range,
                                            mode = mode)
    else:

        Rl = 0
        Wl = 0
        dataTE = get_spectra(data, nxspec, mode = 2)

        RlTE, WlTE = xspectra2xfreq(dataTE, dlweight[2], nmap,
                                    nfreq, frequencies, multipole_range,
                                    normed = False)


        Rl = Rl + RlTE
        Wl = Wl + WlTE
        dataET = get_spectra(data, nxspec, mode = 3)
        RlET, WlET = xspectra2xfreq(dataET, dlweight[3], nmap,
                                    nfreq, frequencies, multipole_range,
                                    normed = False)

        Rl = Rl + RlET
        Wl = Wl + WlET

        ell, spectra = select_spectra_unbin(Rl / Wl, nmap,
                                            nfreq, frequencies,
                                            multipole_range,
                                            mode = mode)

    return(ell, spectra)

def read_covmat_unbin(xf1, xf2, block,
                      nmap, nfreq, frequencies,
                      lmax, cov_path, multipole_range):

    # Read the covmat fits file
    fname = cov_path + "fll_NPIPE_detset_TEET_{}_{}.fits".format(xf1, xf2)
    covmat = fits.getdata(fname, 0).field(0)
    N = int(np.sqrt(len(covmat)))
    covmat = covmat.reshape(N, N)
    covmat = select_covblock(covmat, block = block)
    covmat = covmat[:lmax +1, :lmax + 1]

    # Apply the good multipole cuts to binned covmat
    modes = ["TT", "EE", "TE"]
    mode1 = block[:2]
    mode2 = block[2:]
    id1 = modes.index(mode1)
    id2 = modes.index(mode2)
    mlt_range = multipole_range
    lmin1 = mlt_range[1][id1][set_lists(nmap, nfreq, frequencies).index(xf1)]
    lmax1 = mlt_range[2][id1][set_lists(nmap, nfreq, frequencies).index(xf1)]
    lmin2 = mlt_range[1][id2][set_lists(nmap, nfreq, frequencies).index(xf2)]
    lmax2 = mlt_range[2][id2][set_lists(nmap, nfreq, frequencies).index(xf2)]

    covmat = covmat[lmin1:lmax1+1, lmin2:lmax2+1]

    return(covmat)

data_path = "../modules/data/Hillipop/"
binning_path = data_path + "Binning/"
spectra_path = data_path + "Data/NPIPE/spectra/"
cov_path = data_path + "Data/NPIPE/pclcvm/fll/"
nmap = 6
nfreq = 3
frequencies = [100, 100, 143, 143, 217, 217]
b1 = np.arange(50, 800, step = 30)
b2 = np.arange(np.max(b1)+60, 1200, step = 60)
b3 = np.arange(np.max(b2) + 100, 9000, step = 100)
binning = np.concatenate((b1, b2, b3))

multipole_range = set_multipole_range(binning_path)
TTells, TTspecs, TTlims = compute_spectra(nmap, nfreq, frequencies, 0,
                                          multipole_range, spectra_path,
                                          binning)
EEells, EEspecs, EElims = compute_spectra(nmap, nfreq, frequencies, 1,
                                          multipole_range, spectra_path,
                                          binning)
TEells, TEspecs, TElims = compute_spectra(nmap, nfreq, frequencies, 2,
                                          multipole_range, spectra_path,
                                          binning)

TTells_unbin, TTspecs_unbin = compute_spectra_unbin(nmap, nfreq, frequencies, 0,
                                        multipole_range, spectra_path)
EEells_unbin, EEspecs_unbin = compute_spectra_unbin(nmap, nfreq, frequencies, 1,
                                        multipole_range, spectra_path)
TEells_unbin, TEspecs_unbin = compute_spectra_unbin(nmap, nfreq, frequencies, 2,
                                        multipole_range, spectra_path)
for i in range(len(TTspecs[0])):
    print(TTspecs[0][i])
    print(np.mean(TTspecs_unbin[0][i*30:(i+1)*30]))

nxfreq = nfreq * (nfreq + 1) // 2
#covTT = []
#covEE = []
covTE = []
#covTT_unbin = []
#covEE_unbin = []
covTE_unbin = []
for i in range(nxfreq):
    #covTT.append(read_covmat(i, i, binning, "TTTT",
     #                        nmap, nfreq, frequencies,
      #                       2500, cov_path, multipole_range).diagonal())
    #covEE.append(read_covmat(i, i, binning, "EEEE",
     #                        nmap, nfreq, frequencies,
      #                       2500, cov_path, multipole_range).diagonal())
    covTE.append(read_covmat(i, i, binning, "TETE",
                             nmap, nfreq, frequencies,
                             2500, cov_path, multipole_range).diagonal())

    #covTT_unbin.append(read_covmat_unbin(i, i, "TTTT",
     #                        nmap, nfreq, frequencies,
      #                       2500, cov_path, multipole_range).diagonal())
    #covEE_unbin.append(read_covmat_unbin(i, i, "EEEE",
                             #nmap, nfreq, frequencies,
                             #2500, cov_path, multipole_range).diagonal())
    covTE_unbin.append(read_covmat_unbin(i, i, "TETE",
                             nmap, nfreq, frequencies,
                             2500, cov_path, multipole_range).diagonal())
output_path = "outputs/data/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

for i in range(nxfreq):
    #TT = np.array([TTells_unbin[i], TTspecs_unbin[i], covTT_unbin[i]])
    #EE = np.array([EEells_unbin[i], EEspecs_unbin[i], covEE_unbin[i]])
    TE = np.array([TEells_unbin[i], TEspecs_unbin[i], covTE_unbin[i]])

    #TTb = np.array([TTells[i], TTspecs[i], covTT[i]])
    #EEb = np.array([EEells[i], EEspecs[i], covEE[i]])
    TEb = np.array([TEells[i], TEspecs[i], covTE[i]])

    #np.savetxt(output_path + "tt{}.dat".format(i), TT)
    #np.savetxt(output_path + "ee{}.dat".format(i), EE)
    np.savetxt(output_path + "te{}.dat".format(i), TE)
    #np.savetxt(output_path + "ttb{}.dat".format(i), TTb)
    #np.savetxt(output_path + "eeb{}.dat".format(i), EEb)
    np.savetxt(output_path + "teb{}.dat".format(i), TEb)
