import astropy.io.fits as fits
import numpy as np
import os

def set_multipole_range(binning_path):
    """
    Read and return multipole cuts from
    binning.fits file.

    Returns
    -------
    lmax: int
        max value for multipole cuts

    lmins: list
        list of ell min for each cross spectrum

    lmaxs: list
        list of ell max for each cross spectrum

    """

    lmins = []
    lmaxs = []
    for hdu in [0, 1, 3, 3]:

        data = fits.getdata(binning_path, hdu + 1)
        lmins.append(np.array(data.field(0), int))
        lmaxs.append(np.array(data.field(1), int))

    lmax = np.max([max(l) for l in lmaxs])

    return(lmax, lmins, lmaxs)


def read_dl_xspectra(spectra_path, nmap, multipole_range, field = 1):

    basename = os.path.join(spectra_path, "cross_NPIPE_detset")
    dldata = []
    for m1 in range(nmap):

        for m2 in range(m1 + 1, nmap):

            tmpcl = []
            for mode, hdu in {"TT": 1, "EE": 2, "TE": 4, "ET": 4}.items():

                filename = "{}_{}_{}.fits".format(basename, m1, m2)
                if mode == "ET":

                    filename = "{}_{}_{}.fits".format(basename, m2, m1)

                data = fits.getdata(filename, hdu)
                ell = np.array(data.field(0), int)
                datacl = np.zeros(np.max(ell) + 1)
                datacl[ell] = data.field(field) * 1e12
                tmpcl.append(datacl[:multipole_range[0] + 1])

            dldata.append(tmpcl)

    return(np.transpose(np.array(dldata), (1, 0, 2)))


def get_spectra(dldata, nxspec, mode):

    dldata = dldata[mode]
    dl = np.array([dldata[xs] for xs in range(nxspec)])

    return(dl)


def set_lists(nmap, nfreq, frequencies):

    xspec2map = []
    for m1 in range(nmap):

        for m2 in range(m1 + 1, nmap):

            xspec2map.append((m1, m2))

    list_fqs = []
    for f1 in range(nfreq):

        for f2 in range(f1, nfreq):

            list_fqs.append((f1, f2))

    freqs = list(np.unique(frequencies))
    xspec2xfreq = []
    for m1 in range(nmap):

        for m2 in range(m1 + 1, nmap):

            f1 = freqs.index(frequencies[m1])
            f2 = freqs.index(frequencies[m2])
            xspec2xfreq.append(list_fqs.index((f1, f2)))

    return(xspec2xfreq)


def xspectra2xfreq(cl, weight, nmap, nfreq,
                   frequencies, multipole_range,
                   normed = True):

    nxfreq = nfreq * (nfreq + 1) // 2
    nxspec = nmap * (nmap - 1) // 2

    xcl = np.zeros((nxfreq, multipole_range[0] + 1))
    xw8 = np.zeros((nxfreq, multipole_range[0] + 1))
    for xs in range(nxspec):

        xcl[set_lists(nmap, nfreq, frequencies)[xs]] += weight[xs] * cl[xs]
        xw8[set_lists(nmap, nfreq, frequencies)[xs]] += weight[xs]

    xw8[xw8 == 0] = np.inf

    if normed:

        return(xcl / xw8)

    else:

        return(xcl, xw8)

def bin_matrix(x, binning):
    """
    Create a matrix operator to bin spectra.

    Inputs
    ------
    x: 1D-array
        x-values for spectra

    binning: 1D-array
        array with different bin edges

    Returns
    -------
    B: 2D-array
        Binning operator
    """

    N = len(x)
    B = []
    for i in range(len(binning) - 1):

        line = np.zeros(N)
        for j in range(N):

            if x[j] >= binning[i] and x[j] < binning[i+1]:

                size = binning[i + 1] - binning[i]
                line[j] = 1 / size

        B.append(line)

    B = np.array(B)

    return(B)


#def bin_spectra(x, spectra, binning):
#    """
#    Bin the spectra with the defined binning.

#   Inputs
#    ------
#    x: 1D-array
#        x-values for spectra

#    spectra: 1D-array
#        spectra to bin

#    binning: 1D-array
#        array with different bin edges

#    Returns
#    -------
#    x_bins: 1D-array
#        binned x-values for the spectra

#    Binned: 1D-array
#        binned spectra
#    """

#    B = bin_matrix(x, binning)
#    Binned = B.dot(spectra)
#    x_bins = B.dot(x)

#    return(x_bins, Binned)
def bin_spectra(x, spectra, binning):

    ellB = np.zeros(len(binning) - 1)
    specB = np.zeros(len(binning) - 1)
    for i in range(len(binning) - 1):

        bmin = binning[i]
        bmax = binning[i+1]
        id = np.where((x >= bmin) & (x < bmax))
        specB[i] = np.mean(spectra[id])
        ellB[i] = np.mean(x[id])

    return(ellB, specB)

def select_spectra(cl, nmap, nfreq,
                   frequencies, binning,
                   multipole_range, mode):

    nxfreq = nfreq * (nfreq + 1) // 2
    acl = np.asarray(cl)
    xl = []
    elll = []
    lims = []
    for xf in range(nxfreq):

        lmin = multipole_range[1][mode][
                   set_lists(nmap, nfreq, frequencies).index(xf)
                                             ]
        lmax = multipole_range[2][mode][
                   set_lists(nmap, nfreq, frequencies).index(xf)
                                             ]
        lims.append([lmin, lmax])
        xl_temp = list(acl[xf, :])
        xl_temp = np.array(xl_temp)
        ell = np.arange(len(xl_temp))
        id_binning = np.where(binning < np.max(ell))
        binning = binning[id_binning]
        ell_temp, xl_temp = bin_spectra(ell, xl_temp, binning)
        id = np.where((ell_temp >= lmin) & (ell_temp <= lmax))
        ell_temp, xl_temp = ell_temp[id], xl_temp[id]
        xl.append(xl_temp)
        elll.append(ell_temp)

    return(elll, xl, lims)

def compute_spectra(nmap, nfreq, frequencies, mode, multipole_range, spectra_path, binning):

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
        ell, spectra, lims = select_spectra(data, nmap,
                                            nfreq, frequencies,
                                            binning, multipole_range,
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

        ell, spectra, lims = select_spectra(Rl / Wl, nmap,
                                            nfreq, frequencies,
                                            binning, multipole_range,
                                            mode = mode)

    return(ell, spectra, lims)

def select_covblock(cov, block = "TTTT"):

    modes = ["TT", "EE", "BB", "TE"]
    blocks = []
    for i, m1 in enumerate(modes):

        line = []
        for j, m2 in enumerate(modes):

            line.append(m1 + m2)

        blocks.append(line)

    blocks = np.array(blocks)
    i, j = np.where(blocks == block)
    i, j = int(i), int(j)
    N = len(cov)
    n = len(modes)
    L = int(N / n)

    return(cov[i * L:(i + 1) * L, j * L:(j + 1) * L])

def read_covmat(xf1, xf2, binning, block,
                nmap, nfreq, frequencies,
                lmax, cov_path, multipole_range):

    # Read the covmat fits file
    fname = cov_path + "fll_NPIPE_detset_TEET_{}_{}.fits".format(xf1, xf2)
    covmat = fits.getdata(fname, 0).field(0)
    N = int(np.sqrt(len(covmat)))
    covmat = covmat.reshape(N, N)
    covmat = select_covblock(covmat, block = block)
    covmat = covmat[:lmax +1, :lmax + 1]

    # Bin the covariance matrix
    N = len(covmat)
    ell = np.arange(N)
    id_binning = np.where(binning < np.max(ell))
    binning = binning[id_binning]
    B = bin_matrix(ell, binning)
    bcovmat = B.dot(covmat.dot(B.T))
    bcovmat *= 1e24
    b_ell = B.dot(ell)

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
    idrow = np.where((b_ell >= lmin1) & (b_ell <= lmax1))
    idcol = np.where((b_ell >= lmin2) & (b_ell <= lmax2))
    min1, max1 = np.min(idrow), np.max(idrow)
    min2, max2 = np.min(idcol), np.max(idcol)
    bcovmat = bcovmat[min1:max1+1, min2:max2+1]

    return(bcovmat)

def svd_pow(A, exponent):
    """
    Compute non integer exponent of a matrix.

    Parameters
    ----------
    A : numpy (N,N) array

    exponent : float

    Returns
    -------
    out : Exponent of the matrix

    """

    E, V = np.linalg.eigh(A)

    return np.einsum("...ab,...b,...cb->...ac", V, E**exponent, V)

def get_foregrounds(data_path, lmax, frequencies):
    fgs = []

    fgsTT = []
    from planck_2020_hillipop import foregrounds_v3 as fg
    fgsTT.append(fg.ps_radio(lmax, frequencies))
    fgsTT.append(fg.ps_dusty(lmax, frequencies))
    fg_lookup = {"dust": fg.dust_model,
                 "SZ": fg.sz_model,
                 "CIB": fg.cib_model,
                 "kSZ": fg.ksz_model,
                 "SZxCIB": fg.szxcib_model}
    fg_names = {"dust": "Foregrounds/Dust_LALmaskSuperExt-ConservativeCO_DX11d",
                "SZ": "Foregrounds/ClSZ_poiss_corr_MD2_cppp_bestCl_NILC_extra12_cibps_lmax10000.fits",
                "kSZ": "Foregrounds/ksz_shaw_bat_PWAS_MD_tau062.fits",
                "CIB": "Foregrounds/CIB_v3",
                "SZxCIB": "Foregrounds/SZxCIB"}

    for name, model in fg_lookup.items():
        filename = os.path.join(data_path, fg_names[name])
        kwargs = dict(mode="TT") if name == "dust" else {}
        fgsTT.append(model(lmax, frequencies,filename, **kwargs))
    fgs.append(fgsTT)

    dust_filename = (os.path.join(data_path, fg_names["dust"]))

    fgsEE = []
    fgsEE.append(fg.dust_model(lmax, frequencies, dust_filename, mode="EE"))
    fgs.append(fgsEE)

    # Init foregrounds TE
    fgsTE = []
    fgsTE.append(fg.dust_model(lmax, frequencies, dust_filename, mode="TE"))
    fgs.append(fgsTE)

    fgsET = []
    fgsET.append(fg.dust_model(lmax, frequencies, dust_filename, mode="ET"))
    fgs.append(fgsET)

    return(fgs)

def compute_fgcal(pars, fgs, nmap, mode):
    nxspec = nmap * (nmap -1) // 2
    cal = []
    for m1 in range(nmap):
        for m2 in range(m1 +1, nmap):
            cal.append(pars["Aplanck"] ** 2 * (1.0 + pars["c%d" % m1] + pars["c%d" % m2]))
    dlmodel = np.zeros(fgs[0][0].compute_dl(pars).shape)
    for fg in fgs[mode]:
        dlmodel += fg.compute_dl(pars)
    spec = np.array([dlmodel[xs] * cal[xs] for xs in range(nxspec)])
    cal = np.array([cal[xs] for xs in range(nxspec)])

    return(spec, cal)

def get_fg_xfreq_vec(pars, fgs, dlweight, multipole_range, nmap, 
                     nfreq, frequencies, mode, binning):
    
    data, cal = compute_fgcal(pars, fgs, nmap, mode)

    if mode != 2:

        data = xspectra2xfreq(data, dlweight[mode], nmap, nfreq, frequencies, multipole_range, normed = True)
        cal = xspectra2xfrea(cal, dlweight[mode], nmap, nfreq, frequencies, multipole_rangem normed = True)

        ell, spectra, lims = select_spectra(data, nmap, nfreq, frequencies, binning, multipole_range, mode)
        ell, gamma, lims = select_spectra(cal, nmap, nfreq, frequencies, binning, multipole_range, mode)

    elif mode == 2:

        dataTE, WTE = xspectra2xfreq(data, dlweight[mode], nmap, nfreq, frequencies, multipole_range, normed = False)
        calTE, WTE = xspectra2xfreq(cal, dlweight[mode], nmap, nfreq, frequencies, multipole_range, normed = False)

        data, cal = compute_fgcal(pars, fgs, nmap, 3)

        dataET, WET = xspectra2xfreq(data, dlweight[3], nmap, nfreq, frequencies, multipole_range, normed = False)
        calET, WET = xspectra2xfreq(cal, dlweight[3], nmap, nfreq, frequencies, multipole_range, normed = False)

        W = WTE + WET
        data = dataTE + dataET
        cal = calTE + calET

        ell, spectra, lims = select_spectra(data/W, nmap, nfreq, frequencies, binning, multipole_range, mode)
        ell, gamma, lims = select_spectra(cal/W, nmap, nfreq, frequencies, binning, multipole_range, mode)

    return(ell, spectra, lims, gamma)
