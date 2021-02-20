import numpy as np
import astropy.io.fits as fits
import os
from planck_2020_hillipop import foregrounds_v3 as fg
# ================================ #

data_path = '/sps/litebird/Users/alaposta/development/hillipop/modules/data/Hillipop/'
multipole_range_file = '/sps/litebird/Users/alaposta/development/corr_coeff/hillipop/modules/data/planck_2020/hillipop/data/binning_lowl_extended.fits'
spectra_path = data_path + 'Data/NPIPE/spectra/'
cov_path = data_path + 'Data/NPIPE/pclcvm/fll/'
binning_file = 'binning_corr_coeff.fits'
invcov_file="cov/invfll_NPIPE_detset_extl_TTTEEE_bin.fits"
output_path = "/sps/litebird/Users/alaposta/development/corr_coeff/python_products/"

# ================================ #

nmap = 6
nfreq = 3
nxfreq = nfreq * (nfreq + 1) // 2
nxspec = nmap * (nmap - 1) // 2
frequencies = [100, 100, 143, 143, 217, 217]

parameters = {"cosmomc_theta": 0.0104064,
              "As": pow(10, -10) * np.exp(3.038),
              "ns": 0.9601,
              "ombh2": 0.02216,
              "omch2": 0.1195,
              "tau": 0.0536,
              "Aplanck": 1, #9.997e-1,
              "c0": 0.0, #7.064e-4,
              "c1": 0.0, #4.000e-3,
              "c2": 0,
              "c3": 0.0, #-6.344e-4,
              "c4": 0.0, #-1.432e-3,
              "c5": 0.0, #-2.863e-3,
              "Aradio": 1.698,
              "Adusty": 6.258e-1,
              "AdustTT": 8.649e-1,
              "AdustPP": 1.156,
              "AdustTP": 1.092,
              "Asz": 1.155,
              "Acib": 1.070,
              "Aksz": 0,
              "Aszxcib": 0}

# ================================== #
# ============= TOOLS ============== #
# ================================== #

def set_multipole_range(mrange_file):

    lmins = []
    lmaxs = []
    for hdu in [0, 1, 3, 3]:

        data = fits.getdata(os.path.join(mrange_file), hdu + 1)
        lmins.append(np.array(data.field(0), int))
        lmaxs.append(np.array(data.field(1), int))

    lmax = np.max([max(l) for l in lmaxs])

    return(lmax, lmins, lmaxs)


def bin_matrix(x, binning):

    N = len(x)
    B = []
    for i in range(len(binning) - 1):

        line = np.zeros(N)
        for j in range(N):

            if x[j] >= binning[i] and x[j] < binning[i+1]:

                size = binning[i+1] - binning[i]
                line[j] = 1 / size

        B.append(line)

    B = np.array(B)

    return(B)

def bin_spectra(x , spectra, binning):

    B = bin_matrix(x, binning)
    Binned = B.dot(spectra)
    x_bins = B.dot(x)

    return(x_bins, Binned)

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

# ======================================= #
# ============== FOREGROUNDS ============ #
# ======================================= #

def get_foregrounds(lmax, frequencies):
    fgs = []

    fgsTT = []
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

# ===================================== #
# ============ SPECTRA DAT ============ #
# ===================================== #

def read_dl_xspectra(nmap, mrange ,field = 1):

    basename = os.path.join(spectra_path, "cross_NPIPE_detset")
    dldata = []
    for m1 in range(nmap):

        for m2 in range(m1 +1, nmap):

            tmpcl = []
            for mode, hdu in {"TT": 1, "EE": 2, "TE": 4, "ET": 4}.items():

                filename = "{}_{}_{}.fits".format(basename, m1, m2)
                if mode == "ET":

                    filename = "{}_{}_{}.fits".format(basename, m2, m1)

                data = fits.getdata(filename, hdu)
                ell = np.array(data.field(0), int)
                datacl = np.zeros(np.max(ell)+1)
                datacl[ell] = data.field(field) * 1e12
                tmpcl.append(datacl[:mrange[0] + 1])

            dldata.append(tmpcl)

    return(np.transpose(np.array(dldata), (1, 0, 2)))

def get_spectra(dldata, nxspec, mode = 0):


    dldata = dldata[mode]

    dl = np.array([dldata[xs] for xs in range(nxspec)])

    return(dl)

def set_lists(nmap, nfreq, frequencies):

    xspec2map = []
    list_xfq = []
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

def xspectra2xfreq(cl, weight, mrange, nmap, nfreq, frequencies, nxspec, nxfreq, normed = True):

    xcl = np.zeros((nxfreq, mrange[0] + 1))
    xw8 = np.zeros((nxfreq, mrange[0] + 1))
    for xs in range(nxspec):

        xcl[set_lists(nmap, nfreq, frequencies)[xs]] += weight[xs] * cl[xs]
        xw8[set_lists(nmap, nfreq, frequencies)[xs]] += weight[xs]

    xw8[xw8 == 0] = np.inf

    if normed:

        return xcl / xw8

    else:

        return (xcl, xw8)

def select_spectra(cl, nxfreq, mrange,  nmap, nfreq, frequencies, binning, mode = 0):

    acl = np.asarray(cl)
    xl = []
    elll = []
    lims = []
    for xf in range(nxfreq):

        lmin = mrange[1][mode][set_lists(nmap, nfreq, frequencies).index(xf)]

        lmax = mrange[2][mode][set_lists(nmap, nfreq, frequencies).index(xf)]
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

def select_spectra_tb(cl, nxfreq, nmap, nfreq, frequencies, mode = 0):
    acl = np.asarray(cl)
    xl = []
    elll = []
    for xf in range(nxfreq):
        lmin = set_multipole_range()[1][mode][set_lists(nmap, nfreq, frequencies).index(xf)]
        lmax = set_multipole_range()[2][mode][set_lists(nmap, nfreq, frequencies).index(xf)]
        xl.append(acl[xf, :])
        ell = np.arange(len(acl[xf, :]))
        elll.append(ell)
    return(elll, xl, 0)


def compute_spectra(mrange, nmap, nfreq, nxspec, frequencies, nxfreq, mode, binning):

    dlsig = read_dl_xspectra(nmap, mrange_file, field = 2)
    dlsig[dlsig == 0] = np.inf
    dlweight = 1.0 / dlsig ** 2

    data = read_dl_xspectra(nmap, mrange)
    data = get_spectra(data, nxspec, mode = mode)
    data = xspectra2xfreq(data, dlweight[mode], mrange, nmap, nfreq, frequencies, nxspec, nxfreq, normed = True)
    ell, spectra, lims = select_spectra(data, nxfreq, mrange, nmap, nfreq, frequencies, binning, mode = mode)

    return(ell, spectra, lims)

# ===================================== #
# ============ SPECTRA SIM ============ #
# ===================================== #

def compute_model(pars, dlth, fgs, nmap, nxspec, mode = 0):

    cal = []
    for m1 in range(nmap):
        for m2 in range(m1 +1, nmap):
            cal.append(pars["Aplanck"] ** 2 * (1.0 + pars["c%d" % m1] + pars["c%d" % m2]))

    dlmodel = [dlth[mode]] * nxspec
    for fg in fgs[mode]:
        print(np.shape(fg.compute_dl(pars)))
        dlmodel += fg.compute_dl(pars)

    Spec = np.array([dlmodel[xs] * cal[xs] for xs in range(nxspec)])

    return(Spec)

def compute_basemodel(pars, fgs, mrange, nmap, nfreq, nxspec, frequencies, nxfreq, mode, binning, lmax):

    import camb
    from camb import model, initialpower
    params = camb.CAMBparams()
    params.set_cosmology(cosmomc_theta = pars["cosmomc_theta"],
                         ombh2 = pars["ombh2"],
                         omch2 = pars["omch2"],
                         mnu = 0.06, omk = 0,
                         tau = pars["tau"])
    params.InitPower.set_params(As = pars["As"], ns = pars["ns"], r = 0)
    params.set_for_lmax(lmax, lens_potential_accuracy=1)
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit = "muK")
    CLtot = powers['total']
    cl = {"tt": CLtot[:, 0],
          "ee": CLtot[:, 1],
          "te": CLtot[:, 3]}
    lth = np.arange(lmax + 1)
    dlth = []
    for pol in ["tt", "ee", "te", "te"]:
        dlth += [cl[pol][lth]]
    dlth = np.asarray(dlth)
    print(dlth)
    dlsig = read_dl_xspectra(nmap,mrange, field = 2)
    dlsig[dlsig == 0] = np.inf
    dlweight = 1.0 / dlsig ** 2

    #data = read_dl_xspectra(nmap)
    data = compute_model(pars, dlth, fgs, nmap, nxspec, mode)
    #data = get_spectra(data, nxspec, mode = mode)
    if mode != 2:
        data = xspectra2xfreq(data, dlweight[mode], mrange,nmap, nfreq, frequencies, nxspec, nxfreq, normed = True)
        ell, spectra, lims = select_spectra(data, nxfreq, mrange,nmap, nfreq, frequencies, binning, mode = mode)
    elif mode == 2:
        dataTE, WTE = xspectra2xfreq(data, dlweight[mode], mrange,nmap, nfreq, frequencies, nxspec, nxfreq, normed = False)
        data = compute_model(pars, dlth, fgs, nmap, nxspec, 3)
        dataET, WET = xspectra2xfreq(data, dlweight[3], mrange,nmap, nfreq, frequencies, nxspec, nxfreq, normed = False)
        data = dataTE + dataET
        W = WTE + WET
        ell, spectra, lims = select_spectra(data/W, nxfreq, mrange,nmap, nfreq, frequencies, binning, mode = 2)

    return(ell, spectra, lims)

# ==================================== #
# ============ COVARIANCE ============ #
# ==================================== #

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

    return(cov[i * L:(i+1) * L, j*L:(j+1) * L])

def read_covmat(xf1, xf2, binning, block, nmap, nfreq, frequencies, lmax):

    fname = cov_path + "fll_NPIPE_detset_TEET_{}_{}.fits".format(xf1, xf2)
    covmat = fits.getdata(fname, 0).field(0)
    N = int(np.sqrt(len(covmat)))
    covmat = covmat.reshape(N, N)
    covmat = select_covblock(covmat, block = block)
    covmat = covmat[:lmax+1,:lmax+1]
    N = len(covmat)

    ell = np.arange(N)
    id_binning = np.where(binning < np.max(ell))
    binning = binning[id_binning]
    B = bin_matrix(ell, binning)
    bcovmat = B.dot(covmat.dot(B.T))
    bcovmat *= 1e24 ## K to muK
    b_ell = B.dot(ell)

    modes = ["TT", "EE", "TE", "ET"]
    mode1 = block[:2]
    mode2 = block[2:]
    id1 = modes.index(mode1)
    id2 = modes.index(mode2)
    lmin1 = set_multipole_range()[1][id1][set_lists(nmap, nfreq, frequencies).index(xf1)]
    lmax1 = set_multipole_range()[2][id1][set_lists(nmap, nfreq, frequencies).index(xf1)]
    lmin2 = set_multipole_range()[1][id2][set_lists(nmap, nfreq, frequencies).index(xf2)]
    lmax2 = set_multipole_range()[2][id2][set_lists(nmap, nfreq, frequencies).index(xf2)]
    idrow = np.where((b_ell >= lmin1) & (b_ell <= lmax1))
    idcol = np.where((b_ell >= lmin2) & (b_ell <= lmax2))
    min1, max1 = np.min(idrow), np.max(idrow)
    min2, max2 = np.min(idcol), np.max(idcol)

    bcovmat = bcovmat[min1:max1+1, min2:max2+1]

    return(bcovmat)

def produce_cov_pattern(nxfreq, modes):

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

def compute_full_covmat(nxfreq, modes, binning, nmap, nfreq, frequencies, lmax):

    print("Creating pattern for the full covariance matrix ...")
    pattern_cov = produce_cov_pattern(nxfreq, modes)
    full_cov = []
    print("Starting to fill the matrix ...")
    for i in range(len(pattern_cov)):

        line = []
        for j in range(len(pattern_cov)):

            mode, xfreq_couple = pattern_cov[i][j]
            xf1, xf2 = xfreq_couple
            xf1, xf2 = int(xf1), int(xf2)
            m1, m2 = mode[:2], mode[2:]
            try:
                line.append(
                    read_covmat(xf1, xf2, binning, mode,
                                nmap, nfreq, frequencies, lmax)
                               )
            except:
                line.append(
                    read_covmat(xf2, xf1, binning, m2+m1,
                                nmap, nfreq, frequencies, lmax).T)
        full_cov.append(line)

    full_cov = np.block(full_cov)
    print("End of computation.")

    return(full_cov)

# ======================================== #
# ============= SIMULATIONS ============== #
# ======================================== #

def get_sim(base_model, sqrcov):

    sim = base_model + sqrcov.dot(np.random.randn(len(base_model)))
    return(sim)

# ============================================== #
# ============== RTE COMPUTATION =============== #
# ============================================== #


def compute_bias(i, binning, nmap, nfreq, frequencies,
                 TT, EE, TE, minTT, maxTT, minEE,
                 maxEE, minTE, maxTE):

    def quickcov(mode):
        return(read_covmat(i, i, binning, mode, nmap,
                           nfreq, frequencies, 2500))

    bias = (3 / 8) * quickcov("TTTT")[minTT:maxTT+1,
                                      minTT:maxTT+1] / np.outer(TT, TT)

    bias += (3 / 8) * quickcov("EEEE")[minEE:maxEE+1,
                                       minEE:maxEE+1] / np.outer(EE, EE)

    bias -= 0.5 * quickcov("TTTE")[minTT:maxTT+1,
                                   minTE:maxTE+1] / np.outer(TT, TE)

    bias -= 0.5 * quickcov("EETE")[minEE:maxEE+1,
                                   minTE:maxTE+1] / np.outer(EE, TE)

    bias += 0.25 * quickcov("TTEE")[minTT:maxTT+1,
                                    minEE:maxEE+1] / np.outer(TT, EE)

    bias = bias.diagonal()

    return(bias)

def compute_RTE(nmap, nfreq, nxspec, frequencies, nxfreq, binning):

    TTells, TTspecs, TTlims = compute_spectra(nmap, nfreq, nxspec, frequencies, nxfreq, 0, binning)
    EEells, EEspecs, EElims = compute_spectra(nmap, nfreq, nxspec, frequencies, nxfreq, 1, binning)
    TEells, TEspecs, TElims = compute_spectra(nmap, nfreq, nxspec, frequencies, nxfreq, 2, binning)

    R_dict = {}
    C_dict = {}
    cfreq = ['100x100', '100x143', '100x217', '143x143', '143x217', '217x217']
    for i in range(nxfreq):

        Rmin = np.max([TTlims[i][0], EElims[i][0], TElims[i][0]])
        print(TTlims[i][0], EElims[i][0], TElims[i][0])
        print(TTlims[i][1], EElims[i][1], TElims[i][1])
        if i != 5 and i != 4:
            Rmax = np.min([TTlims[i][1], EElims[i][1], TElims[i][1]])
        else:
            Rmax = 1500

        TTell, TTspec = TTells[i], TTspecs[i]
        EEell, EEspec = EEells[i], EEspecs[i]
        TEell, TEspec = TEells[i], TEspecs[i]

        idTT = np.where((TTell >= Rmin) & (TTell <= Rmax))
        idEE = np.where((EEell >= Rmin) & (EEell <= Rmax))
        idTE = np.where((TEell >= Rmin) & (TEell <= Rmax))
        TTmin, TTmax = np.min(idTT), np.max(idTT)
        EEmin, EEmax = np.min(idEE), np.max(idEE)
        TEmin, TEmax = np.min(idTE), np.max(idTE)
        Rspec = TEspec[idTE] / np.sqrt(EEspec[idEE] * TTspec[idTT])
        biasR = compute_bias(i, binning, nmap, nfreq, frequencies,
                             TTspec[idTT], EEspec[idEE], TEspec[idTE], TTmin, TTmax,
                             EEmin, EEmax, TEmin, TEmax)
        Rspec = Rspec * (1 - biasR)
        Rell = TTell[idTT]

        R_dict[cfreq[i]] = np.array([Rell, Rspec]).T
        R_dict[cfreq[i], 'lim'] = [Rmin, Rmax]

        C_dict[cfreq[i], "TT"] = np.array([Rell, TTspec[idTT]]).T
        C_dict[cfreq[i], "EE"] = np.array([Rell, EEspec[idEE]]).T
        C_dict[cfreq[i], "TE"] = np.array([Rell, TEspec[idTE]]).T

    return(R_dict, C_dict)

def get_vec_bias(TTells, TTspecs, TTlims, EEells, EEspecs, EElims,
                 TEells, TEspecs, TElims, nmap, nfreq, nxspec,
                 frequencies, nxfreq, binning):

    bias_vec = []
    ids = {'TT': [],
           'EE': [],
           'TE': []}
    for i in range(nxfreq):

        Rmin = np.max([TTlims[i][0], EElims[i][0], TElims[i][0]])
        print(TTlims[i][0], EElims[i][0], TElims[i][0])
        print(TTlims[i][1], EElims[i][1], TElims[i][1])
        if i != 5 and i != 4:
            Rmax = np.min([TTlims[i][1], EElims[i][1], TElims[i][1]])
        else:
            Rmax = 1500

        TTell, TTspec = TTells[i], TTspecs[i]
        EEell, EEspec = EEells[i], EEspecs[i]
        TEell, TEspec = TEells[i], TEspecs[i]

        idTT = np.where((TTell >= Rmin) & (TTell <= Rmax))
        idEE = np.where((EEell >= Rmin) & (EEell <= Rmax))
        idTE = np.where((TEell >= Rmin) & (TEell <= Rmax))
        ids['TT'].append(idTT)
        ids['EE'].append(idEE)
        ids['TE'].append(idTE)
        TTmin, TTmax = np.min(idTT), np.max(idTT)
        EEmin, EEmax = np.min(idEE), np.max(idEE)
        TEmin, TEmax = np.min(idTE), np.max(idTE)

        biasR = compute_bias(i, binning, nmap, nfreq, frequencies,
                             TTspec[idTT], EEspec[idEE], TEspec[idTE],
                             TTmin, TTmax, EEmin, EEmax, TEmin, TEmax)

        bias_vec = np.concatenate((bias_vec, biasR))

    bias_vec = np.array(bias_vec)

    return(bias_vec, ids)


lmax, lmins, lmaxs = set_multipole_range(multipole_range_file)
mrange = set_multipole_range(multipole_range_file)
fgs = get_foregrounds(lmax, frequencies)
binning = fits.getdata(output_path + binning_file, hdu = 0).field(0)

base_ellTT, base_modelTT, base_limsTT = compute_basemodel(parameters, fgs, mrange,nmap,
                                                          nfreq, nxspec,
                                                          frequencies, nxfreq, 0,
                                                          binning, lmax)
base_ellEE, base_modelEE, base_limsEE = compute_basemodel(parameters, fgs, mrange,nmap,
                                                          nfreq, nxspec,
                                                          frequencies, nxfreq, 1,
                                                          binning, lmax)
base_ellTE, base_modelTE, base_limsTE = compute_basemodel(parameters, fgs, mrange,nmap,
                                                          nfreq, nxspec,
                                                          frequencies, nxfreq, 2,
                                                          binning, lmax)



cfreq = ['100x100','100x143','100x217','143x143','143x217','217x217']

base_vec = np.concatenate((np.concatenate(base_modelTT),
                           np.concatenate(base_modelEE),
                           np.concatenate(base_modelTE)))
base_ell = np.concatenate((np.concatenate(base_ellTT),
                           np.concatenate(base_ellEE),
                           np.concatenate(base_ellTE)))

modes = ["TT", "EE", "TE"]
RL = False
CL = True

np.savetxt("../python_products/sims/bf_input.dat", base_vec)
np.savetxt("../python_products/sims/ell.dat", base_ell)
if CL:
    #cov = compute_full_covmat(nxfreq, modes, binning, nmap, nfreq, frequencies, lmax)
    invcov = fits.getdata(output_path + invcov_file, hdu = 0).field(0)
    N = int(np.sqrt(len(invcov)))
    invcov = invcov.reshape(N, N)
    cov = np.linalg.inv(invcov)
    cov *= 1e24
    sqrcov = svd_pow(cov, 0.5)
    #np.savetxt("../outputs/sims/TTTEEE/ell.dat", base_ell)
    Nsims = 100

    import os
    sims = []
    for i in range(Nsims):
        print("Sim {}".format(i))
        if not os.path.exists('../python_products/sims/'):
            os.makedirs('../python_products/sims/')
        sim = get_sim(base_vec, sqrcov)
        sims.append(sim)
        #np.savetxt("../python_products/sims/sim%02d.dat"%i, get_sim(base_vec, sqrcov))
        #np.savetxt("../python_products/sims/sim%02d.dat" % i, sim)
    print(sims[0]-sims[1])

    
    for i, elem in enumerate(sims):
        np.savetxt("../python_products/sims/sim_nc_{}.dat".format(i), elem)
