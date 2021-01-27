from corr_coeff_utils import set_multipole_range, compute_spectra, read_covmat
import os
import numpy as np
import astropy.io.fits as fits

def get_data_dict(nmap, nfreq, frequencies, multipole_range,
                  spectra_path, binning):

    TTells, TTspecs, TTlims = compute_spectra(nmap, nfreq, frequencies, 0,
                                              multipole_range, spectra_path,
                                              binning)
    EEells, EEspecs, EElims = compute_spectra(nmap, nfreq, frequencies, 1,
                                              multipole_range, spectra_path,
                                              binning)
    TEells, TEspecs, TElims = compute_spectra(nmap, nfreq, frequencies, 2,
                                              multipole_range, spectra_path,
                                              binning)

    data_dict = {}
    nxfreq = nfreq * (nfreq + 1) // 2

    for i in range(nxfreq):

        Rmin = np.max([TTlims[i][0], EElims[i][0], TElims[i][0]])
        Rmax = np.min([TTlims[i][1], EElims[i][1], TElims[i][1]])
        if Rmax > 1500:
            Rmax = 1500
        idTT = np.where((TTells[i] >= Rmin) & (TTells[i] <= Rmax))
        idEE = np.where((EEells[i] >= Rmin) & (EEells[i] <= Rmax))
        idTE = np.where((TEells[i] >= Rmin) & (TEells[i] <= Rmax))

        data_dict['ell', i] = TTells[i][idTT]
        data_dict['lims', i] = [Rmin, Rmax]
        data_dict['TT', i] = TTspecs[i][idTT]
        data_dict['EE', i] = EEspecs[i][idEE]
        data_dict['TE', i] = TEspecs[i][idTE]
        data_dict['TT', i, 'ids'] = idTT
        data_dict['EE', i, 'ids'] = idEE
        data_dict['TE', i, 'ids'] = idTE

    return(data_dict)


def compute_bias(i, binning, nmap, nfreq, frequencies,
                 data_dict, cov_path, multipole_range):

    def quickcov(mode):
        return(read_covmat(i, i, binning, mode, nmap,
                           nfreq, frequencies, 2500, cov_path, multipole_range))

    TT = data_dict['TT', i]
    EE = data_dict['EE', i]
    TE = data_dict['TE', i]
    minTT, maxTT = np.min(data_dict['TT', i, 'ids']), np.max(data_dict['TT', i, 'ids'])
    minEE, maxEE = np.min(data_dict['EE', i, 'ids']), np.max(data_dict['EE', i, 'ids'])
    minTE, maxTE = np.min(data_dict['TE', i, 'ids']), np.max(data_dict['TE', i, 'ids'])

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


def compute_RTE(data_dict, nmap, nfreq, frequencies,
                binning, cov_path, multipole_range):

    R_dict = {}

    cfreq = ['100x100', '100x143', '100x217', '143x143', '143x217', '217x217']
    for i in range(len(cfreq)):


        TT = data_dict['TT', i]
        EE = data_dict['EE', i]
        TE = data_dict['TE', i]
        ell = data_dict['ell', i]

        RTE = TE / np.sqrt(EE * TT)
        biasR = compute_bias(i, binning, nmap, nfreq, frequencies, data_dict, cov_path, multipole_range)
        RTE = RTE * (1 - biasR)

        R_dict[cfreq[i]] = np.array([ell, RTE]).T

        R_dict[cfreq[i], 'lim'] = data_dict['lims', i]

    return(R_dict)


def compute_R_cov(i, j, binning, xpol, nmap, nfreq, frequencies,
                  data_dict, cov_path, multipole_range):

    def quickcov(mode):
        return(read_covmat(i, j, binning, mode, nmap,
                           nfreq, frequencies, 2500, cov_path, multipole_range))

    p1, p2 = xpol[:2], xpol[2:]

    TTi = data_dict['TT', i]
    EEi = data_dict['EE', i]
    TEi = data_dict['TE', i]

    TTj = data_dict['TT', j]
    EEj = data_dict['EE', j]
    TEj = data_dict['TE' ,j]

    minTTi, maxTTi = np.min(data_dict['TT', i, 'ids']), np.max(data_dict['TT', i, 'ids'])
    minTTj, maxTTj = np.min(data_dict['TT', j, 'ids']), np.max(data_dict['TT', j, 'ids'])

    minEEi, maxEEi = np.min(data_dict['EE', i, 'ids']), np.max(data_dict['EE', i, 'ids'])
    minEEj, maxEEj = np.min(data_dict['EE', j, 'ids']), np.max(data_dict['EE', j, 'ids'])

    minTEi, maxTEi = np.min(data_dict['TE', i, 'ids']), np.max(data_dict['TE', i, 'ids'])
    minTEj, maxTEj = np.min(data_dict['TE', j, 'ids']), np.max(data_dict['TE', j, 'ids'])

    cov = quickcov(p1 + p2)[minTEi:maxTEi+1,
                            minTEj:maxTEj+1] / np.outer(TEi, TEj)

    cov += 0.25 * quickcov("TTTT")[minTTi:maxTTi+1,
                                   minTTj:maxTTj+1] / np.outer(TTi, TTj)

    cov += 0.25 * quickcov("EEEE")[minEEi:maxEEi+1,
                                   minEEj:maxEEj+1] / np.outer(EEi, EEj)

    cov -= 0.5 * quickcov(p1 + "TT")[minTEi:maxTEi+1,
                                     minTTj:maxTTj+1] / np.outer(TEi, TTj)

    cov -= 0.5 * quickcov("TT" + p2)[minTTi:maxTTi+1,
                                  minTEj:maxTEj+1] / np.outer(TTi, TEj)

    cov -= 0.5 * quickcov(p1 + "EE")[minTEi:maxTEi+1,
                                     minEEj:maxEEj+1] / np.outer(TEi, EEj)

    cov -= 0.5 * quickcov("EE" + p2)[minEEi:maxEEi+1,
                                     minTEj:maxTEj+1] / np.outer(EEi, TEj)

    cov += 0.25 * quickcov("TTEE")[minTTi:maxTTi+1,
                                   minEEj:maxEEj+1] / np.outer(TTi, EEj)

    cov += 0.25 * quickcov("EETT")[minEEi:maxEEi+1,
                                   minTTj:maxTTj+1] / np.outer(EEi, TTj)

    return(cov)


def compute_R_covdict(nmap, nfreq, frequencies, R_dict, data_dict, cov_path, binning):

    cov_dict = {}
    cfreq = ['100x100', '100x143', '100x217', '143x143', '143x217', '217x217']
    nxfreq = nfreq * (nfreq + 1) // 2
    for i in range(nxfreq):

        for j in range(i, nxfreq):

            cov = compute_R_cov(i, j, binning, "TETE", nmap, nfreq, frequencies,
                                data_dict, cov_path, multipole_range)

            cov *= np.outer(R_dict[cfreq[i]][:, 1], R_dict[cfreq[j]][:, 1])
            cov_dict[cfreq[i], cfreq[j]] = cov
            print("{}x{} done !".format(cfreq[i], cfreq[j]))

    return(cov_dict)


def get_fullcov_RTE(covR_dict, spec_order):

    fullcov = []
    for xfreq1 in spec_order:

        line = []
        for xfreq2 in spec_order:
            try:
                line.append(covR_dict[xfreq1, xfreq2])
            except:
                line.append(covR_dict[xfreq2, xfreq1].T)
        fullcov.append(line)
    fullcov = np.block(fullcov)

    return(np.linalg.inv(fullcov))


def get_vec_bias(nmap, nfreq, frequencies, binning, cov_path, data_dict):

    nxfreq = (nfreq * nfreq + 1) // 2
    bias_vec = []
    for i in range(nxfreq):

        biasR = compute_bias(i, binning, nmap, nfreq, frequencies,
                             data_dict, cov_path, multipole_range)

        bias_vec = np.concatenate((bias_vec, biasR))

    bias_vec = np.array(bias_vec)

    return(bias_vec)

# ==================== #

data_path = "../modules/data/Hillipop/"
binning_path = data_path + "Binning/"
spectra_path = data_path + "Data/NPIPE/spectra/"
cov_path = data_path + "Data/NPIPE/pclcvm/fll/"

output_path = "outputs/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

save_invcov_path = output_path + "RTE_NPIPE_invcov_bin_sim_TTEETEET.fits"
save_bias_path = output_path + "corr_coeff_bias.dat"

# ==================== #

nmap = 6
nfreq = 3
frequencies = [100, 100, 143, 143, 217, 217]
spec_order = ["100x100", "100x143", "100x217", "143x143", "143x217", "217x217"]

b1 = np.arange(50, 800, step = 30)
b2 = np.arange(np.max(b1)+60, 1200, step = 60)
b3 = np.arange(np.max(b2) + 100, 9000, step = 100)
binning = np.concatenate((b1, b2, b3))
np.savetxt(output_path + "binning_corr_coeff.dat", binning)

multipole_range = set_multipole_range(binning_path)
print(multipole_range[0])
data_dict = get_data_dict(nmap, nfreq, frequencies, multipole_range, spectra_path, binning)
R_dict  = compute_RTE(data_dict, nmap, nfreq, frequencies, binning, cov_path, multipole_range)

covR_dict = compute_R_covdict(nmap, nfreq, frequencies, R_dict, data_dict, cov_path, binning)
invcovR = get_fullcov_RTE(covR_dict, spec_order)

biasR = get_vec_bias(nmap, nfreq, frequencies, binning, cov_path, data_dict)
np.savetxt(save_bias_path, biasR)
col = [fits.Column(name = 'invfullcov', format = 'D', array = invcovR.reshape(len(invcovR) * len(invcovR)))]
hdulist=fits.BinTableHDU.from_columns(col)
hdulist.writeto(save_invcov_path, overwrite=True)
print("DONE !")
