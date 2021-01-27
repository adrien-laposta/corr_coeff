import numpy as np
import matplotlib.pyplot as plt
from corr_coeff_utils import bin_spectra
import astropy.io.fits as fits

binning_file = "../python_products/binning_corr_coeff.fits"
gibbs_path = "../python_products/gibbs_inputs/"

binning = fits.getdata(binning_file, hdu = 0).field(0)
Cb = np.loadtxt(gibbs_path + "Cb_example.dat")
Q = np.loadtxt(gibbs_path + "Q_example.dat")
ell = np.loadtxt(gibbs_path + "ell_cmb.dat")

index = [0]
for i in range(len(ell) - 1):
    if ell[i+1] < ell[i]:
        index.append(i+1)
index.append(len(ell))

def cut_ell(index, L):
    cut = []
    for i in range(len(index) - 1):
        cut.append(L[index[i]:index[i+1]])
    
    return(cut)

ellTT, ellEE, ellTE = cut_ell(index, ell)
CbTT, CbEE, CbTE = cut_ell(index, Cb)

nTT = len(CbTT)
nEE = len(CbEE)
nTE = len(CbTE)

minTT = 0
maxTT = nTT
minEE = nTT
maxEE = nTT + nEE
minTE = nTT + nEE
maxTE = nTT + nEE + nTE

ell_R_min = np.max([ellTT[0], ellEE[0], ellTE[0]])
ell_R_max = np.min([ellTT[-1], ellEE[-1], ellTE[-1]])

idTT = np.where((ellTT >= ell_R_min) & (ellTT <= ell_R_max))
idEE = np.where((ellEE >= ell_R_min) & (ellEE <= ell_R_max))
idTE = np.where((ellTE >= ell_R_min) & (ellTE <= ell_R_max))

cutTT, cutEE, cutTE = CbTT[idTT], CbEE[idEE], CbTE[idTE]
ell_R = ellTT[idTT]

Q_TTTT = Q[minTT:maxTT, minTT:maxTT][np.min(idTT):np.max(idTT)+1, np.min(idTT):np.max(idTT)+1]
Q_TTEE = Q[minTT:maxTT, minEE:maxEE][np.min(idTT):np.max(idTT)+1, np.min(idEE):np.max(idEE)+1]
Q_TTTE = Q[minTT:maxTT, minTE:maxTE][np.min(idTT):np.max(idTT)+1, np.min(idTE):np.max(idTE)+1]

Q_EEEE = Q[minEE:maxEE, minEE:maxEE][np.min(idEE):np.max(idEE)+1, np.min(idEE):np.max(idEE)+1]
Q_EETE = Q[minEE:maxEE, minTE:maxTE][np.min(idEE):np.max(idEE)+1, np.min(idTE):np.max(idTE)+1]

Q_TETE = Q[minTE:maxTE, minTE:maxTE][np.min(idTE):np.max(idTE)+1, np.min(idTE):np.max(idTE)+1]

plt.figure()
plt.plot(ell_R, cutEE/np.sqrt(Q_EEEE.diagonal()))
plt.axhline(5,xmin=-100,xmax=2500)
#plt.errorbar(ell_R, cutEE, yerr=np.sqrt(Q_EEEE.diagonal()))
plt.show()
R = cutTE / np.sqrt(cutEE * cutTT)

alpha = 3/8 * (Q_TTTT / np.outer(cutTT, cutTT) + Q_EEEE / np.outer(cutEE, cutEE))
alpha -= 0.5 * (Q_TTTE / np.outer(cutTT, cutTE) + Q_EETE / np.outer(cutEE, cutTE))
alpha += 0.25 * Q_TTEE / np.outer(cutTT, cutEE)
alpha = alpha.diagonal()
R_unb = R * (1 - alpha)

cov_R = Q_TETE / np.outer(cutTE, cutTE)

cov_R += 0.25 * (Q_TTTT / np.outer(cutTT, cutTT) + Q_EEEE / np.outer(cutEE, cutEE))
cov_R -= Q_TTTE / np.outer(cutTT, cutTE) + Q_EETE / np.outer(cutEE, cutTE)
cov_R += 0.5 * Q_TTEE / np.outer(cutTT, cutEE)

cov_R *= np.outer(R_unb, R_unb)

sigma_R = np.sqrt(cov_R.diagonal())


### BEST FIT
import camb
from camb import model, initialpower
pars = camb.CAMBparams()
pars.set_cosmology(cosmomc_theta = 0.0104064,
                   ombh2 = 0.02216,
                   omch2 = 0.1195,
                   mnu = 0.06,
                   omk = 0,
                   tau = 0.0536)
pars.InitPower.set_params(As = pow(10, -10)*np.exp(3.038), ns = 0.9601, r = 0)
pars.set_for_lmax(2500, lens_potential_accuracy=1)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL = powers['total']
ls = np.arange(totCL.shape[0])[2:]
TTbf = totCL[:, 0][2:]
EEbf = totCL[:, 1][2:]
TEbf = totCL[:, 3][2:]

Rbf = TEbf / np.sqrt(EEbf * TTbf)
binning = binning[np.where(binning < np.max(ls))]
ls_b, TTbf_b = bin_spectra(ls, TTbf, binning)
ls_b, EEbf_b = bin_spectra(ls, EEbf, binning)
ls_b, TEbf_b = bin_spectra(ls, TEbf, binning)
Rbf_b = TEbf_b / np.sqrt(TTbf_b * EEbf_b)

# BEST FIT BINNING
idR = np.where((ls_b >= np.min(ell_R)) & (ls_b <= np.max(ell_R)))
ls_b = ls_b[idR]
Rbf_b = Rbf_b[idR]
ls = ls[int(np.min(ell_R)):int(np.max(ell_R))]
Rbf = Rbf[int(np.min(ell_R)):int(np.max(ell_R))]

chi2 = (R_unb - Rbf_b).dot(np.linalg.inv(cov_R)).dot(R_unb - Rbf_b)
chi2 = round(chi2, 2)
fig = plt.figure(figsize = (8, 6))
ax = fig.add_axes((.1, .3, .8, .6))
ax.plot(ls, Rbf, color = "k", lw = 0.5)
ax.plot(ls_b, Rbf_b, color = "k", ls = '--', lw = 0.5)
ax.errorbar(ell_R, R_unb, yerr = sigma_R, ls = 'None',
             color = "tab:red", elinewidth=1.4, marker = '.')
ax.set_ylabel(r"$\mathcal{R}_\ell^{TE}}$", fontsize = 14)
ax.set_xticklabels([])
ax2 = fig.add_axes((.1, .1, .8, .2))
ax2.set_xlabel(r"$\ell$", fontsize = 14)
ax2.axhline(0, xmin = -100, xmax = 2000, color = "k")
ax2.errorbar(ell_R, (R_unb - Rbf_b), yerr = sigma_R, ls = "None",
            color = "tab:red", marker = ".", elinewidth=1.4)
ax2.set_ylabel(r"$\Delta\mathcal{R}_\ell^{TE}}$", fontsize = 14)
ax2.annotate(r'$\chi^2$ = {}/{}'.format(chi2, len(ell_R)), xy =(0.65,0.85), xycoords='axes fraction', fontsize = 10)
ax2.grid(True, ls = "dotted")
plt.show()
