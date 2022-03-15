from pathlib import Path

import caiman
import caiman.source_extraction.cnmf as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from sklearn import preprocessing

NAS_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")
stat_file_list = sorted(list(NAS_PROC_DIR.rglob('combined/stat.npy')))

#%%
def load_traces(stat_file):
    F = np.load(stat_file.with_name('F.npy'))
    Fneu = np.load(stat_file.with_name('Fneu.npy'))
    iscell = np.load(stat_file.with_name('iscell.npy'))

# def detrend_df_f(C, quantile_min=8, frames_window=200, return_baseline=False):
#     """ Compute dF/F, with baseline computed w/ rolling percentile window.
#
#     :param C:
#     :param quantile_min:
#     :param frames_window:
#     :return:
#     """
#     F0 = scipy.ndimage.percentile_filter(C, quantile_min, (frames_window, 1))
#     C_df = C./F0

def correct_and_deconvolve(C, oasis_params, quantile_norm=True):
    """
        Takes parameters for baseline correction (suite2p.dcnv.preprocess) and deconvolution (oasis AR2, constrained
        foopsi) and returns block-corrected traces.

        :param Fc: neuropil-corrected fluorescence from suite2p ( = F - 0.7 * Fneu)
        :type Fc: np.ndarray
    """
    n_cells, T = C.shape

    # quantile matching & transform, quantile_range=(0.25, 0.75)
    if quantile_norm:
        d = preprocessing.RobustScaler(with_centering=False).fit_transform(C)
    else:
        d = C

    # oasis deconvolution
    C_dec = np.zeros(d.shape)
    Bl = np.zeros(n_cells)
    C1 = np.zeros(d.shape)
    G = np.zeros((n_cells, 2))
    Sn = np.zeros(n_cells)
    Sp = np.zeros((n_cells, T))
    Lam = np.zeros(n_cells)

    for cid in range(C_dec.shape[0]):
        y = d[cid, :]

        #g0 = caiman.source_extraction.cnmf.deconvolution.estimate_time_constant(y, p=2, lags=5, fudge_factor=1.)
        #g = caiman.source_extraction.cnmf.deconvolution.estimate_time_constant(y, p=1, lags=10)
        c, bl, c1, g, sn, sp, lam = caiman.source_extraction.cnmf.deconvolution.constrained_foopsi(y, **oasis_params)
        C_dec[cid, :] = c
        Bl[cid] = bl
        C1[cid] = c1
        G[cid, :] = g
        Sp[cid, :] = sp
        #Sn[cid] = cm.deconvolution.GetSn(y)
        Sn[cid] = sn
        Lam[cid] = lam

    oasis_results = dict(C_dec=C_dec, bl=Bl, c1=C1, g=G, sp=Sp, sn=Sn, lam=Lam, oasis_params=oasis_params)
    return d, oasis_results

def load_and_detrend_F(stat_file):
    F = np.load(stat_file.with_name('F.npy'))
    Fneu = np.load(stat_file.with_name('Fneu.npy'))
    # iscell = np.load(stat_file.with_name('iscell.npy'))
    #cellprob = iscell[:, 1]
    #iscell = iscell[:, 0]

    Fneu_smoothed = scipy.ndimage.gaussian_filter1d(Fneu, sigma=3, axis=1, )
    Fc = F - 0.7 * Fneu_smoothed

    # detrend neuropil-corrected fluorescence
    F0 = scipy.ndimage.percentile_filter(Fc, 20, (1, 200)) # baseline
    C_df = (Fc - F0)/F0  # dF/F
    return C_df, Fc, F0


def main(stat_file):
    C_df, Fc, F0 = load_and_detrend_F(stat_file)

    # caiman deconvolution
    oasis_params = dict(p=1,
                        penalty=0,
                        g=np.array([.88]),
                        smin=1,
                        #lags=10,
                        #smin=None,
                        g_optimize=3,
                        )
    C_qtn, oasis_results = correct_and_deconvolve(C_df, oasis_params=oasis_params, quantile_norm=False)

    save_file = stat_file.with_name('caiman_deconv_results.npy')
    np.save(save_file,
            dict(Fc=Fc, F0=F0, C_df=C_df, C_qtn=C_qtn, oasis_results=oasis_results),
            allow_pickle=True)
    return save_file

#%%
stat_file_list = stat_file_list[2:]
saved_files = []

for item in stat_file_list:
    print(f"\n{item.relative_to(NAS_PROC_DIR)}")
    saved_files.append(main(item))
    print('file saved')




#%%

# sn_fcc = [cm.deconvolution.GetSn(y, method='mean') for y in Fcc]
#
# upper_quantile = np.percentile(Fcc, 75, axis=1)
# lower_quantile = np.percentile(Fcc, 25, axis=1)
# scaling_factor = upper_quantile - lower_quantile
# Fcc = (Fcc-lower_quantile[:, np.newaxis])/scaling_factor[:, np.newaxis]

#%% plot oasis results for list of cells
cids = np.arange(0, 5)+40

fig, axarr = plt.subplots(nrows=len(cids), ncols=1, figsize=(12, 10), sharey='col', tight_layout=True)

for cid, ax in zip(cids, axarr.flat):
    print(cid)
    #ax.plot(C_qtn[cid, :])
    #ax.plot(C_qtn[cid, :] - oasis_results['bl'][cid])
    #sn = oasis_results['sn'][cid]
    y = C_qtn[cid, :] - oasis_results['bl'][cid]
    #ax.axhline(np.percentile(y, 50))

    ax.plot(y)
    ax.plot(oasis_results['C_dec'][cid, :])
    ax.set_title(f"cid={cid} |"
                 f" snr={oasis_results['sn'][cid]:.3f}|"
                 f" g={oasis_results['g'][cid]}|"
                 f" lambda={oasis_results['lam'][cid]:.3f} | "
                 )
plt.show()

#%%
# F0 = scipy.ndimage.percentile_filter(Fc, 8, (1, 350))
#
# cids = np.arange(0, 5)+100
# fig, axarr = plt.subplots(nrows=len(cids), ncols=1, sharey='col',
#                           figsize=(12, 10),
#                           tight_layout=True)
#
# for cid, ax in zip(cids, axarr.flat):
#     ax.plot(Fc[cid, :])
#     ax.plot(F0[cid, :])
# plt.show()
# #%%
#
# fig, axarr = plt.subplots(nrows=len(cids), ncols=1, figsize=(12, 10), sharey='col', tight_layout=True)
# for cid, ax in zip(cids, axarr.flat):
#     ax.plot(Fc[cid, :])
#     ax.plot(F[cid, :])
#     ax.plot(F_smoothed[cid, :])
#     ax.plot(Fneu_smoothed[cid, :])
# plt.show()
# #%% plot bulk fluorescence
#
# fig, axarr = plt.subplots(nrows=2, ncols=1)
# Ftot = F.sum(axis=0) - 0.7* Fneu.sum(axis=0)
# F0 = scipy.ndimage.percentile_filter(Ftot, 10, 200)
#
# axarr[0].plot(Ftot)
# axarr[0].plot(F0)
# #axarr[0].plot(Fneu.sum(axis=0))
# axarr[1].plot((Ftot-F0)/F0)
# plt.show()
