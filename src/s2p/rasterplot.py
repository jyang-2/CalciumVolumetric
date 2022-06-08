from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from rastermap.mapping import Rastermap
from scipy.stats import zscore

plt.rcParams.update({'pdf.fonttype': 42,
                     'text.usetex': False})

suite2p_ops = {'n_components': 1, 'n_X': 100, 'alpha': 1., 'K': 1.,
               'nPC': 200, 'constraints': 2, 'annealing': True, 'init': 'pca',
               }

NAS_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")


def get_vmin(imdata, baseline_prctile=20, cell_prctile=30):
    """Compute vmin for displaying rasterplots.

    Args:
        imdata ():
        baseline_prctile ():
        cell_prctile ():

    Returns:
        vmin (np.float): recommended vmin value for rasterplot colormap

    Example::

        ax.imshow(raster_imdata,<br>
                        aspect='auto', cmap='gray_r',<br>
                        vmin=get_vmin(raster_imdata, 20, 30),<br>
                        vmax=get_vmax(raster_imdata, 90, 90))<br>

    """
    bl = np.percentile(imdata, baseline_prctile, axis=1)
    return np.percentile(imdata, cell_prctile)


def get_vmax(imdata, peak_prctile=90, cell_prctile=90):
    """Compute vmax for displaying rasterplots.

    Args:
        imdata (np.ndarray): image or rastermap data
        peak_prctile ():
        cell_prctile ():

    Returns:
        vmin (np.float): recommended vmin value for rasterplot colormap

    Example::

        ax.imshow(raster_imdata,<br>
                        aspect='auto', cmap='gray_r',<br>
                        vmin=get_vmin(raster_imdata, 20, 30),<br>
                        vmax=get_vmax(raster_imdata, 90, 90))<br>

    """
    bl = np.percentile(imdata, peak_prctile, axis=1)
    return np.percentile(imdata, cell_prctile)


def suite2p_reordered_trial_display(sp, df_stimulus, frametimes, nbin=None, nplotrows=400,
                                    pre_stim=-20, post_stim=40,
                                    figure_kwargs={}, imshow_kwargs={}):
    df = df_stimulus.sort_values(by='stimname')
    n_cells = sp.shape[0]

    if nbin is None:
        # selects nbin so that the rasterplot has ~400 rows
        nbin = int(n_cells / nplotrows)
        nbin = round(nbin / 10) * 10

    img = running_average(sp, nbin)
    img = zscore(img, axis=1)

    # plot each trial
    fig_kwargs = dict(nrows=1, ncols=df_stimulus.shape[0], figsize=(24, 12))
    fig_kwargs.update(figure_kwargs)
    fig, axarr = plt.subplots(**fig_kwargs)

    im_kwargs = dict(vmin=-0.5, vmax=0.5, aspect='auto', cmap='gray_r', origin='lower')
    im_kwargs.update(imshow_kwargs)

    for odor, t_stim, ax in zip(df['stimname'], df['stim_ict'], axarr.flat):
        # time intervals, before and after stimulus onset
        # t = [start, stim onset, end] for trial
        t = [t_stim + pre_stim, t_stim, t_stim + post_stim]

        idx = np.floor(np.interp(t, frametimes, np.arange(frametimes.size))).astype(int)
        print(idx, type(idx[0]))

        im = ax.imshow(img[:, idx[0]:idx[-1]], **im_kwargs)
        ax.set_title(f"{odor}")
        ax.axvline(idx[1] - idx[0], linestyle='--', linewidth=1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    axarr[0].get_yaxis().set_visible(True)
    fig.colorbar(im)

    return fig, axarr


def compute_cluster_trace(cells_x_time, xid):
    uxid = np.unique(xid)
    n_clust = uxid.size
    mean_clust_traces = np.zeros((n_clust, cells_x_time.shape[1]))

    for i, iclust in enumerate(uxid):
        mean_clust_traces[i, :] = np.nanmean(cells_x_time[xid == iclust, :], axis=0)
    return mean_clust_traces, uxid


def norm_traces(traces):
    traces = np.squeeze(traces)
    traces = zscore(traces, axis=-1)
    traces = np.maximum(-4, np.minimum(8, traces)) + 4
    traces /= 12
    return traces


def prep_to_plot(spF):
    spF = zscore(spF, axis=1)
    spF = np.minimum(8, spF)
    spF = np.maximum(-4, spF) + 4
    spF /= 12
    return spF


def get_cluster_counts(xid):
    """
    Get counts for # of neurons assigned to each unique cluster (useful for plotting).

    Args:
        xid (np.ndarray): 1d np.array, containing cluster assignments from Rastermap.model

    Returns:
         xid_counts (dict): {xid: # of occurrences}
    """
    xid_counts = Counter(xid)
    xid_counts = dict(sorted(xid_counts.items()))
    return xid_counts


# this function performs a running average filter over the first dimension of X
# (faster than doing gaussian filtering)

def running_average(x, nbin=50):
    Y = np.cumsum(x, axis=0)
    Y = Y[nbin:, :] - Y[:-nbin, :]
    return Y


def suite2p_display(S, isort=None, nbin=50, figure_kwargs={}, imshow_kwargs={}):
    if isort is not None:
        S = S[isort, :]

    # Sfilt = running_average(S, nbin)
    # Sfilt = zscore(Sfilt, axis=1)

    figure_kwargs0 = dict(figsize=(11, 8.5))
    figure_kwargs0.update(figure_kwargs)

    vmin = get_vmin(S)
    vmax = get_vmax(S)
    imshow_kwargs0 = dict(vmin=vmin, vmax=vmax, aspect='auto', cmap='gray_r', origin='lower')
    imshow_kwargs0.update(imshow_kwargs)

    fig1, ax = plt.subplots(**figure_kwargs0)
    img = ax.imshow(S, **imshow_kwargs0)
    fig1.colorbar(img)

    plt.xlabel('time points')
    plt.ylabel('sorted neurons')
    return fig1, ax


def run_initial_rastermap_embedding(stat_file, save=True):
    """ Run initial rastermap embedding (on xrds_suite2p_outputs.nc) and save results"""
    if stat_file.with_name('embedding_allcells.npy').is_file():
        print(f"NO! ALREADY RAN INITIAL RASTERMAP EMBEDDING!!!!")
        return None
    else:
        F = np.load(stat_file.with_name('F.npy'))
        Fneu = np.load(stat_file.with_name('Fneu.npy'))
        Fc = F - 0.7 * Fneu
        Fc_normed = norm_traces(Fc)

        ops = {'n_components': 1,
               'n_X': 40,
               'alpha': 1.,
               'K': 1.,
               'nPC': 200, 'constraints': 2, 'annealing': True, 'init': 'pca'}

        model = Rastermap(**ops)
        embedding = model.fit_transform(Fc_normed)

        if save:
            np.save(stat_file.with_name('embedding_allcells.npy'),
                    dict(ops=ops, model=model, data_description='Fc_normed'))

            np.save(stat_file.with_name('embedding_allcells_asdict.npy'),
                    dict(ops=ops, model=model.__dict__, data_description='Fc_normed'))
            print(f"{stat_file.with_name('embedding_allcells.npy')} saved.")

        return model


def plot_split_rastermap_from_xrds(ds, data_var_key='Fc_normed'):
    """
    Plot split rastermap, interpreting from 
    Args:
        ds (): 

    Returns:

    """""
    xid_counts = get_cluster_counts(ds.xid.to_numpy())
    hratio = [v for v in xid_counts.values()]

    fig, axarr = plt.subplots(nrows=len(xid_counts), ncols=1,
                              figsize=(11, 8.5),
                              gridspec_kw=dict(height_ratios=hratio,
                                               wspace=0.05,
                                               hspace=0.05)
                              )
    da = ds['Fc_normed']
    da = da.sortby(['xid', 'embedding'])

    vmin = get_vmin(da.data)
    vmax = get_vmax(da.data)

    for ax, (label, da_grp) in zip(axarr.flat, da.groupby('xid')):
        data = da_grp.data
        ax.imshow(data, vmin=vmin, vmax=vmax, cmap='gray_r', aspect='auto')

        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

        ax.set_ylabel(str(label), rotation=0, fontsize=8,
                      loc='center',
                      va='center',  # {center, 'top', 'bottom', 'center_baseline'}
                      labelpad=6
                      )

        # ax.spines[['top', 'left', 'right', 'bottom']].set_color('red')
        ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
    axarr[0].set_title(f"# cells: {ds.dims['cells']}")
    return fig, axarr


def plot_initial_rastermap_clustering(file):
    with xr.open_dataset(file, engine='h5netcdf') as ds:
        embedding = np.load(file.with_name('embedding_allcells.npy'), allow_pickle=True).item()
        model = embedding['model']

        Fc = ds['Fc'].to_numpy()
        Fc_normed = norm_traces(Fc)
        ds = ds.assign(Fc_normed=(('cells', 'time'), Fc_normed))

        ds = ds.assign_coords(dict(xid=('cells', model.xid)))
        ds = ds.assign_coords(dict(embedding=('cells', model.embedding.squeeze())))
        ds = ds.sortby(['xid', 'embedding'])

        xid_counts = get_cluster_counts(model.xid)
        hratio = [v for v in xid_counts.values()]

        fig, axarr = plt.subplots(nrows=len(xid_counts), ncols=1,
                                  figsize=(8.5, 11),
                                  gridspec_kw=dict(height_ratios=hratio,
                                                   wspace=0.01,
                                                   hspace=0.01)
                                  )
        da = ds['Fc_normed']
        vmin = get_vmin(da.data)
        vmax = get_vmax(da.data)

        for ax, (label, da_grp) in zip(axarr.flat, da.groupby('xid')):
            data = da_grp.data
            ax.imshow(data, vmin=vmin, vmax=vmax, cmap='gray_r', aspect='auto')

            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])

            ax.set_ylabel(str(label), rotation=0, fontsize=8, labelpad=6)

            ax.spines[['top', 'left', 'right', 'bottom']].set_color('red')
        print(f"# cells: {ds.dims['cells']}")
        axarr[0].set_title(f"# cells: {ds.dims['cells']}")
        fig.suptitle(file.relative_to(NAS_PROC_DIR), fontsize=10, verticalalignment='center')
    return fig, axarr


def main(xrds_file):
    # plot initial rastermap
    fig1, axarr1 = plot_initial_rastermap_clustering(xrds_file)
    plt.show()

    pdf_file = xrds_file.with_name('rastermap_embedding_allcells.pdf')
    if ~pdf_file.is_file():
        with PdfPages(pdf_file) as pdf:
            pdf.savefig(fig1)
            print('\trastermap_embedding_allcells.pdf saved.')
    else:
        print(f"\tpdf file already exists: {pdf_file.relative_to(NAS_PROC_DIR)}")

    # compute and load in datasetells', 'time'), Fc_normed))
    ds = xr.open_dataset(xrds_file)
    embedding = np.load(xrds_file.with_name('embedding_allcells.npy'), allow_pickle=True).item()
    model = embedding['model']

    Fc = ds['Fc'].to_numpy()
    Fc_normed = norm_traces(Fc)

    ds = ds.assign(Fc_normed=(('cells', 'time'), Fc_normed))
    ds = ds.assign_coords(dict(xid=('cells', model.xid)))
    ds = ds.assign_coords(dict(embedding=('cells', model.embedding.squeeze())))
    ds = ds.sortby(['xid', 'embedding'])
    ds.attrs['good_xid'] = embedding['good_xid']
    print(ds)

    ds.to_netcdf(xrds_file.with_name('xrds_suite2p_outputs_xid0.nc'))

    #
    good_xid = embedding['good_xid']
    ds_good_xid = ds.where(ds.xid.isin(good_xid), drop=True).sortby(['xid', 'xid'])

    # plot good clusters only
    fig2, axarr2 = plot_split_rastermap_from_xrds(ds_good_xid)
    fig2.suptitle(xrds_file.relative_to(NAS_PROC_DIR), fontsize=10, verticalalignment='center')
    plt.show()

    with PdfPages(xrds_file.with_name('rastermap_embedding_allcells_good_xid.pdf')) as pdf:
        pdf.savefig(fig2)
        print(f'\trastermap_embedding_allcells_good_xid.pdf saved.')

    ds.close()

    return xrds_file.with_name('rastermap_embedding_allcells_good_xid.pdf')


if __name__ == '__main__':
    xrds_file_list = sorted(list(NAS_PROC_DIR.rglob('suite2p/combined/xrds_suite2p_outputs.nc')))

    for i, item in enumerate(xrds_file_list):
        print(f'\n---')
        print(f"{i}\t{item.relative_to(NAS_PROC_DIR)}")
        main(item)

    # stat_file_list = list(NAS_PROC_DIR.rglob("combined/stat.npy"))
    # stat_file_list = list(filter(lambda x: not x.with_name('embedding_allcells.npy').is_file(),
    #                              stat_file_list))
    #
    # for i, item in enumerate(stat_file_list):
    #     print(f"{i}\t{item.relative_to(NAS_PROC_DIR)}")
    # %%

    # for stat_file in stat_file_list:
    #     if stat_file.with_name('embedding_allcells.npy').is_file():
    #         print(f"{stat_file.with_name('embedding_allcells.npy').relative_to(NAS_PROC_DIR)} exists already.")
    #     else:
    #         model = run_initial_rastermap_embedding(stat_file, save=True)
#
# ops_good_xid = {'n_components': 2,
#                 'n_X': 40,
#                 'alpha': 1.,
#                 'K': 1.,
#                 'nPC': 200, 'constraints': 2, 'annealing': True, 'init': 'pca'}
#
# model1 = Rastermap(**ops_good_xid)
# model1.fit(ds_good_xid.Fc_normed.to_numpy())
#
# #%%
#
# cell_embedding = model1.embedding[:, 0]
# time_embedding = model1.embedding[:, 1]
# #%%
# ds_good_xid = ds_good_xid.assign_coords(dict(cell_embedding=('cells', cell_embedding)))
# ds_good_xid = ds_good_xid.assign_coords(dict(time_embedding=('time', time_embedding)))
# ds_good_xid = ds_good_xid.sortby('time_embedding')
# ds_good_xid = ds_good_xid.sortby('cell_embedding')
#
# fig2, axarr2 = plot_split_rastermap_from_xrds(ds_good_xid)
# plt.show()
# # %%
# plt.scatter(time_embedding, cell_embedding, s=20, )
# # %%
# Fc_tot = Fc.sum(axis=0)
# fig, ax = plt.subplots(1, 1, figsize=(8, 2))
# ax.plot(zscore(Fc_tot, axis=-1))
# plt.show()
# %%
#         plt.show()

#
# xrds_file_list = sorted(list(NAS_PROC_DIR.rglob('kiwicombined/xrds_suite2p_outputs.nc')))
# kiwi_file_list = [xrds_file_list[i] for i in [0, 2, 4, 7]]
#
# for i, item in enumerate(xrds_file_list):
#     print(f"\n{i}.\t{item}")
# # %%
# for ds_file in xrds_file_list:
#     print(f"\n{ds_file}")
#


#
#     ops = {'n_components': 1,
#            'n_X': 40, 'alpha': 1., 'K': 1.,
#            'nPC': 200, 'constraints': 2, 'annealing': True, 'init': 'pca'}
#
#     model = Rastermap(**ops)
#     embedding = model.fit_transform(Fc_normed)
#
#     np.save(ds_file.with_name('embedding_allcells.npy'),
#             dict(ops=ops, model=model, data_description='Fc_normed'))
#
#     np.save(ds_file.with_name('embedding_allcells_asdict.npy'),
#             dict(ops=ops, model=model.__dict__, data_description='Fc_normed'))
#
# ops = {'n_components': 1,
#        'n_X': 40,
#        'alpha': 1.,
#        'K': 1.,
#        'nPC': 200, 'constraints': 2, 'annealing': True, 'init': 'pca'}
#
# time_model = Rastermap(**ops)
# unorm = (model.u)
# time_model.fit(Fc_normed)
# %%
#
# pdf = PdfPages("/local/storage/Remy/natural_mixtures/report_data/"
#                "xrds_suite2p_outputs__allcells__rasterclust.pdf")
#
# for ds_file in xrds_file_list:
#     with xr.open_dataset(ds_file, engine='h5netcdf') as xrds_suite2p_outputs:
#         print(f"\n{ds_file}")
#         stat_file = ds_file.with_name('stat.npy')
#         Fc = xrds_suite2p_outputs['Fc'].to_numpy()
#         Fc_normed = norm_traces(Fc)
#
#         if stat_file.with_name('embedding.all_cells.npy').is_file():
#             embedding = np.load(ds_file.with_name('embedding_allcells.npy'), allow_pickle=True).item()
#             model = embedding['model']
#         else:
#             model = run_initial_rastermap_embedding(stat_file, save=True)
#
#         xrds_suite2p_outputs = xrds_suite2p_outputs.assign_coords(dict(xid=('cells', model.xid)))
#         xrds_suite2p_outputs = xrds_suite2p_outputs.assign_coords(dict(embedding=('cells', model.embedding.squeeze())))
#
#         ds = xrds_suite2p_outputs.coords.to_dataset()
#         ds = ds.assign(Fc_normed=(('cells', 'time'), Fc_normed))
#         ds = ds.sortby(['xid', 'embedding'])
#         da = ds['Fc_normed']
#
#         xid_counts = get_cluster_counts(model.xid)
#
#         hratio = [v for v in xid_counts.values()]
#
#         fig, axarr = plt.subplots(nrows=len(xid_counts), ncols=1,
#                                   figsize=(8.5, 11),
#                                   gridspec_kw=dict(height_ratios=hratio,
#                                                    wspace=0.005,
#                                                    hspace=0.002)
#                                   )
#
#         vmin = get_vmin(da.data)
#         vmax = get_vmax(da.data)
#
#         for ax, (label, da_grp) in zip(axarr.flat, da.groupby('xid')):
#             data = da_grp.data
#             ax.imshow(data, vmin=vmin, vmax=vmax, cmap='gray_r', aspect='auto')
#
#             ax.axes.xaxis.set_ticks([])
#             ax.axes.yaxis.set_ticks([])
#
#             ax.set_ylabel(str(label), rotation=0, fontsize=8, labelpad=6)
#
#             ax.spines[['top', 'left', 'right', 'bottom']].set_color('red')
#
#             fig.suptitle(ds_file.relative_to(NAS_PROC_DIR), fontsize=10)
#
#         plt.show()
#         pdf.savefig(fig)
#
# pdf.close()
# %%
# xrds_suite2p_outputs = xrds_suite2p_outputs.assign_coords(dict(xid=('cell_ids', model.embedding),
#                                                                embedding=('cell_ids', )))
#
# xrds_suite2p_outputs.assign_coords()

# for ds_file in xrds_file_list:
#     with xr.open_dataset(ds_file, engine='h5netcdf') as xrds_suite2p_outputs:
#         print(f"\n{ds_file}")
#         fps = round(1 / np.diff(xrds_suite2p_outputs.stack_times).mean())
#
#         Fc = xrds_suite2p_outputs['Fc'].to_numpy()
#         traces = process_traces(Fc, win=60 * fps)
#         print(f"win={60 * fps}")
#
#         embedding = np.load(ds_file.with_name('embedding_allcells.npy'), allow_pickle=True).item()
#         model = embedding['model']
#
#         xrds_suite2p_outputs = xrds_suite2p_outputs.assign_coords(dict(xid=('cells', model.xid)))
#         xrds_suite2p_outputs = xrds_suite2p_outputs.assign_coords(dict(embedding=('cells', model.embedding.squeeze())))
#
#         ds = xrds_suite2p_outputs.coords.to_dataset()
#         for k, v in traces.items():
#             ds = ds.assign({k: (('cells', 'time'), v)})
#         ds = ds.sortby(['xid', 'embedding'])
#         ds = ds.assign_attrs(xrds_suite2p_outputs.attrs)
#         ds.to_netcdf(ds_file.with_name('xrds_s2p_traces.nc'))
# data_vars = {}
#
# for k in ds.data_vars.keys():
#     cells_x_time = ds[k].to_numpy()
#     trials_x_cells_x_time = trial_tensors.make_trial_tensor(cells_x_time, ts, olf_ict, trial_ts)
#     da = xr_helpers.make_xrda_traces(trials_x_cells_x_time,
#                                      stim,
#                                      xrds.cell_ids.to_numpy(),
#                                      trial_ts,
#                                      attrs=xrds.attrs)
#     data_vars[k] = da


# # %%
# def main(statfile, save_outputs=True, nbin=None, figure_kwargs={}, imshow_kwargs={}, title_str=None):
#     if isinstance(statfile, str):
#         statfile = Path(statfile)
#
#     stat = np.load(statfile, allow_pickle=True)[0]
#     iscell = np.load(statfile.with_name('iscell.npy'))
#     cellprob = iscell[:, 1]
#     iscell = iscell[:, 0].astype(np.bool_)
#     Fcell = np.load(statfile.with_name('F.npy'))
#     Fneu = np.load(statfile.with_name('Fneu.npy'))
#     F = Fcell - Fneu * 0.7
#
#     clf_thresh = cellprob[iscell].min()
#
#     # select good cells
#     sp = F[iscell, :]
#     spn = norm_traces(sp)
#
#     if nbin is None:
#         nbin = math.ceil(sp.shape[0] / 400)
#         print(f"nbin = {nbin}")
#
#     use_default = True
#     if use_default:
#         ops = suite2p_ops
#     else:
#         ops = {'n_components': 2,  # (default: 2) dimension of the embedding space
#                'n_X': 40,  # (default: 40) size of the grid on which the Fourier modes are rasterized
#                'alpha': 1.0,  # (default: 1.0) exponent of the power law enforced on component n as: 1/(K+n)^alpha
#                'K': 1.0,  # (default: 1.0) additive offset of the power law enforced on component n as: 1/(K+n)^alpha
#                'nPC': 200,  # (default: 400) how many of the top PCs to use during optimization
#                'constraints': 2,  # (default: 1.0) exponent of the power law enforced on component n as: 1/(K+n)^alpha
#                'annealing': True,
#                'init': 'pca',  # (default: 'pca') can use 'pca', 'random', or a matrix n_samples x n_components
#                }
#     model = Rastermap(**ops)
#
#     # fit does not return anything, it adds attributes to model
#     # attributes: embedding, u, s, v, isort1
#     embedding = model.fit_transform(spn)
#     isort1 = np.argsort(embedding[:, 0])
#
#     # %% plot rastermap
#     spp = prep_to_plot(spn)
#     fig, ax = suite2p_display(spp, isort=isort1, nbin=nbin,
#                               figure_kwargs=figure_kwargs,
#                               imshow_kwargs=imshow_kwargs)
#
#     if title_str is not None:
#         ax.set_title(f"{title_str}\nclf threshold={clf_thresh:.3f}; # cells={sp.shape[0]}")
#     plt.show()
#
#     if save_outputs:
#         np.save(statfile.with_name('embedding.npy'), model.__dict__)
#         fig.savefig(statfile.with_name('rastermap.png'))
#         fig.savefig(statfile.with_name('rastermap.pdf'))
#
#     return model, fig, ax
#
#
# # %%
# if __name__ == 'main':
#     NAS_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")
#     TEMP_SAVE_DIR = Path("/local/gerty/Remy's Dropbox Folder/HongLab @ Caltech Dropbox/Remy/temp_rastermap_embeddings")
#
#     stat_file_list = sorted(list(NAS_PROC_DIR.rglob("downsampled_3/suite2p/combined/stat.npy")))
#
#     fig_list = []
#     for stat_file in stat_file_list:
#         model, fig, ax = main(stat_file,
#                               save_outputs=True,
#                               imshow_kwargs=dict(vmin=0.3, vmax=3),
#                               figure_kwargs=dict(tight_layout=True),
#                               title_str=stat_file.relative_to(NAS_PROC_DIR))
#         fig_list.append(fig)
#
#     with PdfPages('../../plots/natural_mixtures_rasterplots_vmin0_vmax3.pdf') as pdf:
#         for fig in fig_list:
#             pdf.savefig(fig)
#         # save_file = "__".join(stat_file.with_name('embedding.npy').relative_to(NAS_PROC_DIR).parts)
#         # np.save(TEMP_SAVE_DIR.joinpath(save_file), model.__dict__)
#
#     # folder = Path.cwd().joinpath('data', 'processed_data', '2021-08-24') \
#     #     .rglob('**/suite2p/combined/stat.npy')
#     # folder = sorted(list(folder))
#     # for file in folder:
#     #     main(file)
# # mdl, fig_rastermap = main(sys.argv[1])
# ("2022-02-10/1/kiwi/downsampled_3/suite2p/combined/embedding_allcells.npy", [14, 15, 16, 17, 18, 30, ]
