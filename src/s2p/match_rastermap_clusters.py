from itertools import chain
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pydantic
import seaborn as sns
import xarray as xr
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from scipy.optimize import linear_sum_assignment
from scipy.stats import zscore

import trial_tensors
from config import *
from pydantic_models import FlatFlyAcquisitions

movie_types = ['kiwi', 'control1', 'control1_top2_ramps', 'kiwi_ea_eb_only']

manifest_json = Path("/local/storage/Remy/natural_mixtures/manifestos/flat_linked_thor_acquisitions.json")
flat_acqs = pydantic.parse_file_as(List[FlatFlyAcquisitions], manifest_json)
flat_acqs = [item for item in flat_acqs if item.movie_type in movie_types]

with CONFIG_DIR.joinpath('heatmap_stim_ord.json').open('r') as f:
    heatmap_ords = json.load(f)


# %%


def np_pearson_corr(x, y):
    """ Computes correlation between the rows/columns of 2 arrays."""
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


# %%

def make_data_file(flacq):
    stat_file = NAS_PROC_DIR.joinpath(flacq.s2p_stat_file)
    print(stat_file)
    ds = xr.load_dataset(stat_file.with_name('xrds_suite2p_outputs_xid1.nc'))

    ds_xid = ds.groupby('xid1').mean(dim='cells')
    ds_xid = ds_xid.assign_attrs(ds.attrs)
    ds_xid_trials = trial_tensors.make_xrds_trial_tensor(ds_xid, trial_ts=None)
    ds_xid.to_netcdf(stat_file.with_name('xrds_suite2p_xid1_clustmeans_traces.nc'))
    ds_xid_trials.to_netcdf(stat_file.with_name('xrds_suite2p_xid1_clustmeans_trials.nc'))

    xrds_mean_peak = ds_xid_trials.sel(time=slice(2, 8)).max(dim='time')
    xrds_mean_baseline = ds_xid_trials.sel(time=slice(-5, -1)).mean(dim='time')
    xrds_std_baseline = ds_xid_trials.sel(time=slice(-5, -1)).std(dim='time')
    xrds_peak_amp = xrds_mean_peak - xrds_mean_baseline
    xrds_peak_amp = xrds_peak_amp.assign_attrs(ds.attrs)
    xrds_peak_amp.to_netcdf(stat_file.with_name('xrds_suite2p_xid1_clustmeans_peak_amp.nc'))

    ds_peak_amp_mean = xrds_peak_amp.groupby('stim').mean()
    ds_peak_amp_mean = ds_peak_amp_mean.assign_attrs(ds.attrs)
    ds_peak_amp_mean.to_netcdf(stat_file.with_name('xrds_suite2p_xid1_clustmeans_peak_amp_mean.nc'))

    return True


# %% load datasets into file


from sklearn.preprocessing import maxabs_scale

save_plots = True
da_name = 'Fc'
zscore_axis = 'none'  # ['not', 'row', 'col']

for panel in movie_types:
    fig_list = []

    if panel in ['kiwi', 'control1']:
        stim_ord = heatmap_ords[panel]
    else:
        stim_ord = heatmap_ords[panel][0]
    panel_flacqs = [item for item in flat_acqs if item.movie_type == panel]

    ds_files = [NAS_PROC_DIR.joinpath(item.s2p_stat_file).with_name('xrds_suite2p_xid1_clustmeans_peak_amp_mean.nc')
                for item in panel_flacqs]
    ds_list = [xr.load_dataset(file) for file in ds_files]
    df_list = [ds.data_vars[da_name].T.to_pandas()[stim_ord] for ds in ds_list]
    df_list = [df.apply(maxabs_scale, axis=1, ) for df in df_list]

    if zscore_axis == 'col':
        df_list = [df.apply(zscore, axis=0) for df in df_list]
    if zscore_axis == 'row':
        df_list = [df.apply(zscore, axis=1) for df in df_list]

    df_ref = df_list[0]

    fig, axarr = plt.subplots(1, len(ds_list), figsize=(11, 8.5), constrained_layout=True)

    for df, ax in zip(df_list, axarr.flat):
        cost = np_pearson_corr(df_ref.to_numpy().T, df.to_numpy().T)
        row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

        sns.heatmap(df.loc[col_ind, :], ax=ax, cmap='Spectral_r', robust=True)
        ax.tick_params(axis='y', labelrotation=0)

    fig.suptitle(f"{panel}: peak_amp of {da_name}\n{zscore_axis} zscored")
    fig.supylabel("rastermap clusters")
    fig.supxlabel("odors")
    fig_list.append(fig)

    if save_plots:
        save_file = NAS_PLOT_DIR.joinpath("rastermap_cluster_types_matched",
                                          f"{panel}_{da_name}__xid1_clustermeans__{zscore_axis}_zscored.pdf")
        print(save_file.name)
        # fig.savefig(save_file)
        # fig.savefig(save_file.with_suffix('.png'))

    plt.show()
# %%
save_plots = False
da_name = 'Fc_normed'

for panel in ['control1_top2_ramps', 'kiwi_ea_eb_only']:
    panel_flacqs = [item for item in flat_acqs if item.movie_type == panel]

    stim_grid = stim_grids[panel]
    stim_ord = list(chain(*stim_grids[panel]))
    stim_ord_map = zip(stim_ord, range(len(stim_ord)))

    # load files
    ds_files = [NAS_PROC_DIR.joinpath(item.s2p_stat_file)
                    .with_name('xrds_suite2p_xid1_clustmeans_peak_amp_mean.nc')
                for item in panel_flacqs]
    ds_list = [xr.load_dataset(file) for file in ds_files]
    df_list = [ds.data_vars[da_name].T.to_pandas() for ds in ds_list]

    for ds, flacq in zip(ds_list, panel_flacqs):
        stim_ord_idx = [stim_ord.index(item) for item in ds.stim.to_numpy()]
        ds = ds.assign_coords(stim_ord_idx=('stim', stim_ord_idx))
        ds = ds.sortby('stim_ord_idx')

    # reshape
        fig = plt.figure(figsize=(11, 8.5))
        # grid = ImageGrid(fig, 111, nrows_ncols=(5, 8),
        #                  # ngrids=40,
        #                  share_all=False,
        #                  axes_pad=0.15, cbar_size='15%', cbar_pad=0.05,
        #                  cbar_mode="each", cbar_location='right')

        grid = ImageGrid(fig, 111, nrows_ncols=(6, 7),
                         # ngrids=40,
                         share_all=True,
                         axes_pad=(0.4, 0.3),  # (horizontal, vertical)
                         label_mode='L',
                         cbar_size='10%', cbar_pad=0.05,
                         cbar_mode="each", cbar_location='right')

        for ax, iclust in zip(grid, range(40)):
            corrmat = ds.Fc_normed.sel(xid1=iclust).to_numpy()
            corrmat = corrmat.reshape((4, 4))
            sns.heatmap(data=corrmat, ax=ax, cbar_ax=ax.cax,
                        # cbar_kws=dict(orientation='horizontal')
                        )
            ax.cax.tick_params(labelsize=8)
            ax.set_title(f"{iclust:d}")
        fig.suptitle(flacq.filename_base())
        plt.show()
        save_file = f"{flacq.filename_base()}__peakampgrid.pdf"
        fig.savefig(NAS_PLOT_DIR.joinpath('peak_amp_grids', save_file))
    #
    # df_list = [df.loc[:, ]]
#%%


fig = plt.figure(figsize=(11, 8.5))
grid = ImageGrid(fig, 111, nrows_ncols=(6, 6), share_all=True,
                 axes_pad=0.3, cbar_size='15%',
                 cbar_mode="each", cbar_location= 'right')

for ax, iclust in zip(grid, range(40)):
    corrmat = ds.sel(xid1=iclust).Fc_normed.to_numpy()
    corrmat = corrmat.reshape((4, 4))
    sns.heatmap(data=corrmat, ax=ax, cbar_ax=ax.cax,
                # cbar_kws=dict(orientation='horizontal')
                )
    ax.set_title(f"{iclust:d}")
plt.show()

# %%

panel = 'kiwi'
da_name = 'Fc_normed'

panel_flacqs = [item for item in flat_acqs if item.movie_type == panel]
ds_files = [NAS_PROC_DIR.joinpath(item.s2p_stat_file).with_name('xrds_suite2p_xid1_clustmeans_peak_amp_mean.nc')
            for item in panel_flacqs]
ds_list = [xr.load_dataset(file) for file in ds_files]
df_list = [ds.data_vars[da_name].T.to_pandas() for ds in ds_list]

df_ref = df_list[0]
xid_assignments = np.empty((df_ref.shape[0], len(df_list)), dtype='int')

for i, df in enumerate(df_list):
    cost = np_pearson_corr(df_ref.to_numpy().T, df.to_numpy().T)
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    xid_assignments[:, i] = col_ind
# %%
# %%
# cost = squareform(pdist('correlation'))

n_ds = len(ds_list)

plt.show()

# %%
# clusters x trials
# sklearn munkres
plt.plot(ds_xid.Fc_normed.sel(xid1=10))
plt.show()
