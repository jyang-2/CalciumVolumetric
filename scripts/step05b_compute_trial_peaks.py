"""Convert calcium trial timeseries into response amplitude arrays w/ dim. cells x trials.



"""

import copy

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import natmixconfig
import ryeutils

# import hvplot
# import hvplot.xarray
# import hvplot.pandas
# import panel as pn
# import holoviews as hv
# from holoviews import opts
# import seaborn as sns

tom_format = False
if tom_format:
    allowed_kiwi_odor1 = {'2H @ -5',
                          'FUR @ -4',
                          'MS @ -3',
                          'OCT @ -3',
                          'VA @ -3',
                          'control mix @ -1',
                          'control mix @ -2',
                          'control mix @ 0',
                          'pfo @ 0',
                          }
    allowed_control_odor1 = {
        'EA @ -4.2',
        'EB @ -3.5',
        'EtOH @ -2',
        'IAA @ -3.7',
        'IAol @ -3.6',
        'pfo @ 0',
        '~kiwi @ -1',
        '~kiwi @ -2',
        '~kiwi @ 0'}
    allowed_odor1 = allowed_kiwi_odor1.union(allowed_control_odor1)

    probe_odors = ['3mt1p @ -6.0', 't2h @ -6.0']

    partial_odors = ['cmix_no_2h @ 0.0',
                     'cmix_no_1o3ol @ 0.0',
                     'kiwi_no_eta @ 0.0',
                     'kiwi_no_etb @ 0.0',
                     'kiwi_no_etoh @ 0.0',
                     'kiwi_no_etoh @ -1.0',
                     'kiwi_no_etoh @ -2.0',
                     ]
    stim_to_drop = probe_odors + partial_odors

# %%
# flat_acqs = pydantic.parse_file_as(List[FlatFlyAcquisitions], natmixconfig.MANIFEST_FILE)
# flat_acqs = list(filter(lambda x: x.stat_file() is not None, flat_acqs))
# flat_acqs = list(filter(lambda x: not x.is_pair(), flat_acqs))

STAT_DIR_INPUT_FILES = ["xrds_suite2p_trials_xid0.nc"]
""" Files required in ```stat_dir=flacq.stat_file().parent```"""

STAT_DIR_OUTPUT_FILES = [
    "xrds_suite2p_respvec_mean_peak.nc",
    "xrds_suite2p_respvec_max_peak.nc",
    "xrds_suite2p_respvec_max_peak_idx.nc"
]
""" Files created in ```stat_dir=flacq.stat_file().parent```"""


# %%
def has_input_files(stat_file):
    return [stat_file.with_name(item).is_file() for item in STAT_DIR_INPUT_FILES]


def has_output_files(stat_file):
    return [stat_file.with_name(item).is_file() for item in STAT_DIR_OUTPUT_FILES]


# noinspection PyShadowingNames
def subtract_trial_baseline(ds_trials, baseline_range=(-5, 0)):
    """Baseline-corrects calcium timeseries by subtracting the pre-stimulus mean.

    Args:
        ds_trials (xr.Dataset): suite2p outputs w/ shape (trials, cells, time). Stimulus onset at time=0.
        baseline_range (tuple): time period to treat as baseline, default = (-5, 0)

    Returns:
        xr.Dataset: baseline-corrected suite2p outputs w/ dims (trials, cells, time)

    Examples:
        >>> ds_trials = xr.load_dataset("xrds_suite2p_trials_xid0.nc")
        >>> ds_trials_bc = subtract_trial_baseline(ds_trials, baseline_range=(-5, 0))
    """
    ds_trial_baseline_mean = ds_trials.sel(time=slice(*baseline_range)).mean(dim='time')
    return ds_trials - ds_trial_baseline_mean


# noinspection PyShadowingNames
def compute_mean_peaks(ds_trials, peak_range=(2, 8), baseline_range=(-5, 0),
                       subtract_baseline=False):
    """
    Computes neuron response peaks by averaging over `peak_range` (expected .

    Args:
        ds_trials (xr.Dataset): suite2p outputs w/ dims (trials, cells, time)
        peak_range (tuple): time period of expected response peak
        baseline_range (tuple): pre-stimulus time period (stim. onset in `ds_trials` is at t=0.)
        subtract_baseline (bool): whether or not to subtract the mean baseline.
                                    If True, `baseline_range` is used.

    Returns:
        xr.Dataset: trial response amplitudes w/ dim. (trials x cells)

    Examples:
        >>> ds_trials = xr.load_dataset("xrds_suite2p_trials_xid0.nc")
        >>> ds_mean_peak = subtract_trial_baseline(ds_trials, peak_range=(2, 8))

    """

    ds_mean_peak = ds_trials.sel(time=slice(*peak_range)).mean(dim='time')

    if subtract_baseline:
        ds_mean_baseline = ds_trials.sel(time=slice(*baseline_range)).mean(dim='time')
        ds_mean_peak = ds_mean_peak - ds_mean_baseline

    return ds_mean_peak


def compute_max_peaks(ds_trials, peak_range=(2, 8), baseline_range=(-5, 0),
                      subtract_baseline=False):
    """Computes peak amplitudes (trials x cells), taking the max value during time `peak_range`.

    Args:
        ds_trials (xr.Dataset): suite2p outputs w/ dims (trials, cells, time)
        peak_range (tuple): time range (should be post-stimulus onset) to search for peak max.
        baseline_range (tuple): pre-stimulus time period, only used if subtract_baseline=True
        subtract_baseline (bool): whether or not to subtract the mean during `baseline_range`

    Returns:
        ds_max_peak (xr.Dataset): peak maxima, w/ dims (trials, cells)
        ds_maxidx (xr.Dataset): time when each max peak occurred, has dims (trials, cells)

    Examples:
        >>> ds_trials = xr.load_dataset("xrds_suite2p_trials_xid0.nc")
        >>> ds_trials_bc = subtract_trial_baseline(ds_trials, baseline_range=(-5, 0))

    """
    # baseline correct
    if subtract_baseline:
        ds_trials_bc = subtract_trial_baseline(ds_trials, baseline_range=baseline_range)
    else:
        ds_trials_bc = ds_trials

    # look at peak_range period only
    ds_trial_response_period = ds_trials_bc.sel(time=slice(*peak_range))

    ds_max_peak = ds_trial_response_period.max(dim='time')
    ds_maxidx = ds_trial_response_period.idxmax(dim='time')

    return ds_max_peak, ds_maxidx


# %%
def preprocess_ds_peaks(ds_peaks, good_xids_only=True, sort_embedding0=True):
    if good_xids_only:
        ds_proc = ds_peaks.where(ds_peaks.xid0.isin(ds_peaks.attrs['good_xid']), drop=True)
    else:
        ds_proc = copy.deepcopy(ds_peaks)

    if sort_embedding0:
        ds_proc = ds_proc.sortby(['xid0', 'embedding0'])

    return ds_proc


def plot_trial_heatmap(da_trials, title=None):
    """Plots trial timeseries as heatmap w/ (cells x time) using hvplot

    Args:
        da_trials (xr.DataArray): Processed (ideally baseline-subtracted) fluorescence traces, with
                                    dims (trials, cells, time)

    Returns:
        xarray.plot.facetgrid.FacetGrid

    Examples:

    >>> mov_dir = flacq.mov_dir(relative_to=natmixconfig.NAS_PROC_DIR)
    >>> stat_file = natmixconfig.NAS_PROC_DIR / flacq.stat_file()
    >>> title_str = flacq.filename_base()
    >>> ds_trials = xr.load_dataset(stat_file.with_name('xrds_suite2p_trials_xid0.nc'))
    >>> stim_list = ds_trials.stim.to_numpy().tolist()
    >>> da = ds_trials \
        .pipe(subtract_trial_baseline) \
        .where(ds_trials.xid0.isin(ds_trials.attrs['good_xid']), drop=True) \
        .sortby(['xid0', 'embedding0']) \
        .get('Fc_zscore_smoothed')
    >>> hmap_grid = plot_trial_traces(da)


    """
    var_name = da_trials.name

    hmap_grid = da_trials.reset_index('cells') \
        .plot.pcolormesh(x='time', y='cells',
                         col='trials',
                         vmin=0, vmax=3,
                         robust=True,
                         cmap='viridis',
                         # figsize=(8.5, 11),
                         aspect=0.5, size=3,
                         add_labels=True,
                         add_colorbar=False,
                         cbar_kwargs=dict(
                                 orientation='vertical',
                                 shrink=0.1,
                                 aspect=10,
                                 label=var_name),
                         col_wrap=9)

    # adjust so subplots are 0.5 inches from top of figure
    width, height = hmap_grid.fig.get_size_inches()
    top = 1 - 1.0 / height
    title_y = 1 - 0.5 / height

    # adjust colorbar
    ax = hmap_grid.axes[0, -1]
    # divider = make_axes_locatable(hmap_grid.axes[0, -1])
    # cax = divider.append_axes("right", size="10%", pad=0.1)

    cax = inset_axes(ax,
                     width='10%',
                     height="100%",
                     loc='lower left',
                     bbox_to_anchor=(1.05, 0., 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0,
                     )
    plt.colorbar(ax.get_children()[0], cax=cax)

    cbar_right = 1 - 1.0 / width
    hmap_grid.fig.subplots_adjust(top=top,
                                  right=cbar_right
                                  )

    if title is not None:
        hmap_grid.fig.suptitle(title, y=title_y)

    return hmap_grid


# %%


def main(flacq, save_files=False):
    mov_dir = flacq.mov_dir(relative_to=natmixconfig.NAS_PROC_DIR)
    stat_file = natmixconfig.NAS_PROC_DIR / flacq.stat_file()
    title_str = flacq.filename_base()

    # load xarray dataaset
    ds_trials = xr.load_dataset(stat_file.with_name('xrds_suite2p_trials_xid0.nc'))
    attrs = copy.deepcopy(ds_trials.attrs)

    # compute extra stimulus info
    stim_list = ds_trials.stim.to_numpy().tolist()
    df_stim = pd.DataFrame(ryeutils.index_stimuli(stim_list))

    # compute mean peak values
    ds_mean_peak = compute_mean_peaks(ds_trials, peak_range=(2, 8), baseline_range=(-5, 0),
                                      subtract_baseline=True)
    ds_mean_peak = ds_mean_peak.assign_attrs(attrs)

    # compute max peak values
    ds_max_peak, ds_maxidx = compute_max_peaks(ds_trials, subtract_baseline=True)
    ds_max_peak = ds_max_peak.assign_attrs(attrs)
    ds_maxidx = ds_maxidx.assign_attrs(attrs)

    # save peak amplitudes
    if save_files:
        save_file = stat_file.with_name('xrds_suite2p_respvec_mean_peak.nc')
        print(f"\n\tsave ds_mean_peak to: {save_file}")
        ds_mean_peak.to_netcdf(save_file)
        print(f"\tdone!")

        save_file = stat_file.with_name('xrds_suite2p_respvec_max_peak.nc')
        print(f"\n\tsave ds_max_peak to: {save_file}")
        ds_max_peak.to_netcdf(save_file)
        print(f"\tdone!")

        save_file = stat_file.with_name('xrds_suite2p_respvec_max_peak_idx.nc')
        print(f"\n\tsave ds_max_peak_idx to: {save_file}")
        ds_maxidx.to_netcdf(save_file)
        print(f"\tdone!")

    return ds_mean_peak, ds_max_peak, ds_maxidx

# %%
# def rsa(flacq):
#     mov_dir = flacq.mov_dir(relative_to=natmixconfig.NAS_PROC_DIR)
#     stat_file = natmixconfig.NAS_PROC_DIR / flacq.stat_file()
#     title_str = flacq.filename_base()
#
#     # load xarray dataaset
#     ds_trials = xr.load_dataset(stat_file.with_name('xrds_suite2p_trials_xid0.nc'))
#     attrs = copy.deepcopy(ds_trials.attrs)
#
#     # plot trials
#     da = ds_trials \
#         .pipe(subtract_trial_baseline) \
#         .where(ds_trials.xid0.isin(ds_trials.attrs['good_xid']), drop=True) \
#         .sortby(['xid0', 'embedding0']) \
#         .get('Fc_zscore_smoothed')
#
#     hmap_grid = plot_trial_heatmap(da, title=flacq.filename_base())
#     plt.show()
#
#     # trial peak amplitude calculation
#     ds_max_peak, ds_maxidx = compute_max_peaks(ds_trials, subtract_baseline=True)
#     da_peak = ds_max_peak \
#         .where(ds_trials.xid0.isin(ds_trials.attrs['good_xid']), drop=True) \
#         .sortby(['xid0', 'embedding0']) \
#         .get('Fc_zscore_smoothed')
#
#     hmap_peaks = da_peak.reset_index('cells') \
#         .plot.pcolormesh(x='trials', y='cells',
#                          vmin=0, vmax=4,
#                          robust=True,
#                          cmap='viridis',
#                          #aspect=0.5,
#                          #size=8.5,
#                          figsize=(8.5, 11),
#                          add_labels=True,
#                          add_colorbar=True,
#                          cbar_kwargs=dict(
#                                  orientation='vertical',
#                                  shrink=0.25,
#                                  label='Fc_zscore_smoothed'),
#                          )
#     plt.show()
#
#     metric = 'correlation'
#     corrmat = squareform(pdist(da_peak, metric=metric))
#
#     n_trials = da_peak.shape[da_peak.dims.index('trials')]
#     da_corr = xr.DataArray(name='max_peak_corr',
#                            data=corrmat,
#                            dims=['trial_row', 'trial_col'],
#                            coords=dict(
#                                    trial_row=range(n_trials),
#                                    trial_col=range(n_trials),
#                                    # stim_row=('trial_row', mi_stim),
#                                    # stim_col=('trial_col', mi_stim),
#                                    stim_row=('trial_row', pd.MultiIndex.from_frame(df_stim.add_prefix(
#                                            'row_'))),
#                                    stim_col=('trial_col', pd.MultiIndex.from_frame(df_stim.add_prefix(
#                                            'col_')))
#                            )
#                            )
#
#     df_corr = da_corr.to_pandas()
#     df_corr.index = stim_list
#     df_corr.columns = stim_list
#
#     fig, ax = plt.subplots(figsize=(11, 11))
#     sns.heatmap(1 - df_corr,
#                 cmap='RdBu_r',
#                 vmin=-1, vmax=1,
#                 square=True,
#                 ax=ax,
#                 cbar_kws=dict(
#                         orientation='vertical',
#                         shrink=0.25,
#                         label='Fc_zscore_smoothed')
#                 )
#     fig.suptitle(title_str)
#     plt.show()
#     # %%
#     da_rdm = _rdm_over_trials(ds_max_peak['Fc_zscore_smoothed'])
#
#     df_rdm_plot = da_rdm\
#         .sortby(['stim_row', 'trial_row'])\
#         .sortby(['stim_col', 'trial_col'])\
#         .set_index(trial_row='stim_row')\
#         .set_index(trial_col='stim_col')\
#         .to_pandas()
#
#     da_rdm_plot = da_corr.set_index(dict(trial_row='stim_row',
#                                          trial_col='stim_col'
#                                          ),
#                                     append=True)
#     df_stim = pd.DataFrame(ryeutils.index_stimuli(stim_list))
#
#     df_stim_sorted = df_stim\
#         .assign(stim=stim_list)\
#         .sort_values(['stim', 'stim_occ'])
#
#     sort_idx = df_stim_sorted.index.to_numpy()
#     df_corr = df_corr.loc[sort_idx, :].loc[:, sort_idx]
#
#     fig, ax = plt.subplots(1, 1, figsize=(11, 11), constrained_layout=True)
#     sns.heatmap(1 - df_corr,
#                 cmap='RdBu_r',sn
#                 vmin=-1, vmax=1,
#                 square=True,
#                 ax=ax,
#                 cbar_kws=dict(
#                         orientation='vertical',
#                         shrink=0.25,
#                         label='Fc_zscore_smoothed')
#                 )
#     fig.suptitle(title_str)
#
#     ngrid = tidy_plot.assign(trial_row=tidy_plot.trial_row % 3, trial_col=tidy_plot.trial_col % 3) \
#         .hvplot.scatter(x='trial_row', y='trial_col', row='stim_row', col='stim_col')
#
#     da_rdm_plot.reorder_levels({'trial_col': ['stim_col', 'trial_col_level_0'], 'trial_row': [
#         'stim_row', 'trial_row_level_0']}
#
#
#     return True
#
#
# # %%
# # %%
#
#
# def main_tom(flacq, good_xid0_only=False, drop_probe_odors=True, metric='cosine'):
#     print('---')
#     print(flacq.filename_base())
#
#     mov_dir = flacq.mov_dir()
#     stat_file = natmixconfig.NAS_PROC_DIR / flacq.stat_file()
#
#     ds_file = stat_file.with_name('xrds_suite2p_trials_xid0.nc')
#     xrds_suite2p_trials_xid0 = xr.load_dataset(ds_file)
#
#     # # stim list
#     # if drop_probe_odors:
#     #     xrds_good_trials = xrds_suite2p_trials_xid0.where(~xrds_suite2p_trials_xid0.stim.isin(
#     #             stim_to_drop), drop=True)
#     #     stim_list = xrds_good_trials.stim.to_numpy().tolist()
#     #     df_stimuli = pd.DataFrame(ryeutils.index_stimuli(stim_list))
#     #     mask = df_stimuli.to_numpy()
#     #
#     #     xrds_suite2p_trials_xid0.drop_sel(stim=probe_odors)
#
#     ##################################################
#     # make datasets containing only good cluster cells
#     ##################################################
#     if good_xid0_only:
#         xrds_trials = xrds_suite2p_trials_xid0.where(
#                 xrds_suite2p_trials_xid0.xid0.isin(xrds_suite2p_trials_xid0.attrs['good_xid']),
#                 drop=True)
#     else:
#         xrds_trials = copy.deepcopy(xrds_suite2p_trials_xid0)
#
#     # make sure minimum fluorescence values are always above 1.0
#     #   - ds_trials_nonneg: offset `xrds_trials`
#     ds_Fc_min = xrds_trials.min(dim='time').min(dim='trials')
#     neg_mask = ds_Fc_min < 0
#     ds_offset = -1 * ds_Fc_min.where(neg_mask, other=0) + (neg_mask * 1.0)
#     ds_trials_nonneg = xrds_trials + ds_offset
#
#     xrds_mean_peak = xrds_trials.sel(time=slice(2, 8)).mean(dim='time')
#     xrds_mean_baseline = xrds_trials.sel(time=slice(-5, -0.25)).mean(dim='time')
#     xrds_std_baseline = xrds_trials.sel(time=slice(-5, -0.25)).std(dim='time')
#     xrds_peak_amp = xrds_mean_peak - xrds_mean_baseline
#
#     ##########################
#     # pick data var (Fc_zscore_smoothed?) and compute peak amplitude
#     ##########################
#     data_var = 'Fc_zscore_smoothed'
#
#     # peak amp
#     da_peak_amp = xrds_peak_amp[data_var]
#
#     if flacq.bad_trials is not None:
#         da_peak_amp.loc[flacq.bad_trials] = np.nan
#
#     df_odors, df_odors_b = tom.make_m_odors_for_corr(flacq, astype='dataframe')
#
#     mask = (df_odors['odor1'].isin(allowed_odor1)) & (df_odors['repeat'] < 3)
#     mi_odor = pd.MultiIndex.from_frame(df_odors.loc[mask, :].reset_index(drop=True))
#     mi_odor_b = pd.MultiIndex.from_frame(df_odors_b.loc[mask, :].reset_index(drop=True))
#
#     corrmat = 1 - squareform(pdist(da_peak_amp, metric=metric))
#     corrmat_good = corrmat[mask, :]
#     corrmat_good = corrmat_good[:, mask]
#
#     da_corr = xr.DataArray(data=corrmat_good,
#                            dims=['odor', 'odor_b'],
#                            coords=dict(odor=mi_odor,
#                                        odor_b=mi_odor_b,
#                                        )
#                            )
#
#     # panel info
#     panel_info = dict(panel=flacq.panel(),
#                       fly_panel_id=flat_acqs.index(flacq),
#                       date=np.datetime64(flacq.date_imaged, 'ns'),
#                       fly_num=flacq.fly_num)
#     da_corr = da_corr.assign_coords(panel_info)
#     return da_corr
#
#
# da_meancorr = da_corrs.groupby('panel').mean(dim='fly_panel')
# tidy_corrs = da_meancorr.to_dataframe(name='mean_corr').dropna().reset_index()
# tidy_kiwi_corr = tidy_corrs.loc[tidy_corrs.panel == 'kiwi']
# da_plot = tidy_kiwi_corr.groupby(['odor1', 'odor1_b']).mean().reset_index()
#
# stim_ord = ['pfo @ 0', 'EtOH @ -2', 'IAA @ -3.7', 'IAol @ -3.6', 'EA @ -4.2', 'EB @ -3.5',
#             '~kiwi @ 0', '~kiwi @ -2', '~kiwi @ -1', ]
#
# da_plot.sort_values('odor1', key=lambda x: x.argsort(order=stim_ord))
#
# g = sns.relplot(
#         data=da_plot,
#         x="odor1", y="odor1_b", hue="mean_corr", size="mean_corr",
#         palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
#         height=10, sizes=(50, 250), size_norm=(-.2, .8),
# )
# # tidy_kiwi_corr = da_kiwi_meancorr.to_dataframe(name='kiwi_corr').reset_index()
#
# corrs = da_corrs.sel(is_pair=False, is_pair_b=False).copy()
# corrs = corrs.dropna('fly_panel', how='all')
#
# # TO DO make a fn for grouping -> mean (+ embedding number of flies [maybe also w/ a
# # separate list of metadata for those flies] in the DataArray attrs or something)
# # -> maybe require those type of attrs for corresponding natmix plotting fn?
# for panel, garr in corrs.reset_index(['odor', 'odor_b']).groupby('panel',
#                                                                  squeeze=False,
#                                                                  restore_coord_dims=True):
#
#     garr = dropna_odors(garr)
#
#     # garr.mean(...) will otherwise throw out some / all of this information.
#     meta_dict = {
#         'date': garr.date.values,
#         'fly_num': garr.fly_num.values,
#     }
#     if hasattr(garr, 'thorimage_id'):
#         meta_dict['thorimage_id'] = garr.thorimage_id.values
#
#     panel_mean = garr.mean('fly_panel')
#
#     # garr seems to be of shape:
#     # (# flies[/recordings sometimes maybe], # odors, # odors)
#     # TO DO may need to fix. check against dedeuping below
#     n = len(garr)
#
#     fig = natmix.plot_corr(panel_mean, title=f'{panel} (n={n})')
#
#     fig.savefig(corr_plot_root / f'{panel}_mean.{plot_fmt}')
#
#     meta_df = pd.DataFrame(meta_dict)
#     assert len(meta_df) == len(meta_df[['date', 'fly_num']].drop_duplicates())
#
#     assert len(meta_df) == n
#     meta_df.to_csv(corr_plot_root / f'{panel}_flies.csv', index=False)
#
#
# def dropna_odors(arr: xr.DataArray, _checks=True) -> xr.DataArray:
#     """Drops data where all NaN for either a given 'odor' or 'odor_b' index value.
#     """
#     if _checks:
#         notna_before = arr.notnull().sum().item()
#
#     # "dropping along multiple dimensions simultaneously is not yet supported"
#     arr = arr.dropna('odor', how='all').dropna('odor_b', how='all')
#
#     if _checks:
#         assert arr.notnull().sum().item() == notna_before
#
#     return arr
#
#
# corr_plot_root = Path("/local/matrix/Remy-Data/projects/natural_mixtures/for_tom/correlation"
#                       "/cosine_goodxid0")
# #
# if __name__ == '__main__':
#     flat_acqs = pydantic.parse_file_as(List[FlatFlyAcquisitions], natmixconfig.MANIFEST_FILE)
#     flat_acqs = list(filter(lambda x: x.stat_file() is not None, flat_acqs))
#     flat_acqs = list(filter(lambda x: not x.is_pair(), flat_acqs))
#
#     metric = 'cosine'
#
#     da_corr_list = []
#
#     for flat_acq in flat_acqs:
#         da_corrmat = main(flat_acq, good_xid0_only=True, drop_probe_odors=True, metric=metric)
#         da_corr_list.append(da_corrmat)
#
#     kiwi_corrs = [item for item in da_corr_list if item.panel == 'kiwi']
#     control_corrs = [item for item in da_corr_list if item.panel == 'kiwi']
#
#     da_corrs_kiwi = xr.concat(kiwi_corrs, 'fly_panel')
#     da_corrs_control = xr.concat(control_corrs, 'fly_panel')
#     da_corrs = xr.concat(da_corr_list, 'fly_panel')
#
#     SAVE_DIR = Path("/local/matrix/Remy-Data/projects/natural_mixtures/for_tom/correlation"
#                     "/cosine_goodxid0")
#
#     with open(SAVE_DIR.joinpath(f'da_corrs_kiwi_{metric}.pkl'), 'wb') as f:
#         pkl.dump(da_corrs_kiwi, f)
#
#     with open(SAVE_DIR.joinpath(f'da_corrs_control_{metric}.pkl'), 'wb') as f:
#         pkl.dump(da_corrs_control, f)
#
#     with open(SAVE_DIR.joinpath(f'da_corrs_{metric}.pkl'), 'wb') as f:
#         pkl.dump(da_corrs, f)
#
#     #
#     # save_file = Path("/local/matrix/Remy-Data/projects/natural_mixtures/for_tom/correlation/da_corrs.pkl")
#     # with open(save_file, 'wb') as f:
#     #     pkl.dump(da_corrs, f)
#     #
#     # save_file = Path("/local/matrix/Remy-Data/projects/natural_mixtures/for_tom/correlation/da_corrs.pkl")
#     #
#     # with open(save_file, 'rb') as f:
#     #     da_corrs = pkl.load(f)
# # %%
# da_mean_corr = da_corrs.groupby('panel').mean(dim='fly_panel')
# da_kiwi = da_mean_corr.sel(panel='control')
# da_control = da_kiwi = da_mean_corr.sel(panel='control')
