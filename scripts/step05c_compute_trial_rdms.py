import matplotlib

import ryeutils

matplotlib.use('agg')
import matplotlib.pyplot as plt

import copy
import xarray as xr
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
import natmixconfig

import seaborn as sns

STAT_DIR_INPUT_FILES = [
    "xrds_suite2p_respvec_mean_peak.nc",
    "xrds_suite2p_respvec_max_peak.nc",
]

STAT_DIR_OUTPUT_FILES = [
    'RDM_trials'
]

control_ord = ['2h @ -5.0',
               'fur @ -4.0',
               'ms @ -3.0',
               '1o3ol @ -3.0',
               'va @ -3',
               'cmix @ 0.0',
               'cmix @ -1.0',
               'cmix @ -2.0',
               'cmix_no_1o3ol @ 0.0',
               'cmix_no_2h @ 0.0',
               'pfo @ 0.0',
               't2h @ -6.0',
               '3mt1p @ -6.0',
               ]

stim_ord = ['pfo @ 0.0',
            '1-6ol @ -4.0', '2-but @ -5.0',
            'sulc @ -4.0',
            'limon @ -4.0',
            '2PhEtOh @ -3.0',
            'ea @ -5.0',
            'LinOx @ -3.0',
            't3h1ol @ -4.0',
            'IaOH @ -4.0',
            '6noicAcid @ -3.0',
            'hb @ -3.0']


# %%
def _rdm_over_trials(da_peak, metric='correlation'):
    """Computes representation dissimilarity matrix between trials, w/ specified distance metric.

    Args:
        da_peak (xr.DataArray): response peak amplitudes w/ dims (trials x cells)
        metric (str): distance metric, anything that scipy.distance.pdist takes

    Returns:
        xr.DataArray: RDM returned by `squareform(pdist(..., metric=metric))`, with dims =
                        ('trial_row', 'trial_col')
    """

    trials = da_peak.trials.to_numpy()
    stim = da_peak.stim.to_numpy()

    # trials = da_peak.stim.to_series().reset_index()
    # mi_trials = pd.MultiIndex.from_frame(trials)

    distmat = squareform(pdist(da_peak, metric=metric))
    da_trial_rdm = xr.DataArray(name=metric,
                                data=distmat,
                                dims=['trial_row', 'trial_col'],
                                coords=dict(
                                        trial_row=trials,
                                        trial_col=trials,
                                        stim_row=('trial_row', stim),
                                        stim_col=('trial_col', stim),
                                ),
                                )
    return da_trial_rdm


def rdm_over_trials(ds_peak, metric='correlation', copy_attrs=False):
    attrs_ = copy.deepcopy(ds_peak.attrs)

    rdms = {k: _rdm_over_trials(v, metric=metric) for k, v in ds_peak.items()}
    ds_trial_rdms = xr.Dataset(rdms)

    if copy_attrs:
        ds_trial_rdms = ds_trial_rdms.assign_attrs(attrs_)

    return ds_trial_rdms


def stim_coord_to_multiindex(ds, coord_name, prefix=None):
    """

    Args:
        ds ():
        coord_name ():
        prefix ():

    Returns:
        pd.MultiIndex

    Examples:
        >>> mi_col = step05b_interodor_corrmats.stim_coord_to_dataframe(ds_rdm, 'stim_col', prefix='col_')
        >>> mi_row = step05b_interodor_corrmats.stim_coord_to_dataframe(ds_rdm, "stim_row",
        >>>                                     prefix='row_')
        >>> ds_rdm.assign_coords(dict(stim_col=mi_col, stim_row=mi_row))
        >>>



    """
    return pd.MultiIndex.from_frame(stim_coord_to_dataframe(ds, coord_name, prefix=prefix))


# def process_ds_rdm(ds_rdm, col_vars = )

def stim_coord_to_dataframe(ds, coord_name, prefix=None, stim_ord=None):
    """Converts a coordinate along (trials,) dim into an indexed stimulus dataframe

    Examples:
        >>> ds_rdm = step05b_interodor_corrmats.rdm_over_trials(ds_max_peak)
        >>> df_col = step05b_interodor_corrmats.stim_coord_to_dataframe(ds_rdm, 'stim_col',
        prefix='col_')
        >>> df_col = stim_coord_to_dataframe(ds_rdm, 'stim_col', prefix='col_')
    """
    stim_list0 = ds.coords[coord_name].to_numpy().tolist()
    df_stim0 = pd.DataFrame(ryeutils.index_stimuli(stim_list0))

    if stim_ord is not None:
        df_stim0['stim'] = pd.Categorical(stim_list0, ordered=True, categories=stim_ord)

    if prefix is not None:
        df_stim0 = df_stim0.add_prefix(prefix)

    return df_stim0


def main(flacq,
         save_files=False,
         metrics=['correlation', 'cosine'], make_plots=False):
    mov_dir = flacq.mov_dir(relative_to=natmixconfig.NAS_PROC_DIR)
    stat_file = natmixconfig.NAS_PROC_DIR / flacq.stat_file()
    title_str = flacq.filename_base()

    respvec_types = ['mean_peak', 'max_peak']

    for respvec in respvec_types:
        input_filename = f'xrds_suite2p_respvec_{respvec}.nc'
        ds_peak = xr.load_dataset(stat_file.with_name(input_filename))

        ds_peak = ds_peak \
            .where(ds_peak.xid0.isin(ds_peak.attrs['good_xid']), drop=True) \
            .sortby(['xid0', 'embedding0'])

        stim_list = ds_peak.stim.to_numpy().tolist()

        for metric in metrics:
            ds_rdm = rdm_over_trials(ds_peak, metric=metric, copy_attrs=True)

            # tidy
            df_rdm = ds_rdm.to_dataframe()
            df_rdm = df_rdm.set_index(['stim_row', 'stim_col'], append=True)

            if save_files:

                # make output directory in suite2p/combined directory
                output_dir0 = stat_file.with_name('RDM_trials')

                # make folder in {NAS_PRJ_DIR}/analysis_outputs/RDM_trials
                output_dir1 = natmixconfig.NAS_PRJ_DIR.joinpath('analysis_outputs', 'RDM_trials',
                                                                flacq.filename_base())

                # save to both directories
                for output_dir in [output_dir0, output_dir1]:
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # save netcdf xarray
                    output_filename = f"{flacq.filename_base()}__trialRDM__{respvec}__{metric}.nc"
                    ds_rdm.to_netcdf(output_dir.joinpath(output_filename))

                    # save in long format
                    output_filename = f"{flacq.filename_base()}__trialRDM__{respvec}__{metric}__tidy.csv"
                    df_rdm.to_csv(output_dir.joinpath(output_filename))

                    # save as multisheet excel file
                    output_filename = f"{flacq.filename_base()}__trialRDM__{respvec}__{metric}.xlsx"
                    with pd.ExcelWriter(output_dir.joinpath(output_filename)) as writer:
                        mi_row = pd.MultiIndex.from_frame(
                            ds_rdm.trial_row.to_dataframe()
                            )
                        mi_col = pd.MultiIndex.from_frame(
                                ds_rdm.trial_col.to_dataframe()
                        )

                        for k, v in ds_rdm.items():
                            df = v.to_pandas()
                            df.index = mi_row
                            df.columns = mi_col
                            df.to_excel(writer, sheet_name=k)

            if make_plots:
                title_str = f"{flacq.filename_base()}\nrespvec={respvec}, metric={metric}"

                fig, axarr = plt.subplots(1, 2, figsize=(11, 8.5), constrained_layout=True)
                for ax, data_var in zip(axarr.flat, ['Fc_zscore', 'Fc_zscore_smoothed']):
                    plot_rdm(ds_rdm, data_var=data_var, ax=ax, metric=metric)
                fig.suptitle(title_str)

                # save figures
                output_filename = f"{flacq.filename_base()}__{respvec}__{metric}.png"
                fig.savefig(output_dir0.joinpath(output_filename))
                fig.savefig(output_dir0.joinpath(output_filename).with_suffix('.pdf'))

    return True


def add_stim_ord_index(ds_rdm, stim_ord):
    stim_ord_row = [stim_ord.index(item) for item in ds_rdm.stim_row.to_numpy()]
    stim_ord_col = [stim_ord.index(item) for item in ds_rdm.stim_col.to_numpy()]

    ds_rdm_with_ord = ds_rdm.assign_coords(dict(stim_ord_row=('trial_row',
                                                              stim_ord_row))
                                           )
    ds_rdm_with_ord = ds_rdm_with_ord.assign_coords(dict(stim_ord_col=('trial_col', stim_ord_row))
                                                    )
    return ds_rdm_with_ord


# def test_plot_tidy():
#     tidy_rdm = pd.read_csv(
#         '/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/2022-08-20/1/validation0'
#         '/source_extraction_s2p/suite2p/combined/RDM_trials'
#         '/xrds_suite2p_respvec_max_peak__trialRDM__correlation__tidy.csv')
#
#     stim_ord = ['pfo @ 0.0',
#                 '1-6ol @ -4.0',  '2-but @ -5.0',
#                 'sulc @ -4.0',
#                 'limon @ -4.0',
#                 '2PhEtOh @ -3.0',
#                 'ea @ -5.0',
#                 'LinOx @ -3.0',
#                 't3h1ol @ -4.0',
#                 'IaOH @ -4.0',
#                 '6noicAcid @ -3.0',
#                 'hb @ -3.0']
#
#     stim_ord_row = [stim_ord.index(item) for item in ds_rdm.stim_col.to_numpy()]
#
#     rows = list(tidy_rdm.loc[:, ['stim_row', 'trial_row']].itertuples(index=False, name=None))
#     cols = list(tidy_rdm.loc[:, ['stim_col', 'trial_col']].itertuples(index=False, name=None))
#
#     col_ord = sorted(list(set(cols)),
#                      key=lambda x: (stim_ord.index(x[0]), x[1])
#                      )
#     row_ord = sorted(list(set(rows)),
#                      key=lambda x: (stim_ord.index(x[0]), x[1])
#                      )
#
#     data = dict(x_values=cols,
#                 y_values=rows,
#                 )

def get_RDM_ordered_dataframes(ds_rdm, stim_order, sort_keys=['stim', 'stim_occ']):
    """ For an RDM dataset (repr. dissim. matrix), get sort_idx for `trial_row` and `trial_col`.

    Args:
        ds_rdm (xr.Dataset): RDM w/ dims (trial_row, trial_col) and coords `stim_row`, `stim_col`
        stim_order (list): stimulus ordering (for plotting heatmaps)
        sort_keys (list): column names to sort by in df_stim (see `ryeutils.index_stimuli`)

    Returns:
        df_stim_row (pd.DataFrame): sorted `df_stim` for row dimension of `ds_rdm`
        df_stim_col (pd.DataFrame): sorted `df_stim` for col dimension of `ds_rdm`

    Examples:
        >>> df_sort_row, df_sort_col = step05c_compute_trial_rdms.get_RDM_ordered_dataframes(ds_rdm_correlation, stim_ord)
        >>> ds_rdm_ordered = ds_rdm_correlation.isel(trial_row=df_sort_row.index.to_numpy(),\
        >>>     trial_col=df_sort_col.index.to_numpy())
        >>> ds_rdm_ordered['Fc_zscore_smoothed'].to_pandas()
    """
    df_stim_row = stim_coord_to_dataframe(ds_rdm, 'stim_row')
    df_stim_row['stim'] = pd.Categorical(df_stim_row['stim'], categories=stim_order, ordered=True)
    df_stim_row = df_stim_row.sort_values(sort_keys)

    df_stim_col = stim_coord_to_dataframe(ds_rdm, 'stim_col')
    df_stim_col['stim'] = pd.Categorical(df_stim_col['stim'], categories=stim_order, ordered=True)
    df_stim_col = df_stim_row.sort_values(sort_keys)

    return df_stim_row, df_stim_col


def sort_ds_rdm(ds_rdm, stim_order, sort_keys=['stim', 'stim_occ']):
    """Sort rows and columns of RDM according to stimulus order"""
    df_row_sorted, df_col_sorted = get_RDM_ordered_dataframes(ds_rdm, stim_order,
                                                              sort_keys=sort_keys)

    mi_row = pd.MultiIndex.from_frame(df_row_sorted)
    mi_col = pd.MultiIndex.from_frame(df_col_sorted)

    ds_rdm_sorted = ds_rdm.isel(trial_row=df_row_sorted.index.to_numpy(),
                                trial_col=df_col_sorted.index.to_numpy())
    return ds_rdm_sorted


def make_rdm_tidy(ds_rdm, dim_order=['trial_row', 'trial_col'], stim_order=None):
    tidy_rdm = ds_rdm.to_dataframe(dim_order=dim_order)

    tidy_rdm['stim_row'] = pd.Categorical(tidy_rdm['stim_row'],
                                          categories=stim_order,
                                          ordered=True)
    tidy_rdm['stim_col'] = pd.Categorical(tidy_rdm['stim_col'],
                                          categories=stim_order,
                                          ordered=True)
    # tidy_rdm = tidy_rdm.reset_index(drop=False)
    return tidy_rdm


# %%
def get_label_locs(label_list):
    labels, runs = ryeutils.find_runs(label_list)

    block_start = np.cumsum(runs) - runs[0]
    loc_in_block = (np.array(runs) - 1) / 2

    tick_locs = list(block_start + loc_in_block)
    tick_labels = labels
    return tick_labels, tick_locs


def plot_rdm(rdm, data_var=None, ax=None, rep_matrix_type='rdm', metric='correlation'):
    """

    Args:
        rdm ():
        data_var ():
        ax ():
        rep_matrix_type ():

    Returns:
        ax ():

    Examples:
        >>> ds_rdm = xr.load_dataset(
            '/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/2022-08-29/2/
            validation1_001/source_extraction_s2p/suite2p/combined/RDM_trials/
            xrds_suite2p_respvec_max_peak__trialRDM__cosine.nc')

        >>> ds_rdm_sorted = sort_ds_rdm(ds_rdm, stim_ord)
        >>> ds_rdm_sorted = sort_ds_rdm(ds_rdm, stim_ord, )
        >>> fig, axarr = plt.subplots(2, 2, constrained_layout=True)
        >>> for ax, data_var in zip(['Fc_zscore', 'Fc_zscore_smoothed', 'Fc_normed',\
        'Fc_normed_smoothed'], axarr.flat):
            plot_rdm(\
                    ds_rdm_sorted,\
                    data_var='Fc_zscore_smoothed',\
                    ax=ax, rep_matrix_type='rdm',\
                    )
        >>> plt.show()

    """
    if isinstance(rdm, xr.Dataset):
        da_rdm = rdm[data_var]
        title_str = data_var
    elif isinstance(rdm, xr.DataArray):
        da_rdm = rdm
        title_str = da_rdm.name

    if rep_matrix_type == 'rdm':
        da_plot = 1.0 - da_rdm
    elif rep_matrix_type == 'rsa':
        da_plot = da_rdm

    df_plot = pd.DataFrame(da_plot.to_numpy(),
                           index=da_plot.stim_row.to_numpy(),
                           columns=da_plot.stim_col.to_numpy())

    # sns.heatmap(df_plot,
    #              cmap='RdBu_r',
    #              vmin=-1, vmax=1,
    #              square=True,
    #              ax=ax,
    #             xticklabels='auto',
    #             yticklabels='auto',
    #             cbar_kws=dict(shrink=0.5),
    #              )

    if metric == 'correlation':
        vmin, vmax = (-1, 1)
        cmap = 'RdBu_r'
    elif metric == 'cosine':
        vmin, vmax = (-1, 1)
        cmap = 'Spectral_r'

    sns.heatmap(da_plot.to_numpy(),
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                square=True,
                ax=ax,
                cbar_kws=dict(shrink=0.25)
                )

    ylabels, ylocs = get_label_locs(da_plot.stim_row.to_numpy())
    xlabels, xlocs = get_label_locs(da_plot.stim_col.to_numpy())

    ax.set_xticks(xlocs)
    ax.set_xticklabels(xlabels, rotation=90)

    ax.set_yticks(ylocs)
    ax.set_yticklabels(ylabels, rotation=0)

    ax.set_title(title_str)
    return ax


def test_method():
    stim_ord = ['pfo @ 0.0',
                '1-6ol @ -4.0', '2-but @ -5.0',
                'sulc @ -4.0',
                'limon @ -4.0',
                '2PhEtOh @ -3.0',
                'ea @ -5.0',
                'LinOx @ -3.0',
                't3h1ol @ -4.0',
                'IaOH @ -4.0',
                '6noicAcid @ -3.0',
                'hb @ -3.0']

    ds_rdm = xr.load_dataset(
            '/local/matrix/Remy-Data/projects/odor_space_collab/processed_data/2022-08-29/2/validation1_001/source_extraction_s2p/suite2p/combined/RDM_trials/xrds_suite2p_respvec_max_peak__trialRDM__correlation.nc')

    ds_rdm_ord = add_stim_ord_index(ds_rdm, stim_ord)

    ds_plot = ds_rdm_ord.sortby(['stim_ord_row', 'trial_row']) \
        .sortby(['stim_ord_col', 'trial_col']) \
        .reset_index('trial_row').reset_index('trial_col')

    rsm = 1 - ds_plot['Fc_zscore'].to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

    sns.heatmap(data=rsm,
                xticklabels=ds_plot.stim_row.to_numpy(),
                yticklabels=ds_plot.stim_col.to_numpy(),
                cbar=True,
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                cbar_kws=dict(shrink=0.25
                              ),
                square=True,
                ax=ax
                )
    plt.show()

    stim_list = ds_rdm.stim
    stim_list_cat = pd.Categorical(stim_list, ordered=True, categories=stim_ord)
    df_stim = pd.DataFrame(ryeutils.index_stimuli(stim_list))

    # make stim_list categorical
    df_stim['stim'] = pd.Categorical(df_stim['stim'],
                                     ordered=True, categories=stim_ord)

    da_rdm = ds_rdm['Fc_zscore_smoothed']

    tidy_rdm = da_rdm.to_dataframe().set_index(['stim_row', 'stim_col'], )

    df_rdm = da_rdm.to_pandas()
    ds_rdm['stim_row'] = stim_list_cat
    ds_rdm['stim_col'] = stim_list_cat
    # df_rdm.index = stim_list_cat
    # df_rdm.columns = stim_list_cat

    df_plot = 1 - df_rdm
    df_plot = df_plot.sort_index()
    df_plot = df_plot.loc[:, df_plot.index.tolist()]

    fig, ax = plt.subplots(1, 1, figsize=(11, 11))
    sns.heatmap(1 - ds_rdm_ord['Fc_zscore_smoothed'].to_numpy(),
                xticklabels=ds_rdm_ord.stim_col.to_numpy(),
                yticklabels=ds_rdm_ord.stim_row.to_numpy(),
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                ax=ax,
                square=True,
                cbar_kws=dict(
                        orientation='vertical',
                        shrink=0.25,
                        label='Fc_zscore_smoothed'),
                )
    # fig.suptitle(title_str)
    plt.show()
    return True

