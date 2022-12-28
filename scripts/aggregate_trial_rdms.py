"""
Code for aggregating trial-level RDMs into a single RDM.

Steps:
-----
1. Load FlatFlyAcquisitions from manifest file
2. Load RDMs from each acquisition
3. Define a stimulus ordering for combining the RDMs
4. Order the RDMs according to the stimulus ordering
5. Aggregate the RDMs across acquisitions
6. Save the aggregated RDMs to a file
7. Plot the aggregated RDMs
8. Save the plots to a PDF file


RDM trial dataset:
------------------
>>> ds_rdm
<xarray.Dataset>
Dimensions:             (trial_row: 36, trial_col: 36)
Coordinates:
  * trial_row           (trial_row) int64 0 1 2 3 4 5 6 ... 29 30 31 32 33 34 35
  * trial_col           (trial_col) int64 0 1 2 3 4 5 6 ... 29 30 31 32 33 34 35
    stim_row            (trial_row) object 'pfo @ 0.0' ... '1-6ol @ -4.0'
    stim_col            (trial_col) object 'pfo @ 0.0' ... '1-6ol @ -4.0'
Data variables:
    Fc                  (trial_row, trial_col) float64 0.0 1.107 ... 0.4622 0.0
    F                   (trial_row, trial_col) float64 0.0 1.097 ... 0.4575 0.0
    Fneu                (trial_row, trial_col) float64 0.0 0.6022 ... 0.246 0.0
    spks                (trial_row, trial_col) float64 0.0 1.057 ... 0.5493 0.0
    Fc_smooth           (trial_row, trial_col) float64 0.0 1.107 ... 0.4619 0.0
    Fc_zscore           (trial_row, trial_col) float64 0.0 1.123 ... 0.5935 0.0
    Fc_zscore_smoothed  (trial_row, trial_col) float64 0.0 1.124 ... 0.5934 0.0
    Fc_normed           (trial_row, trial_col) float64 0.0 1.122 ... 0.5928 0.0
    Fc_normed_smoothed  (trial_row, trial_col) float64 0.0 1.123 ... 0.5927 0.0
Attributes: (12/13)
    ...


Stimulus ordering:
------------------
Uses ryeutils.index_stimuli(stim) function, which contains the keys ['stim', 'stim_occ', 'run_idx',
'idx_in_run', 'run_occ'].

For example, ryeutils.index_stimuli(x) where x = 'AAABBBCCCAAABBB' would result in the following:
    -       stim = [ A, A, A, B, B, B, C, C, C, A, A, A, B, B, B, C, C, C]
    -   stim_occ = [ 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5]
    -    run_idx = [ 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    - idx_in_run = [ 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    -    run_occ = [ 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]

Alternatively, for easier viewing:
    -       stim = AAA BBB CCC AAA BBB
    -   stim_occ = 012 012 012 345 345
    -    run_idx = 000 111 222 333 444
    - idx_in_run = 012 012 012 012 012
    -    run_occ = 000 111 222 333 444

To maintain ordering by different type of movie, use the listed keys returned by stimulus_index:
    - For consecutive odor repeats with repeated blocks (i.e. kiwi_components_again), use:
        - stim
        - run_occ
        - idx_in_run

    - For movies with shuffled stimuli (non-consecutive repeats), use:
        - 'stim'
        - 'stim_occ'


MultiIndices in concatenated RDMs:
---------------------------------
In order to avoid complications with multiindex dimensions when saving xarray datasets to file,
we convert reset indices ['trial_row, 'trial_col'] to regular coordinates, with the prefix 'row_' or 'col_'.

Steps
=====
- filter list of FlatFlyAcquisitions
- get list of dataset files, loaded from flat_acqs
- combine datasets with `xr.concat`


Functions:
    - generate_list_of_rdm_net_cdf_files(flat_acqs, respvec, metric, trialavg)
    - combine_rdms(ds_rdm_list, index_stimuli_list)
    - reorder_rdm_concat(): function for setting the stimulus ordering
    - make_ds_rdm_list_from_flat_acqs()
        - take list of FlatFlyAcquisitions
        - `index_stimuli_keys`: list of keys, used to order/group trials by stimulus

Outputs:
------

RDM_OUTPUT_DIR = ../analysis_outputs/by_imaging_type/kc_soma/panel/RDM_concat
 - RDM_cat__

The directory in which the RDMs are saved is specified by the `rdm_dir` variable.

The output directory is sd
/local/matrix/Remy-Data/projects/odor_space_collab/analysis_outputs/by_imaging_type/kc_soma
/local/matrix/Remy-Data/projects/odor_space_collab/analysis_outputs/by_imaging_type/kc_soma
Concatenated RDMs are saved to netcdf files, with the following naming convention:
    - <respvec>_<metric>_<trialavg>_rdms.nc

For example:
    - Fc_zscore_smoothed_trialavg_rdm.nc
options:
- imaging_type
- panel
-



how to group concatenated RDMs?
...{OUTPUT_DIR}/{imaging_type}/{panel}_{respvec}_{metric}.nc
...{OUTPUT_DIR}/{imaging_type}/{panel}_{respvec}_{metric}_trialavg.nc

Save plots separately, in {NAS_PRJ_DIR}/intermediate_plots
- RDM_trials_concat
    -

Generate list of possible flat_acq lists, iterate through, and save to file



Parameters for aggregating trial rdms:
-------------------------------------
    rdm_params:
      - metric: 'cosine' or 'correlation'
      - respvec: 'max_peak' or 'mean_peak'

    panel_params:
      - imaging_type: 'kc_soma', 'pn_boutons'
      - panel: 'megamat'


"""

import copy
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid

import natmixconfig
import ryeutils
from aggregate import PanelMovies, filter_flacq_list
from pydantic_models import FlatFlyAcquisitions
from s2p import suite2p_helpers
from scripts import step05c_compute_trial_rdms

plt.rcParams.update({'pdf.fonttype': 42,
                     'text.usetex': False})
# Load RDMs from each acquisition, that were generated by scripts/step05c_compute_trial_rdms.py.
# Note that the RDMs are xarray Datasets, not DataArrays.
# The RDM files are saved in the folder named RDM_trials, in the suite2p output folder.
# The RDMs are saved in netcdf files named \
# xrds_suite2p_respvec__trialRDM__{respvec}__{metric}.h5, \
# where {respvec} is the name of the response vector type ('mean_peak' or 'max_peak'), \
# and {metric} is the metric used to compute the RDM ('cosine' or 'correlation').
# The RDM netcdf files are located in the folder named RDM_trials, in the suite2p output folder.

ufunc_kwargs_lookup = {
    'robust_scale': dict(with_centering=True,
                         with_scaling=True,
                         quantile_range=(2.5, 97.5)
                         ),
    'quantile_transform': dict(axis=0, n_quantiles=100,
                               output_distribution='normal'),
    'scale': None
}

# %%
flacq_split_params = dict(imaging_type=None,
                          panel=None)

rdm_params = dict(respvec=None,
                  metric=None,
                  index_stimuli_keys=None,
                  trialavg=None,
                  )

respvec_params = dict(respvec='mean_peak',
                      trialavg=True,
                      good_xids_only=False,
                      standardize=True,
                      ufunc=None,
                      ufunc_kwargs=None
                      )

RDM_OUTPUT_DIR = "analysis_outputs/by_imaging_type/{imaging_type}/{panel}/RDM_concat"

RESPVEC_OUTPUT_DIR = "analysis_outputs/by_imaging_type/{imaging_type}/{panel}/respvec_concat"
# %%

# panel_param_grid:
# ---------------
# stores all combinations of options for `imaging_type` and `panel` (i.e. which
# FlatFlyAcquisitions to process/aggregate)
#
# rdm_param:
# ---------
prj = natmixconfig.prj

if prj == 'odor_space_collab':
    panel_param_grid = dict(
            imaging_type=['kc_soma', 'pn_boutons'],
            panel='megamat'
    )

    for panel_param in ParameterGrid(panel_param_grid):
        print(f'\npanel_param:\n------------')
        print(panel_param)

        rdm_param_grid = dict(
                respvec=['mean_peak', 'max_peak'],
                metric=['correlation', 'cosine']
        )
        print('\nrdm_param:')
        for rdm_param in ParameterGrid(rdm_param_grid):
            print(f"  - {rdm_param.__str__()}")
#%%
def get_respvec_param_combinations():
    """Get a list of all possible parameter combinations for aggregating respvec datasets."""


    # standardization_ufunc = [(getattr(preprocessing, k), v)
    #                          for k, v in ufunc_kwargs_lookup.items()]

    respvec_param_grid_options = dict(
            respvec=['mean_peak', 'max_peak'],
            trialavg=[True, False],
            good_xids_only=[True, False],
            ufunc_name=[None, 'robust_scale', 'quantile_transform', 'scale'],
    )
    respvec_param_grid = list(ParameterGrid(respvec_param_grid_options))

    for params in respvec_param_grid:
        params['standardize'] = params['ufunc_name'] is not None

    with_ufuncs = True
    if with_ufuncs:
        for params in respvec_param_grid:
            if params['ufunc_name'] is None:
                params['standardize'] = False
                params['ufunc'] = None
                params['ufunc_kwargs'] = None
            else:
                params['standardize'] = True
                params['ufunc']=getattr(preprocessing, params['ufunc_name'])
                params['ufunc_kwargs'] = ufunc_kwargs_lookup
    return respvec_param_grid
# %%

def load_rdms_by_flacq(flat_acq, respvec, metric, trialavg=False):
    """Load RDMs from a FlatFlyAcquisition.

    Args:
        flat_acq (FlatFlyAcquisition): the FlatFlyAcquisition to load RDMs from
        respvec (str): the response vector type ('mean_peak' or 'max_peak')
        metric (str): the metric used to compute the RDM ('cosine' or 'correlation')
        trialavg (bool): whether to load the trial-averaged RDMs
    Returns:
        rdm_ds (xarray.Dataset): the RDM Dataset
    """
    if trialavg:
        filename = f'xrds_suite2p_respvec__trialavgRDM__{respvec}__{metric}.nc'
    else:
        filename = f'xrds_suite2p_respvec__trialRDM__{respvec}__{metric}.nc'
    # rdm_ds = xr.load_dataset(natmixconfig.NAS_PROC_DIR / flat_acq.stat_file().parent /
    #                          'RDM_trials' /
    #                          )
    ds_file = natmixconfig.NAS_PROC_DIR / flat_acq.stat_file().with_name('RDM_trials').joinpath(
            filename)
    rdm_ds = xr.load_dataset(ds_file)
    # ds_respvec = rdm_ds0.assign_coords(
    #         dict(iacq=iacq,
    #              date_imaged=flat_acq.date_imaged,
    #              fly_num=flat_acq.fly_num,
    #              thorimage=flat_acq.thorimage,
    #              abbrev=('trials',
    #                      [item.split('@')[0].strip() for item in ds_respvec0.stim.to_numpy()]
    #                      ),
    #              )
    # )
    return rdm_ds


def load_rdms_by_flacq_list(flat_acq_list, respvec, metric, trialavg=False):
    """Load RDMs from a list of FlatFlyAcquisitions.

    Args:
        trialavg (bool): whether or
        flat_acq_list (List[FlatFlyAcquisition]): the FlatFlyAcquisitions to load RDMs from
        respvec (str): the response vector type ('mean_peak' or 'max_peak')
        metric (str): the metric used to compute the RDM ('cosine' or 'correlation')

    Returns:
        rdm_ds_list (xarray.Dataset): the RDM Datasets
    """
    rdm_ds_list = []
    for flat_acq in flat_acq_list:
        rdm_ds = load_rdms_by_flacq(flat_acq, respvec, metric, trialavg=trialavg)
        rdm_ds_list.append(rdm_ds)
    # rdm_ds = xr.concat(rdm_ds_list, dim='trial')
    return rdm_ds_list


# %%
def load_respvecs_by_flacq(flat_acq, respvec, iacq=None):
    """Load RDMs from a FlatFlyAcquisition.

       Args:
           iacq ():
           flat_acq (FlatFlyAcquisition): the FlatFlyAcquisition to load RDMs from
           respvec (str): the response vector type ('mean_peak' or 'max_peak')

       Returns:
           ds_respvec (xarray.Dataset): the respvec Dataset w/ flat acq metadata added to
           coordinates
       """
    filename = flat_acq.stat_file(relative_to=natmixconfig.NAS_PROC_DIR) \
        .with_name(f"xrds_suite2p_respvec_{respvec}.nc")
    ds_respvec0 = xr.load_dataset(filename)

    # select which cells are in good_xids, and add as coordinate to dataset
    iscell_good_xid0 = ds_respvec0['xid0'].isin(ds_respvec0.attrs['good_xid']).to_numpy()

    # add flacq metadata to dataset (for concatenating later)
    ds_respvec = ds_respvec0.assign_coords(
            dict(iacq=iacq,
                 date_imaged=flat_acq.date_imaged,
                 fly_num=flat_acq.fly_num,
                 thorimage=flat_acq.thorimage,
                 abbrev=('trials',
                         [item.split('@')[0].strip() for item in ds_respvec0.stim.to_numpy()]
                         ),
                 iscell_good_xid0=('cells', iscell_good_xid0),
                 )
    )
    return ds_respvec


def load_respvecs_by_flacq_list(flat_acq_list, respvec):
    """Load respvecs from a list of FlatFlyAcquisitions, without combining.

    Args:
        flat_acq_list (List[FlatFlyAcquisition]): the FlatFlyAcquisitions to load RDMs from
        respvec (str): the response vector type ('mean_peak' or 'max_peak')

    Returns:
        rdm_ds_list (List[xr.Dataset]): List of respvec Datasets, not contatenated.

    Note: the reason this isn't concatenated within the method is to allow for more flexibility
    in options before combining them
    """

    ds_respvec_list = []

    for iacq, flat_acq in enumerate(flat_acq_list):
        ds_respvec0 = load_respvecs_by_flacq(flat_acq, respvec, iacq=iacq)
        ds_respvec_list.append(ds_respvec0)
    return ds_respvec_list


# %%
def process_ds_respvec_list(ds_respvec_list,
                            trialavg=True,
                            good_xids_only=False,
                            standardize=True,
                            ufunc=None,
                            ufunc_kwargs=None, **kwargs):
    """

    Process a list of respvec datasets, optionally averaging over trials, dropping bad cells,
    and performing feature standardization.

    Args:
        ds_respvec_list (List[xr.Dataset]): List of respvec Datasets, not contatenated.
        trialavg (bool): whether or not to average over trials w/ the same stimulus (default: True)
        good_xids_only (bool): whether or not to drop cells that are not in good_xids
        standardize (bool): whether or not to perform feature standardization, using `ufunc` if True
        ufunc (function): the function to use for standardization ((default: zscore))
        ufunc_kwargs ():

    Returns:
        ds_processed_respvec_list (List[xr.Dataset]): List of processed respvec Datasets

    Examples:
        >>> flat_acqs = [flat_acqs[i] for i in [0, 1, 2, 4]]
        >>> ds_respvec_list =  load_respvecs_by_flacq_list(flat_acqs, 'mean_peak')
        >>> ds_processed_respvec_list = process_ds_respvec_list(ds_respvec_list,)
        >>> ds_respvec_cat = xr.concat(ds_processed_respvec_list, 'cells')
    """
    ds_processed_respvec_list = []

    for ds_respvec0 in ds_respvec_list:
        # trialavg step
        # ----------------
        if trialavg:
            ds = ds_respvec0.groupby('stim').mean('trials')
            core_dims = [['cells', 'stim']]
        else:
            ds = ds_respvec0.sortby('stim').reset_index('trials')
            core_dims = [['cells', 'trials']]

        # filter good_xids
        # ------------------
        if good_xids_only:
            # ds_good = ds.where(ds.xid0.isin(ds.attrs['good_xid']), drop=True)
            ds_good = ds.where(ds.iscell_good_xid0, drop=True)
        else:
            ds_good = ds

        # standardize, if standardize=True
        # -------------------------------------
        if standardize:
            ds_standardized = xr.apply_ufunc(
                    ufunc,
                    ds_good,
                    input_core_dims=core_dims,
                    output_core_dims=core_dims,
                    # vectorize=True,
                    keep_attrs=True,
                    kwargs=ufunc_kwargs
            )
        else:
            ds_standardized = ds_good

        ds_processed_respvec_list.append(ds_standardized)

    return ds_processed_respvec_list


def combine_respvecs(ds_processed_respvec_list):
    """Combine a list of processed respvec datasets into a single dataset.

    Args:
        ds_processed_respvec_list (List[xr.Dataset]): List of processed respvec Datasets

    Returns:
        ds_respvec_cat (xarray.Dataset): the concatenated respvec Dataset
    """
    ds_respvec_cat = xr.concat(ds_processed_respvec_list, 'cells')
    return ds_respvec_cat

def aggregate_respvecs(flat_acq_py)
# %%


def combine_rdms(rdm_list):
    """Combine a list of RDMs into a single RDM.

    Args:
        rdm_list (List[xarray.Dataset]): the RDMs to combine

    Returns:
        rdm_ds (xarray.Dataset): the combined RDM
    """
    rdm_ds = xr.concat(rdm_list, dim='trial')
    return rdm_ds


def get_stim_multiindices(stim_list, index_stimuli_keys=['stim', 'run_occ', 'idx_in_run']):
    """"""
    stim_info = ryeutils.index_stimuli(stim_list)
    df_stim = pd.DataFrame(stim_info).loc[:, index_stimuli_keys]

    mi_row = pd.MultiIndex.from_frame(df_stim.add_prefix('row_'))
    mi_col = pd.MultiIndex.from_frame(df_stim.add_prefix('col_'))

    return mi_row, mi_col


def reassign_row_and_col_index(ds, row_index, col_index):
    """Reassign the row and column index of a Dataset.

    Args:
        ds (xarray.Dataset): the Dataset to reassign the index of
        row_index (pandas.MultiIndex): the new row index
        col_index (pandas.MultiIndex): the new column index

    Returns:
        ds (xarray.Dataset): the Dataset with the new row and column index
    """
    ds_mi = ds.assign_coords(trial_row=('trial_row', row_index))
    ds_mi = ds_mi.assign_coords(trial_col=('trial_col', col_index))
    return ds_mi


# %%

def drop_stim(ds, stim_to_drop):
    attrs = copy.deepcopy(ds.attrs)
    ds_trimmed = ds.where(~ds.stim_row.isin(stim_to_drop), drop=True)
    ds_trimmed = ds_trimmed.where(~ds_trimmed.stim_col.isin(stim_to_drop), drop=True)
    ds_trimmed = ds_trimmed.assign_attrs(attrs)
    return ds_trimmed


# %%
def prepare_dataset_for_concat(ds, index_stimuli_keys=['stim', 'run_occ', 'idx_in_run'],
                               stim_to_drop=None):
    """Prepare a Dataset for concatenation.

    Args:
        stim_to_drop ():
        ds (xarray.Dataset): the Dataset to prepare
        index_stimuli_keys (List[str]): the keys to use to index the stimuli

    Returns:
        ds (xarray.Dataset): the prepared Dataset
    """
    stim_list = ds.stim_row.to_numpy()
    row, col = get_stim_multiindices(stim_list, index_stimuli_keys=index_stimuli_keys)
    ds_prepared = reassign_row_and_col_index(ds, row, col)
    return ds_prepared


def prepare_ds_concat_to_save(ds_concat):
    return ds_concat.reset_index(['trial_row', 'trial_col'])


def make_ds_concat_multiindex(ds_concat, index_stimuli_keys=['stim', 'run_occ', 'idx_in_run']):
    """After loading ds_concat, convert coords to multiindex.

    This avoids issues with having to save the concatenated RDMs as
    """
    row_coords = [f'row_{k}' for k in index_stimuli_keys]
    col_coords = [f'col_{k}' for k in index_stimuli_keys]
    return ds_concat.set_index(trial_row=row_coords,
                               trial_col=col_coords)


# %%
def plot_concat_rdm_heatmaps(rdm_concat, stim_idx_ord=None, respvec=None, metric=None,
                             trialavg=None):
    stim_row = rdm_concat.trial_row['row_stim'].to_numpy()
    stim_col = rdm_concat.trial_col['col_stim'].to_numpy()

    fig_list = []

    for flacq_idx in range(rdm_concat.dims['flacq']):
        ds_rdm = rdm_concat.isel(flacq=flacq_idx)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
        ax.set_facecolor('0.85')

        ax = step05c_compute_trial_rdms.plot_rdm(ds_rdm, data_var='Fc_zscore', ax=ax,
                                                 stim_row=stim_row, stim_col=stim_col,
                                                 )
        ax.set_title(f'Fc_zscore: {respvec} {metric}, trialavg={trialavg}')

        basename = str(ds_rdm.filename_base.to_numpy())
        fig.suptitle(f"#{flacq_idx}\n{basename}")
        plt.show()
        fig_list.append(fig)
    return fig_list


# %%

# def aggregate_rdms(flat_acq_list, respvec, metric, trialavg, drop_stim=None, stim_idx_keys, ):
#     load_rdms_by_flacq_list(flacqs,
#                             respvec=opts['respvec'],
#                             metric=opts['metric'],
#                             trialavg=opts['trialavg'])
# %%

if __name__ == '__main__':
    # get list of allowed movies, depending on panel
    # %%
    prj = 'natural_mixtures'

    # load project manifest
    flat_acqs = pydantic.parse_file_as(List[FlatFlyAcquisitions], natmixconfig.MANIFEST_FILE)
    # %%
    if prj == 'odor_space_collab':
        imaging_type = 'kc_soma'
        # imaging_type = 'pn_boutons'
        megamat_panel = PanelMovies(prj=prj, panel='megamat')
        # validation_panel = PanelMovies(prj=prj, panel='validation')
        flat_acqs = filter_flacq_list(flat_acqs,
                                      allowed_imaging_type=imaging_type,
                                      allowed_movie_types=megamat_panel.movies)

        flat_acqs = filter(lambda x: suite2p_helpers.is_3d(x.stat_file(
                relative_to=natmixconfig.NAS_PROC_DIR)),
                           flat_acqs, )

        stim_idx_keys = ['stim', 'stim_occ']

        OUTPUT_DIR = natmixconfig.NAS_PRJ_DIR.joinpath('analysis_outputs', 'by_imaging_type',
                                                       imaging_type)
        OUTPUT_DIR.mkdir(exist_ok=True)
    if prj == 'natural_mixtures':
        imaging_type = 'kc_soma'
        kiwi_panel = PanelMovies(prj='natural_mixtures', panel='kiwi')
        control_panel = PanelMovies(prj='natural_mixtures', panel='control')

        # filter the flat_acq_list to only include the movies in desired panels

        flat_acqs = filter_flacq_list(flat_acqs,
                                      has_s2p_output=True,
                                      allowed_movie_types=kiwi_panel.movies + control_panel.movies)
        stim_idx_keys = ['stim', 'run_occ', 'idx_in_run']
        OUTPUT_DIR = natmixconfig.NAS_PRJ_DIR.joinpath('analysis_outputs', 'RDM_trials')
    # %%  load and concatenate RDMs
    opts = dict(respvec='mean_peak',
                metric='correlation',
                trialavg=True,
                remove_bad_stim=True)

    # split flat_acqs by odor panel, using the flacq.panel() method
    flat_acqs_by_panel = {}
    for flacq in flat_acqs:
        panel = flacq.panel()
        if panel not in flat_acqs_by_panel:
            flat_acqs_by_panel[panel] = []
        flat_acqs_by_panel[panel].append(flacq)

    # load RDMs for each panel, using the load_rdms_by_flacq_list function
    rdms_by_panel = {}
    for panel, flacqs in flat_acqs_by_panel.items():
        rdms_by_panel[panel] = load_rdms_by_flacq_list(flacqs,
                                                       respvec=opts['respvec'],
                                                       metric=opts['metric'],
                                                       trialavg=opts['trialavg'])

    # filter out unwated stimuli
    # ----------------------------
    if opts['remove_bad_stim']:
        for panel, rdms in rdms_by_panel.items():
            if panel == 'validation':
                bad_stim = ['1-6ol @ -4.0', '2-but @ -5.0', 'pfo @ 0.0']
                rdm_list_filtered = [drop_stim(rdm, bad_stim) for rdm in rdms]
                rdms_by_panel[panel] = rdm_list_filtered

    # prepare the RDMs for concatenation
    prepared_rdms_by_panel = {}
    for panel, rdms in rdms_by_panel.items():
        prepared_rdms_by_panel[panel] = [
            prepare_dataset_for_concat(rdm,
                                       index_stimuli_keys=stim_idx_keys) for rdm in rdms]

    # concatenate rdms
    # ------------------
    rdm_concat_by_panel = {k: xr.concat(v, 'flacq', combine_attrs='drop')
                           for k, v in prepared_rdms_by_panel.items()}

    rdm_concat_mean_by_panel = {}
    rdm_concat_std_by_panel = {}
    summary_figs_by_panel = {}

    # add flacq info
    for panel, rdm_concat in rdm_concat_by_panel.items():
        df_flat_acq = pd.DataFrame([item.dict() for item in flat_acqs_by_panel[panel]])
        # rdm_concat = rdm_concat.assign_coords(flacq=[item.filename_base() for item in
        #                                              flat_acqs_by_panel[panel]])
        rdm_concat = rdm_concat.assign_coords(flacq=np.arange(df_flat_acq.shape[0]))

        rdm_concat = rdm_concat.assign_coords(
                date_imaged=('flacq', df_flat_acq['date_imaged']),
                fly_num=('flacq', df_flat_acq['fly_num']),
                thorimage=('flacq', df_flat_acq['thorimage']),
                filename_base=('flacq', [item.filename_base() for item in
                                         flat_acqs_by_panel[panel]])
        )
        rdm_concat_by_panel[panel] = rdm_concat

        fly_id = pd.MultiIndex.from_frame(df_flat_acq.loc[:, ['date_imaged', 'fly_num']])
        grouper = xr.DataArray(fly_id, dims=['flacq'])

        rdm_concat_flymean = rdm_concat.groupby(grouper).mean('flacq')
        rdm_concat_mean = rdm_concat_flymean.mean('group')
        rdm_concat_std = rdm_concat_flymean.std('group')

        rdm_concat_mean_by_panel[panel] = rdm_concat_mean
        rdm_concat_std_by_panel[panel] = rdm_concat_std

        ########################
        # plot summary figure
        ########################
        fig_hmean, axarr = plt.subplots(1, 2, figsize=(11, 6.0), constrained_layout=True)
        step05c_compute_trial_rdms.plot_rdm(rdm_concat_mean.reset_index(['trial_row', 'trial_col']),
                                            'Fc_zscore',
                                            ax=axarr[0],
                                            row_coord_key='row_stim',
                                            col_coord_key='col_stim'
                                            )
        step05c_compute_trial_rdms.plot_rdm(1 - rdm_concat_std.reset_index(['trial_row',
                                                                            'trial_col']),
                                            'Fc_zscore',
                                            ax=axarr[1],
                                            row_coord_key='row_stim',
                                            col_coord_key='col_stim'
                                            )
        # sns.heatmap(rdm_concat_std['Fc_zscore'].to_numpy(),
        #             ax=axarr[1],
        #             xticklabels=rdm_concat_std.row_stim.to_numpy(),
        #             yticklabels=rdm_concat_std.col_stim.to_numpy(),
        #             square=True,
        #             cmap='RdBu_r',
        #             vmin=-1, vmax=1,
        #             cbar_kws=dict(shrink=0.5)
        #             )
        axarr[0].set_title(f"Fc_zscore: mean")
        axarr[1].set_title(f"Fc_zscore: std")

        title_str = f"Panel={panel}, pooled mean (lhs) and std (rhs)" \
                    + f"\n{opts['respvec']}, metric={opts['metric']}, trialavg" \
                      f"={opts['trialavg']}"
        fig_hmean.suptitle(title_str)
        plt.show()

        # add plot to dict
        summary_figs_by_panel[panel] = fig_hmean

    #
    # import importlib as imp
    # import seaborn as sns
    # imp.reload(step05c_compute_trial_rdms)
    #
    # for panel in rdm_concat_mean_by_panel.keys():
    #     df_mean = (rdm_concat_mean_by_panel[panel]
    #                .to_dataframe()
    #                # .drop(labels=['stim_row', 'stim_col'], axis=1)
    #                # .loc[:, ['Fc_normed']]
    #                .add_prefix('mean_')
    #                )
    #
    #     df_std = (
    #         rdm_concat_std_by_panel[panel]
    #         .to_dataframe()
    #         # .loc[:, ['Fc_normed']]
    #         # .drop(labels=['stim_row', 'stim_col'], axis=1)
    #         .add_prefix('std_')
    #     )
    #     df_rdm_panel_summary = pd.concat([df_mean, df_std], axis=1)
    #     df_plot = df_rdm_panel_summary.reset_index()
    #     df_plot['inv_std_Fc_normed'] = 1 / df_plot['std_Fc_normed']
    #
    #     g = sns.relplot(data=df_plot, x='col_stim', y='row_stim',
    #                     hue='mean_Fc_normed',
    #                     size='inv_std_Fc_normed',
    #                     palette='RdBu_r',
    #                     hue_norm=(-1, 1)
    #                     )
    #     plt.show()

    ###########################
    # save concatenated rdms
    ###########################
    for panel, rdm_concat in rdm_concat_by_panel.items():
        filename_stem = f"rdm_concat__{imaging_type}__{panel}__{opts['respvec']}_" \
                        f"_{opts['metric']}"
        if opts['trialavg']:
            filename_stem = filename_stem + '__trialavg'

        # save concatenated rdms
        save_file = OUTPUT_DIR.joinpath(filename_stem).with_suffix('.nc')
        print(f'saving to: {OUTPUT_DIR}')
        rdm_save = prepare_ds_concat_to_save(rdm_concat)
        rdm_save.to_netcdf(save_file)
        print(f"\t- {panel}: {save_file.name} saved successfully.")

        # save pooled average across flies
        save_file = OUTPUT_DIR.joinpath(filename_stem + '__pooledmean').with_suffix('.nc')
        rdm_mean_save = prepare_ds_concat_to_save(rdm_concat_mean_by_panel[panel])
        rdm_mean_save.to_netcdf(save_file)
        print(f"\t- {panel}: {save_file.name} saved successfully.")

        # save pooled std across flies
        save_file = OUTPUT_DIR.joinpath(filename_stem + '__pooledstd').with_suffix('.nc')
        rdm_std_save = prepare_ds_concat_to_save(rdm_concat_std_by_panel[panel])
        print(f"\t- {panel}: {save_file.name} saved successfully.")
        rdm_std_save.to_netcdf(save_file)

        # save pooled figures
        save_file = OUTPUT_DIR.joinpath(filename_stem + '__pooledplot').with_suffix('.pdf')
        pdf = PdfPages(save_file)
        pdf.savefig(summary_figs_by_panel[panel])
        pdf.close()
        print(f"\t- {panel}: {save_file.name} saved successfully.")
        summary_figs_by_panel[panel].savefig(save_file.with_suffix('.png'))

    # %% get lists of figures per panel

    hmaps_by_panel = {panel: plot_concat_rdm_heatmaps(rdm_concat_by_panel[panel],
                                                      respvec=opts['respvec'],
                                                      metric=opts['metric'],
                                                      trialavg=opts['trialavg']
                                                      )
                      for panel, rdm_concat in rdm_concat_by_panel.items()
                      }
    # %% save to multipage pdf file
    for panel, figs in hmaps_by_panel.items():
        # create filename
        filename_stem = f"rdm_concat__{imaging_type}__{panel}__" \
                        f"{opts['respvec']}__{opts['metric']}"
        if opts['trialavg']:
            filename_stem = filename_stem + '__trialavg'
        pdf_file = OUTPUT_DIR.joinpath(filename_stem).with_suffix('.pdf')

        # write to pdf
        pdf = PdfPages(pdf_file)
        for hmap in figs:
            pdf.savefig(hmap)
        pdf.close()
        print(f"{pdf_file} saved.")

    # %% save to folder of pngs
    # loop through individual heatmaps, and save as pngs
    for panel, figs in hmaps_by_panel.items():
        # create filename
        filename_stem = f"rdm_concat__{imaging_type}__{panel}__" \
                        f"{opts['respvec']}__{opts['metric']}"
        if opts['trialavg']:
            filename_stem = filename_stem + '__trialavg'

        png_folder = filename_stem + '__pngs'
        png_folder = OUTPUT_DIR.joinpath(png_folder)
        png_folder.mkdir(exist_ok=True)

        for fidx, hmap in enumerate(figs):
            # pdf_file = OUTPUT_DIR.joinpath(filename_stem).with_suffix('.pdf')
            png_file = filename_stem + f"__{fidx:02d}.png"
            hmap.savefig(png_folder.joinpath(png_file))
        print(f"pngs saved to:\n{png_folder}")

    # %% plot clustermap of heatmap

    # for panel, rdm_mean in rdm_concat_mean_by_panel.items():
    #     rdm_mean = rdm_concat_mean_by_panel[panel]
    #     rdm_mean = (
    #         rdm_mean
    #         .reset_index(['trial_row', 'trial_col'])
    #         .dropna('trial_row')
    #         .dropna('trial_col')
    #     )
    #
    #     filename_stem = f"rdm_concat__{imaging_type}__{panel}__" \
    #                     f"{opts['respvec']}__{opts['metric']}"
    #
    #     if opts['trialavg']:
    #         filename_stem = filename_stem + '__trialavg'
    #
    #     df_rdm = pd.DataFrame(rdm_mean['Fc_zscore'].to_numpy(),
    #                           index=rdm_mean['row_stim'].to_numpy(),
    #                           columns=rdm_mean['col_stim'].to_numpy(),
    #                           )
    #
    #     gmap = sns.clustermap(df_rdm,
    #                           cmap='RdBu_r',
    #                           vmin=-1, vmax=1,
    #                           # row_colors=network_colors,
    #                           # col_colors=network_colors,
    #                           dendrogram_ratio=(.2, .2),
    #                           cbar_pos=(.02, .32, .03, .2),
    #                           linewidths=.75, figsize=(8, 8))
    #     title_str = f"Panel={panel}" \
    #                 + f"\n{opts['respvec']}, metric={opts['metric']}, trialavg" \
    #                   f"={opts['trialavg']}"
    #     gmap.fig.suptitle(title_str)
    #     plt.show()
    # %%
    # # %% loop through rdms by panel, and save as pdf plots
    # import importlib as imp
    # imp.reload(step05c_compute_trial_rdms)
    #
    # for panel, rdm_concat in rdm_concat_by_panel.items():
    #     stim_row = rdm_concat.trial_row['row_stim'].to_numpy()
    #     stim_col = rdm_concat.trial_col['col_stim'].to_numpy()
    #
    #     # make png folder
    #     png_folder = f'rdm_concat__{imaging_type}__{panel}__{respvec_type}__' \
    #                  f'{metric_type}__pngs'
    #     png_folder = OUTPUT_DIR.joinpath(png_folder)
    #     png_folder.mkdir(exist_ok=True)
    #
    #     # create pdf file
    #     pdf_file = f"rdm_concat__{imaging_type}__{panel}__{respvec_type}__{metric_type}.pdf"
    #     pdf_file = OUTPUT_DIR.joinpath(pdf_file)
    #     pdf = PdfPages(pdf_file)
    #
    #     for i in range(rdm_concat_by_panel[panel].dims['flacq']):
    #         ds_rdm = rdm_concat_by_panel[panel].isel(flacq=i)
    #
    #         fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    #         ax.set_facecolor('0.85')
    #
    #         ax = step05c_compute_trial_rdms.plot_rdm(ds_rdm, data_var='Fc_zscore', ax=ax,
    #                                                  stim_row=stim_row, stim_col=stim_col,
    #                                                  metric=metric_type)
    #         ax.set_title(f'Fc_zscore: {respvec_type} {metric_type}')
    #         fig.suptitle(str(ds_rdm.flacq.to_numpy()))
    #         plt.show()
    #
    #         # save pdf page
    #         pdf.savefig(fig, facecolor='0.85')
    #
    #         # save png file
    #         png_file = f'rdm_concat__{imaging_type}__{panel}__{respvec_type}__' \
    #                    f'{metric_type}__flacq{i:02d}.png'
    #         fig.savefig(png_folder.joinpath(png_file), facecolor='0.85')
    #     pdf.close()
