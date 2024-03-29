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
import seaborn as sns
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from parse import parse
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
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

RESPVEC_OUTPUT_FILENAME = "respvec_agg__{respvec}__trialavg{trialavg}__goodxids{" \
                          "good_xids_only}__stdize_{ufunc_name}.nc"


def fix_string_serialized_attrs(d):
    fixed_d = copy.deepcopy(d)

    for k, v in fixed_d.items():
        if v in ['True', 'False']:
            fixed_d[k] = bool(v)
        elif v == 'None':
            fixed_d[k] = None

    return fixed_d


def parse_respvec_params_from_filename(filename, convert_dtypes=True):
    results = parse(RESPVEC_OUTPUT_FILENAME, filename)
    if convert_dtypes:
        parsed_params = fix_string_serialized_attrs(results.named)
    else:
        parsed_params = results.named
    return parsed_params


def get_respvec_param_combinations(
        respvec=['mean_peak', 'max_peak'],
        trialavg=[True, False],
        good_xids_only=[True, False],
        ufunc_name=[None, 'robust_scale', 'quantile_transform', 'scale'],
):
    """Get a list of all possible parameter combinations for aggregating respvec datasets."""

    # standardization_ufunc = [(getattr(preprocessing, k), v)
    #                          for k, v in ufunc_kwargs_lookup.items()]

    # respvec_param_grid_options = dict(
    #         respvec=['mean_peak', 'max_peak'],
    #         trialavg=[True, False],
    #         good_xids_only=[True, False],
    #         ufunc_name=[None, 'robust_scale', 'quantile_transform', 'scale'],
    # )
    respvec_param_grid_options = dict(
            respvec=respvec, trialavg=trialavg, good_xids_only=good_xids_only, ufunc_name=ufunc_name
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
                params['ufunc'] = getattr(preprocessing, params['ufunc_name'])
                params['ufunc_kwargs'] = ufunc_kwargs_lookup[params['ufunc_name']]
    return respvec_param_grid


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

    ds_file = natmixconfig.NAS_PROC_DIR / flat_acq.stat_file().with_name('RDM_trials').joinpath(
            filename)
    rdm_ds = xr.load_dataset(ds_file)
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


def add_stim_indices_to_respvec(ds_respvec, index_stimuli_keys=None):
    if index_stimuli_keys is None:
        index_stimuli_keys = ['stim', 'stim_occ']

    stim_info = ryeutils.index_stimuli(ds_respvec['stim'].to_numpy())

    mi_stim = {k: ('trials', v) for k, v in stim_info.items()
               if k != 'stim' and k in index_stimuli_keys}

    return ds_respvec.assign_coords(mi_stim)


def make_respvec_trials_stim_multiindex(ds_respvec, index_stimuli_keys=None):
    """Takes a respvec dataset, and converts the `trials` coordinate into a stim multiindex.

    For combining respvec datasets across acquisitions with different #s of trials and/or
    different trial structure.

    End up with a dataset that looks like:
      * trials              (trials) MultiIndex
      - stim                (trials) object '1-6ol @ -3.0' ... 'pfo @ 0.0'
      - stim_occ            (trials) int64 0 1 2 3 0 1 2 0 1 2 ... 3 0 1 2 3 0 1 2 3


    Example:
        >>> ds_respvec_mi = make_respvec_trials_stim_multiindex(ds_respvec)

    """
    if index_stimuli_keys is None:
        index_stimuli_keys = ['stim', 'stim_occ']

    ds_respvec_mi = add_stim_indices_to_respvec(ds_respvec) \
        .reset_index('trials').set_index(
            trials=index_stimuli_keys)
    return ds_respvec_mi


# to be able to combine respvec datasets with differing trial structures (different #s of trials,
# stimulus repeats, etc).

# >>> mi_respvec_list = [make_respvec_trials_stim_multiindex(item) for item in ds_respvec_list]
def get_common_trial_mi_from_mi_respvec_list(mi_respvec_list):
    df_stim_list = [item['trials'].to_index().to_frame(index=False) for item in mi_respvec_list]
    df_stim = pd.concat(df_stim_list, axis=0).drop_duplicates()
    common_mi_stim = pd.MultiIndex.from_frame(df_stim)
    return common_mi_stim


def combine_respvecs_with_trial_structure(ds_respvec_list, index_stimuli_keys=None):
    """Combines """
    if index_stimuli_keys is None:
        index_stimuli_keys = ['stim', 'stim_occ']

    mi_respvec_list = [
        make_respvec_trials_stim_multiindex(item, index_stimuli_keys=index_stimuli_keys)
        .reset_index('cells') for item in
        ds_respvec_list]
    ds_respvec_concat = xr.concat(mi_respvec_list, 'cells')
    return ds_respvec_concat


def align_respvec_trials_by_stim_index(ds_respvec_list, index_stimuli_keys=None):
    """
    Returns respvecs with a common trial ordering (for concatenating responses across
    acquisitions with different trial structures).

    Args:
        ds_respvec_list (List[xr.Dataset]): respvec dataset, with dim 'trials' and 'stim' coord.
        index_stimuli_keys (List[str]): which stim index keys to use

            (differs if you want to maintain repeated stimulus block structures from an
            acquisition with consecutive repeats, like in the natural_mixtures project)

    Returns:
        aligned_respvecs (List[xr.Dataset]): respvec datasets, expanded along the trial
            dim so that all of them have the same 'trial' index. Missing trials will be filled with
            nans.

            Original trial index will be stored in the coordinate 'trials_'.

    Examples:

        >>> ds_respvec_list = load_respvecs_by_flacq_list(flacq_list, 'mean_peak')
        >>> processed_respvecs = process_ds_respvec_list(ds_respvec_list, \
         trialavg=False, good_xids_only=False, standardize=False)
        >>> aligned_respvecs = align_respvec_trials_by_stim_index(processed_respvecs)
        <xarray.Dataset>
        Dimensions:             (trials: 50, cells: 3095)
        Coordinates: (12/14)
          * trials              (trials) MultiIndex
          - stim                (trials) object 'pfo @ 0.0' ... 'MethOct @ -3.0'
          - stim_occ            (trials) int64 0 1 2 0 1 2 0 1 2 0 ... 3 3 3 3 3 3 3 3 3
          * cells               (cells) int64 0 1 2 3 4 5 ... 3090 3091 3092 3093 3094
            iscell              (cells) int64 1 1 1 1 1 1 1 1 1 1 ... 1 1 1 1 1 1 1 1 1
            cellprob            (cells) float64 0.03595 0.4821 0.6495 ... 0.4452 0.01511
            iplane              (cells) int64 2 2 2 2 2 2 2 2 ... 15 15 15 15 15 15 15
            xid0                (cells) int64 38 38 38 39 38 29 33 ... 27 25 3 25 39 39
            ...                  ...
            date_imaged         <U10 '2022-09-21'
            fly_num             int64 1
            thorimage           <U10 'odorspace0'
            abbrev              (trials) object 'pfo' 'pfo' 'pfo' ... nan nan nan
            iscell_good_xid0    (cells) bool False False False ... False False False
            trials_             (trials) float64 0.0 1.0 2.0 3.0 4.0 ... nan nan nan nan
        Data variables:
            Fc                  (trials, cells) float64 -51.13 -4.842 -48.97 ... nan nan
            F                   (trials, cells) float64 -42.87 4.697 -39.27 ... nan nan
            Fneu                (trials, cells) float64 11.79 13.63 13.86 ... nan nan
            spks                (trials, cells) float64 0.0 -1.149 -0.09176 ... nan nan
            Fc_smooth           (trials, cells) float64 -53.39 -10.52 -53.08 ... nan nan
            Fc_zscore           (trials, cells) float64 -0.4991 -0.02775 ... nan nan
            Fc_zscore_smoothed  (trials, cells) float64 -0.5212 -0.0603 ... nan nan
            Fc_normed           (trials, cells) float64 -0.04159 -0.002312 ... nan nan
            Fc_normed_smoothed  (trials, cells) float64 -0.04344 -0.005025 ... nan nan

    """
    if index_stimuli_keys is None:
        index_stimuli_keys = ['stim', 'stim_occ']

    # multiindex respvecs
    mi_respvec_list = [make_respvec_trials_stim_multiindex(item,
                                                           index_stimuli_keys=index_stimuli_keys)
                       for item in ds_respvec_list]

    mi_common = get_common_trial_mi_from_mi_respvec_list(mi_respvec_list).sort_values()
    aligned_mi_respvecs = [item.reindex(dict(trials=mi_common)) for item in mi_respvec_list]
    return aligned_mi_respvecs


def get_rdm_concat_from_aligned_respvec_list(aligned_respvecs, metric='correlation'):
    da_rdm_list = []

    for iacq, da_respvec in enumerate(aligned_respvecs):
        mi_trials = da_respvec['trials'].to_index()
        mi_names = mi_trials.names
        da_rdm = xr.apply_ufunc(pairwise_distances,
                                da_respvec,
                                input_core_dims=[['trials', 'cells']],
                                output_core_dims=[['trial_row', 'trial_col']],
                                vectorize=True,
                                kwargs=dict(metric=metric, force_all_finite=False)
                                )
        da_rdm = da_rdm.assign_coords(
                trial_row=mi_trials.rename([f"row_{item}" for item in mi_names]),
                trial_col=mi_trials.rename([f"col_{item}" for item in mi_names])
        )
        da_rdm_list.append(da_rdm)

    da_rdm_concat = xr.concat(da_rdm_list, 'iacq')
    return da_rdm_concat


def plot_aligned_rdm_concat_heatmaps(rdm_concat):
    stim_row = rdm_concat.trial_row['row_stim'].to_numpy()
    stim_col = rdm_concat.trial_col['col_stim'].to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    stim_row = rdm_concat.trial_row['row_stim'].to_numpy()
    stim_col = rdm_concat.trial_col['col_stim'].to_numpy()
    ax = step05c_compute_trial_rdms.plot_rdm(rdm_concat.sel(iacq=0), data_var='Fc_zscore', ax=ax,
                                             stim_row=stim_row, stim_col=stim_col,
                                             )


def plot_unaligned_rdm_heatmaps(rdms):
    figures = []
    for ds_rdm in rdms:
        stim_row = ds_rdm.trial_row['row_stim'].to_numpy()
        stim_col = rdm.trial_col['col_stim'].to_numpy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
        stim_row = ds_rdm.trial_row['row_stim'].to_numpy()
        stim_col = ds_rdm.trial_col['col_stim'].to_numpy()
        ax = step05c_compute_trial_rdms.plot_rdm(ds_rdm, data_var='Fc_zscore', ax=ax,
                                                 stim_row=stim_row, stim_col=stim_col,
                                                 )

        figures.append(fig)
    return figures


def load_respvecs_by_flacq(flat_acq, respvec, iacq=None):
    """Load RDMs from a FlatFlyAcquisition.

       Args:
           iacq (Union[None, int]): # assigned to the acquisition index
           flat_acq (FlatFlyAcquisition): the FlatFlyAcquisition to load RDMs from
           respvec (str): the response vector type ('mean_peak' or 'max_peak')

       Returns:
           ds_respvec (xarray.Dataset): the respvec Dataset w/ flat acq metadata added to
           coordinates
       """
    filename = flat_acq.stat_file(relative_to=natmixconfig.NAS_PROC_DIR)\
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
            ds = ds.assign_coords(abbrev=('stim',
                                          [item.split('@')[0].strip() for item in
                                           ds.stim.to_numpy()]
                                          ),
                                  )
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
            ds_standardized = ds_good.transpose()

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


def aggregate_respvecs(flat_acq_list,
                       respvec='mean_peak',
                       trialavg=True,
                       good_xids_only=False,
                       standardize=False,
                       ufunc=None,
                       ufunc_kwargs=None,
                       **kwargs):
    """Top-level function for combining respvec datasets with different preprocessing options.

    """

    # make attr dict from respvec parameters
    attrs = dict(
            respvec=respvec, trialavg=trialavg, good_xids_only=good_xids_only,
            standardize=standardize,
            ufunc=ufunc.__name__ if ufunc is not None else None
    )
    if ufunc_kwargs is not None:
        for k, v in ufunc_kwargs.items():
            attrs[f"{ufunc.__name__}__{k}"] = v

    ds_respvec_list = load_respvecs_by_flacq_list(flat_acq_list, respvec)
    ds_processed_respvec_list = process_ds_respvec_list(ds_respvec_list,
                                                        trialavg=trialavg,
                                                        good_xids_only=good_xids_only,
                                                        standardize=standardize,
                                                        ufunc=ufunc,
                                                        ufunc_kwargs=ufunc_kwargs,
                                                        )
    ds_respvec_cat = xr.concat(ds_processed_respvec_list, 'cells')

    ds_respvec_cat = ds_respvec_cat.assign_attrs(attrs)
    return ds_respvec_cat


# %% Functions for combining RDMs


def combine_rdms(rdm_list):
    """Combine a list of RDMs into a single RDM.

    Args:
        rdm_list (List[xarray.Dataset]): the RDMs to combine

    Returns:
        rdm_ds (xarray.Dataset): the combined RDM
    """
    rdm_ds = xr.concat(rdm_list, dim='trial')
    return rdm_ds


def get_stim_multiindex(stim_list, index_stimuli_keys=['stim', 'run_occ', 'idx_in_run']):
    """"""
    stim_info = ryeutils.index_stimuli(stim_list)
    df_stim = pd.DataFrame(stim_info).loc[:, index_stimuli_keys]

    mi_stim = pd.MultiIndex.from_frame(df_stim)
    return mi_stim


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

    ds_prepared = ds_prepared.assign_coords(
            date_imaged=ds_prepared.attrs['date_imaged'],
            fly_num=ds_prepared.attrs['fly_num'],
            thorimage=ds_prepared.attrs['thorimage']
    )
    return ds_prepared


def prepare_ds_concat_to_save(ds_concat):
    """Fix concatenated RDM dataset (reset multiindices) so it can be saved as a .netcdf file."""
    return ds_concat.reset_index(['trial_row', 'trial_col'])


def make_ds_concat_multiindex(ds_concat, index_stimuli_keys=['stim', 'run_occ', 'idx_in_run']):
    """After loading ds_concat, convert coords to multiindex.

    This avoids issues with having to save the concatenated RDMs as
    """
    row_coords = [f'row_{k}' for k in index_stimuli_keys]
    col_coords = [f'col_{k}' for k in index_stimuli_keys]
    return ds_concat.set_index(trial_row=row_coords,
                               trial_col=col_coords)


def plot_concat_rdm_heatmaps(rdm_concat, stim_idx_ord=None, respvec=None, metric=None,
                             trialavg=None, acq_dimname='flacq', data_var='Fc_zscore'):
    """

    Args:
        data_var ():
        rdm_concat ():
        stim_idx_ord ():
        respvec ():
        metric ():
        trialavg ():
        acq_dimname ():

    Returns:

    Examples:
        >>> fig_list = plot_concat_rdm_heatmaps(rdm_concat.rename({'iacq': 'flacq'}))
        >>> for fig in fig_list:
        >>>     fig.show()

    """
    stim_row = rdm_concat.trial_row['row_stim'].to_numpy()
    stim_col = rdm_concat.trial_col['col_stim'].to_numpy()

    if respvec is None:
        respvec = rdm_concat.attrs['respvec']
    if metric is None:
        metric = rdm_concat.attrs['metric']
    if trialavg is None:
        trialavg = rdm_concat.attrs['trialavg']

    figures = []

    for flacq_idx in range(rdm_concat.dims[acq_dimname]):
        ds_rdm = rdm_concat.isel(flacq=flacq_idx)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
        ax.set_facecolor('0.85')

        ax = step05c_compute_trial_rdms.plot_rdm(ds_rdm, data_var=data_var, ax=ax,
                                                 stim_row=stim_row, stim_col=stim_col,
                                                 )
        ax.set_title(f'{data_var}: {respvec} {metric}, trialavg={trialavg}',
                     fontsize=8)

        # basename = str(ds_rdm.filename_base.to_numpy())
        title_str = "# {:02d}: {}/{:02d}\n{}".format(
                ds_rdm['flacq'].item(),
                ds_rdm['date_imaged'].item(),
                ds_rdm['fly_num'].item(),
                ds_rdm['thorimage'].item() if 'thorimage' in ds_rdm.coords.keys() else ''

        )
        fig.suptitle(title_str, fontsize=10)
        plt.show()

        figures.append(fig)
    return figures


def aggregate_rdms(flat_acq_list,
                   respvec='mean_peak',
                   metric='correlation',
                   trialavg=True,
                   good_xids_only=True,
                   index_stimuli_keys=['stim', 'stim_occ']):
    """

    Args:
        flat_acq_list ():
        respvec ():
        metric ():
        trialavg ():
        good_xids_only ():
        index_stimuli_keys ():

    Returns:

    Examples:
        >>>

    """
    rdms = load_rdms_by_flacq_list(flat_acq_list,
                                   respvec=respvec,
                                   metric=metric,
                                   trialavg=trialavg)
    prepared_rdms = [prepare_dataset_for_concat(rdm, index_stimuli_keys=index_stimuli_keys)
                     for rdm in rdms]
    prepared_rdms = [rdm.assign_coords(flacq=iflacq)
                     for iflacq, rdm in enumerate(prepared_rdms)]

    rdm_concat = xr.concat(prepared_rdms, 'flacq', combine_attrs='drop')
    rdm_concat = rdm_concat.assign_attrs(
            dict(
                    respvec=respvec,
                    metric=metric,
                    trialavg=str(trialavg),
                    good_xids_only=str(good_xids_only),
                    index_stimuli_keys=index_stimuli_keys
            )
    )
    return rdm_concat


# %%
# analysis_outputs/RDM_trials/data_by_acq
# analysis_outputs/RDM_trials/plots_by_acq
# params = dict(respvec='mean_peak',
#               metric='correlation',
#               trialavg=False,
#               good_xids_only=True,
#               )
#
# rdm_list = load_rdms_by_flacq_list(flat_acqs,
#                                    **{k: v for k, v in params.items() if k !='good_xids_only'})
#
# rdm_list = [prepare_dataset_for_concat(item, index_stimuli_keys=['stim', 'stim_occ'])
#             for item in rdm_list]
# ds_rdm_concat = aggregate_rdms(flat_acqs, **params)
#
# if params['trialavg']:
#     pass
# if not params['trialavg']:
#     respvecs = load_respvecs_by_flacq_list(flat_acqs, 'mean_peak')
#     aligned_respvec_cat = combine_respvecs_with_trial_structure(respvecs)
#     trimmed_respvec_cat = aligned_respvec_cat.dropna('trials')
#     common_trial_mi = trimmed_respvec_cat['trials'].to_index()
#
#     mi_row = common_trial_mi.set_names([f"row_{item}" for item in common_trial_mi.names])
#     mi_col = common_trial_mi.set_names([f"col_{item}" for item in common_trial_mi.names])
#     ds_rdm_concat_trimmed = ds_rdm_concat.sel(trial_row=mi_row, trial_col=mi_col, drop=True)
#
#
# fig_list = plot_concat_rdm_heatmaps(ds_rdm_concat)
# trimmed_fig_list = plot_concat_rdm_heatmaps(ds_rdm_concat_trimmed)
#
# # %%
# for figc, figc_trimmed, flacq in zip(fig_list, trimmed_fig_list, flat_acqs):
#     save_folder = natmixconfig.NAS_PRJ_DIR.joinpath('analysis_outputs',
#                                                     'RDM_trials',
#                                                     'plots_by_acq',
#                                                     flacq.filename_base()
#                                                     )
#     save_folder.mkdir(exist_ok=True, parents=True)
#
#     # save complete trial RDM
#     filename_template = 'heatmap__RDM__{respvec}__{metric}__trialavg{trialavg}.pdf'
#     filename = filename_template.format(**params)
#     figc.savefig(save_folder.joinpath(filename), transparent=False)
#     figc.savefig(save_folder.joinpath(filename).with_suffix('.png'))
#
#     # save trimmed trial RDM
#     trimmed_filename_template = 'heatmap__RDM__{respvec}__{metric}__trialavg{trialavg}' \
#                                 '__trimmed.pdf'
#     trimmed_filename = trimmed_filename_template.format(**params)
#     figc_trimmed.savefig(save_folder.joinpath(trimmed_filename), transparent=False)
#     figc_trimmed.savefig(save_folder.joinpath(trimmed_filename).with_suffix('.png'))
#
# for figc, flacq in zip(fig_list, flat_acqs):
#     save_folder = natmixconfig.NAS_PRJ_DIR.joinpath('analysis_outputs',
#                                                     'RDM_trials',
#                                                     'plots_by_acq',
#                                                     flacq.filename_base()
#                                                     )
#     save_folder.mkdir(exist_ok=True, parents=True)
#
#     # save complete trial RDM
#     filename_template = 'heatmap__RDM__{respvec}__{metric}__trialavg{trialavg}.pdf'
#     filename = filename_template.format(**params)
#     figc.savefig(save_folder.joinpath(filename), transparent=False)
#     figc.savefig(save_folder.joinpath(filename).with_suffix('.png'))


# %%
# def aggregate_rdms(flat_acq_list, respvec, metric, trialavg, drop_stim=None, stim_idx_keys, ):
#     load_rdms_by_flacq_list(flacqs,
#                             respvec=opts['respvec'],
#                             metric=opts['metric'],
#                             trialavg=opts['trialavg'])
# %%
if __name__ == '__main__':

    # load FlatFlyAcquisitions
    all_flat_acqs = pydantic.parse_file_as(List[FlatFlyAcquisitions], natmixconfig.MANIFEST_FILE)

    panel = PanelMovies(prj='odor_space_collab', panel='odorspace')
    flat_acqs = filter_flacq_list(
            all_flat_acqs,
            allowed_imaging_type=None,
            allowed_movie_types=panel.movies,
            has_s2p_output=True)

    print('\n')
    print('odorspace flat_acqs:')
    print('-------------------')
    for item in flat_acqs:
        print(item.filename_base())

    ds_rdm_concat = aggregate_rdms(flat_acqs,
                                   respvec='mean_peak',
                                   metric='correlation',
                                   trialavg=False)
    fig_list = plot_concat_rdm_heatmaps(ds_rdm_concat)
    # %%
    trimmed_stim_list = \
        ['1-6ol @ -3.0',
         # '1-6ol @ -4.0',
         '1p3ol @ -3.0',
         '2-but @ -3.0',
         # '2-but @ -5.0',
         'ECin @ -3.0',
         'MethOct @ -3.0',
         'PropAc @ -3.0',
         'benz @ -3.0',
         'd-dlac @ -3.0',
         # 'eug @ -3.0',
         'g-6lac @ -3.0',
         # 'pfo @ 0.0'
         ]

    tidy_rdm = (ds_rdm_concat
                .to_dataframe()
                .set_index(['date_imaged', 'fly_num', 'thorimage'], append=True)
                .loc[:, ['Fc_zscore']]
                .dropna()
                )

    trimmed_tidy_rdm = tidy_rdm.query('(trial_row in @trimmed_stim_list) & (trial_col in '
                                      '@trimmed_stim_list)')

    fig_list = []
    for name, df in tidy_rdm.groupby(level=['flacq', 'date_imaged', 'fly_num', 'thorimage']):
        print('---')
        print(name)
        # print(df)

        df_rdm = (df.reset_index(level=['trial_row', 'trial_col'])
                  .pivot(index='trial_row', columns='trial_col', values='Fc_zscore')
                  )
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 6))

        meshgrid = sns.heatmap(data=1 - df_rdm,
                               square=True,
                               vmin=-1,
                               vmax=1,
                               cmap='RdBu_r',
                               linewidths=0,
                               ax=ax,
                               cbar_kws=dict(shrink=0.5)
                               )
        title_str = "{:02d}: {}/{:02}/{}".format(*name)
        fig.suptitle(title_str, fontsize=10)
        plt.show()
        fig_list.append(fig)

    for figc, flacq in zip(fig_list, flat_acqs):
        save_folder = natmixconfig.NAS_PRJ_DIR.joinpath('analysis_outputs',
                                                        'RDM_trials',
                                                        'plots_by_acq',
                                                        flacq.filename_base()
                                                        )
        save_folder.mkdir(exist_ok=True, parents=True)

        # save complete trial RDM
        # filename_template = 'heatmap__RDM__mean_peak__correlation__trialavgTrue.pdf'
        # filename = filename_template
        filename = 'heatmap__RDM__mean_peak__correlation__trialavgTrue.pdf'

        figc.savefig(save_folder.joinpath(filename), transparent=False)
        figc.savefig(save_folder.joinpath(filename).with_suffix('.png'))

    save_folder.parent.joinpath(filename)
def main2():
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

        flat_acqs = list(filter(lambda x: suite2p_helpers.is_3d(x.stat_file(
                relative_to=natmixconfig.NAS_PROC_DIR)),
                                flat_acqs, ))

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

    return True
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
