import copy
import json
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pydantic
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.stats import zscore

import natmixconfig
import pathparse
import trial_tensors
import xr_helpers
from pydantic_models import FlatFlyAcquisitions
# import suite2p
# import rastermap
# from rastermap.mapping import Rastermap
from s2p import rasterplot, suite2p_helpers

plt.rcParams.update({'pdf.fonttype': 42,
                     'text.usetex': False})

# Set project directory
from natmixconfig import *
# NAS_PRJ_DIR = Path("/local/storage/Remy/natural_mixtures")
# NAS_PROC_DIR = NAS_PRJ_DIR.joinpath("processed_data")


def process_traces(cells_x_time, win=90):
    """Process fluorescence array (cells x time) """
    traces = dict()
    traces['Fc_zscored'] = zscore(cells_x_time, axis=-1)
    print('Fc_zscored computed.')

    traces['Fc_normed'] = rasterplot.norm_traces(cells_x_time)
    print('Fc_normed computed.')

    traces['Fc_smooth'] = gaussian_filter1d(cells_x_time, sigma=1)
    print('Fc_smooth computed.')

    traces['F0'] = percentile_filter(traces['Fc_smooth'], size=win, percentile=5)
    print('F0 computed.')

    traces['F0_smooth'] = gaussian_filter1d(traces['F0'], sigma=win)
    print('F0_smooth computed.')

    traces['F_bc'] = cells_x_time - traces['F0_smooth']  # baseline subtraced/corrected
    print('F_bc computed.')

    traces['F_bc_zscored'] = zscore(traces['F_bc'], axis=-1)
    print('F_bc computed.')

    traces['F_bc_normed'] = rasterplot.norm_traces(traces['F_bc'])
    print('F_bc_normed computed.')

    return traces


def traces_2_trial_tensors(xrds):
    ts = xrds.time.to_numpy()
    olf_ict = xrds.attrs['olf_ict']
    trial_ts = np.arange(-5, 20.01, 0.25)
    stim = xrds.attrs['stim']

    iscell = xrds.coords['iscell'].to_numpy()
    cellprob = xrds.coords['cellprob'].to_numpy()

    data_vars = {}

    for k in xrds.data_vars.keys():
        cells_x_time = xrds[k].to_numpy()
        trials_x_cells_x_time = trial_tensors.make_trial_tensor(cells_x_time, ts, olf_ict, trial_ts)

        da = xr_helpers.make_xrda_traces(trials_x_cells_x_time,
                                         stim,
                                         xrds.cell_ids.to_numpy(),
                                         trial_ts,
                                         attrs=xrds.attrs)
        data_vars[k] = da

    xrds_trial_tensors = xr.Dataset(data_vars=data_vars, attrs=xrds.attrs)
    xrds_trial_tensors = xrds_trial_tensors.assign_coords(dict(cellprob=('cell_ids', cellprob)))
    xrds_trial_tensors = xrds_trial_tensors.assign_coords(dict(iscell=('cell_ids', iscell)))

    return xrds_trial_tensors


def suite2p_2_xarray(Fall, ts, attrs):
    """ Converts suite2p outputs into xarray dataset containing F, Fneu, and Fc. """
    if isinstance(Fall, Path):
        stat_file = Fall.with_name('stat.npy')
        Fall = loadmat(Fall, squeeze_me=True)

    iscell, cellprob = Fall['iscell'].T
    iscell = iscell.astype('int')

    Fc = Fall['F'] - 0.7 * Fall['Fneu']
    n_cells, T = Fc.shape

    data_vars = {'Fc': (["cells", "time"], Fc),
                 'F': (["cells", "time"], Fall['F']),
                 'Fneu': (["cells", "time"], Fall['Fneu']),
                 'spks': (["cells", "time"], Fall['spks']),
                 }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            cells=range(n_cells),
            time=ts
        ),
        attrs=attrs
    )

    ds = ds.assign_coords(dict(iscell=('cells', iscell)))
    ds = ds.assign_coords(dict(cellprob=('cells', cellprob)))

    is_multiplane = suite2p_helpers.is_3d(stat_file)
    if is_multiplane:
        iplane = [int(stat['iplane']) for stat in Fall['stat']]
        ds = ds.assign_coords(dict(iplane=('cells', iplane)))
    return ds


def flacq_2_xarray_attrs(flacq):
    """Takes a flat linked acquisition and returns a dict containing attrs for xarray

    Args:
        flacq ():

    Returns:
        dict:
    """
    if isinstance(flacq, dict):
        MOV_DIR = pathparse.flacq2dir(flacq)
        xr_attrs = copy.deepcopy(flacq)
    elif isinstance(flacq, FlatFlyAcquisitions):
        MOV_DIR = flacq.mov_dir()
        xr_attrs = flacq.dict()

    # load frame and trial timing info
    timestamps = np.load(MOV_DIR.joinpath('timestamps.npy'), allow_pickle=True).item()

    with open(MOV_DIR.joinpath('stim_list.json'), 'r') as f:
        stim_list = json.load(f)

    # add time and stim info to attrs
    xr_attrs['stack_times'] = timestamps['stack_times']
    xr_attrs['olf_ict'] = timestamps['olf_ict']
    xr_attrs['stim'] = stim_list['stim_list_flatstr']
    return xr_attrs


def xrds_suite2p_outputs_2_trials(xrds_file):
    """
    Converts xrds_suite2p_outputs(_xid0_.nc --> xrds_suite2p_output_trials(_xid0).nc

    Resamples/interpolates 2D timeseries data w/ dims ['cells', 'time'] to 3D trial structured data
    w/ dims ['trials', 'cells', 'time'].

    - Copy attrs
    - For non-dim coordinates,
    """
    ds = xr.load_dataset(xrds_file)
    attrs = ds.attrs.copy()
    return None


def xrds_suite2p_outputs_xid0_2_xrds_suite2p_output_trials_xid0(xrds_file, trial_ts=None):
    """Converts xrds_suite2p_outputs_xid0.nc --> xrds_suite2p_output_trials_xid0.nc

    Args:
        xrds_file (object):
        trial_ts ():

    """
    if trial_ts is None:
        trial_ts = np.arange(-5, 20, 0.5)

    if isinstance(xrds_file, Path):
        ds = xr.load_dataset(xrds_file)
    elif isinstance(xrds_file, xr.Dataset):
        ds = xrds_file

    attrs = ds.attrs.copy()

    dim_names = list(ds.coords.dims)
    coord_names = list(ds.coords.keys())
    nondim_coords = [item for item in coord_names if item not in dim_names]

    n_trials = len(ds.attrs['stim'])
    cells = ds.coords['cells'].to_numpy()

    data_vars = {}
    for k, v in ds.data_vars.items():
        trials_x_cells_x_time = trial_tensors.make_trial_tensor(v.to_numpy(),
                                                                ts=ds.time.to_numpy(),
                                                                stim_ict=ds.attrs['olf_ict'],
                                                                trial_ts=trial_ts)
        data_vars[k] = (("trials", "cells", "time"), trials_x_cells_x_time)

    ds_trial_tensors = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            trials=range(n_trials),
            cells=cells,
            time=trial_ts,
            stim=('trials', ds.attrs['stim'])
        ),
        attrs=attrs
    )

    for cname in nondim_coords:
        if 'time' not in ds.coords[cname].dims:
            ds_trial_tensors = ds_trial_tensors.assign_coords({cname: (ds.coords[cname].dims, ds.coords[cname].values)})

    return ds_trial_tensors




def flacq_2_xrds_suite2p_outputs(flat_acq, save_netcdf=False):
    """Creates xarray dataset from all suite2p/combined subfolders found in the movie folder for `flat_acq`

    Returns:
        xr.Dataset: Dataset containing traces F, Fc, Fneu,
    """

    # MOV_DIR = pathparse.flacq2dir(flat_acq)
    MOV_DIR = flat_acq.mov_dir()

    # load thorimage metadata
    # meta = utils2p.Metadata(MOV_DIR.joinpath('Experiment.xml'))

    # load frame and trial timing info
    timestamps = np.load(MOV_DIR.joinpath('timestamps.npy'), allow_pickle=True).item()

    # load stimulus info
    with open(MOV_DIR.joinpath('stim_list.json'), 'r') as f:
        stim_list = json.load(f)

    stat_file = flat_acq.stat_file(relative_to=natmixconfig.NAS_PROC_DIR)
    fname_Fall = stat_file.with_name('Fall.mat')
    # fname_Fall = list(MOV_DIR.rglob("suite2p/combined/Fall.mat"))[0]

    # add time and stim info to attrs
    attrs = copy.deepcopy(flat_acq.dict())
    attrs['stack_times'] = timestamps['stack_times']
    attrs['olf_ict'] = timestamps['olf_ict']
    attrs['stim'] = stim_list['stim_list_flatstr']
    attrs['s2p_stat_file'] = str(fname_Fall.with_name('stat.npy'))
    attrs['fname_Fall'] = str(fname_Fall)

    ds = suite2p_2_xarray(fname_Fall, timestamps['stack_times'], attrs)

    if save_netcdf:
        ds.to_netcdf(stat_file.with_name('xrds_suite2p_outputs.nc'))

    return ds


# %% load flat acquisition manifesto
if __name__ == '__main__':
    with open(NAS_PRJ_DIR.joinpath('manifestos', 'flat_linked_thor_acquisitions.json'), 'r') as f:
        flat_linked_acquisitions = json.load(f)

    flat_linked_acquisitions = pydantic.parse_obj_as(List[FlatFlyAcquisitions],
                                                     flat_linked_acquisitions)

    for i, item in enumerate(flat_linked_acquisitions):
        print(f"{i}\t{item.mov_dir()}")
    # %% Save to xrds_s2p_traces.nc in suite2p/combined folder
    # also copied to natural_mixtures/report_data

    flacqs_to_process = flat_linked_acquisitions[-3:]

    print(f"flacqs_to_process:")
    print(f"-----------------")
    for i, item in enumerate(flacqs_to_process):
        print(f"{i}\t{item.mov_dir()}")

    # %% convert suite2p outputs (under `suite2p/combined/Fall.npy`) to xr.Dataset containing relevant metadata and
    #  trial/timing information.

    for i, flacq in enumerate(flacqs_to_process):
        mov_dir = flacq.mov_dir()
        print(f"\n---{i}/{len(flacqs_to_process)}---")
        print(flacq)
        print(f"\nMOV_DIR: {mov_dir}\n")

        stat_file = list(mov_dir.rglob("combined/stat.npy"))[0]
        print(f"\n\tstat_file: {stat_file}")

        # create xrds_suite2p_outputs.nc
        # -------------------------------------

        xrds_suite2p_outputs = flacq_2_xrds_suite2p_outputs(flacq.dict(), save_netcdf=True)
        print(f"\n\t- saved {stat_file.relative_to(mov_dir).with_name('xrds_suite2p_outputs.nc')}")

        # --------------------------------------------
        # run initial rastermap embedding
        # --------------------------------------------
        rasterplot.run_initial_rastermap_embedding(stat_file)

        xrds_file = stat_file.with_name("xrds_suite2p_outputs.nc")
        fig1, axarr1 = rasterplot.plot_initial_rastermap_clustering(xrds_file)
        plt.show()

        pdf_file = xrds_file.with_name('rastermap_embedding_allcells.pdf')
        if ~pdf_file.is_file():
            with PdfPages(pdf_file) as pdf:
                pdf.savefig(fig1)
                print('\trastermap_embedding_allcells.pdf saved.')
        else:
            print(f"\tpdf file already exists: {pdf_file.relative_to(NAS_PROC_DIR)}")

        # -------------------------------------------
        # create xrds_suite2p_traces.nc
        # -------------------------------------------
        stack_times = xrds_suite2p_outputs.time.to_numpy()
        Fc = xrds_suite2p_outputs['Fc'].to_numpy()
        fps = round(1 / np.diff(stack_times).mean())

        trace_data_vars = process_traces(Fc, win=45 * fps)

        xrds_suite2p_traces = xrds_suite2p_outputs.coords.to_dataset()
        for k, v in trace_data_vars.items():
            xrds_suite2p_traces[k] = (('cells', 'time'), v)
        xrds_suite2p_traces.to_netcdf(stat_file.with_name('xrds_suite2p_traces.nc'))
        print(f"\t- saved {stat_file.relative_to(mov_dir).with_name('xrds_suite2p_traces.nc')}")


        # -------------------------------------------
        # create xrds_suite2p_trials.nc
        # -------------------------------------------
        olf_ict = xrds_suite2p_traces.attrs['olf_ict']
        stim = xrds_suite2p_traces.attrs['stim']
        cell_ids = xrds_suite2p_traces.cells.to_numpy()

        trial_ts = np.arange(-5, 20.01, 0.25)

        trial_data_vars = {}
        for k in xrds_suite2p_traces.data_vars.keys():
            cells_x_time = xrds_suite2p_traces[k].to_numpy()
            trials_x_cells_x_time = trial_tensors.make_trial_tensor(cells_x_time, stack_times, olf_ict, trial_ts)
            trial_data_vars[k] = trials_x_cells_x_time

        xrds_suite2p_trials = xr_helpers.make_xrds_trials(trial_data_vars,
                                                          stim,
                                                          cell_ids,
                                                          trial_ts, attrs=xrds_suite2p_traces.attrs)
        xrds_suite2p_trials.to_netcdf(stat_file.with_name('xrds_suite2p_trials.nc'))
        print(f"\t- saved {stat_file.relative_to(mov_dir).with_name('xrds_suite2p_trials.nc')}")
        print(f"done")

# xrds_file_list = list(NAS_PROC_DIR.rglob("xrds_suite2p_outputs_xid0.nc"))
# %%
# #%%
# for flacq in flacqs_to_process:
#     MOV_DIR = pathparse.flacq2dir(flacq)
#
#     # load thorimage metadata
#     meta = utils2p.Metadata(MOV_DIR.joinpath('Experiment.xml'))
#
#     # load frame and trial timing info
#     timestamps = np.load(MOV_DIR.joinpath('timestamps.npy'), allow_pickle=True).item()
#
#     # load stimulus info
#     with open(MOV_DIR.joinpath('stim_list.json'), 'r') as f:
#         stim_list = json.load(f)
#
#     # add time and stim info to attrs
#     attrs = copy.deepcopy(flacq)
#     attrs['stack_times'] = timestamps['stack_times']
#     attrs['olf_ict'] = timestamps['olf_ict']
#     attrs['stim'] = stim_list['stim_list_flatstr']
#
#     # load suite2p outputs
#     stat_file = NAS_PROC_DIR.joinpath(*flacq['s2p_stat_file'].split('/'))
#     Fall = loadmat(stat_file.with_name('Fall.mat'), squeeze_me=True)
#     iscell, cellprob = Fall['iscell'].T
#     iscell = iscell.astype('int')
#
#     # process traces
#     Fc = Fall['F'] - 0.7 * Fall['Fneu']
#     trace_dict = process_traces(Fc)
#     trace_dict['Fc'] = Fc
#
#     # make xarray dataset
#     xrds_traces = xr_helpers.make_xrds_traces(trace_dict,
#                                               cell_ids=range(Fc.shape[0]),
#                                               ts=timestamps['stack_times'],
#                                               attrs=attrs)
#
#     xrds_traces = xrds_traces.assign_coords(dict(cellprob=('cell_ids', cellprob)))
#     xrds_traces = xrds_traces.assign_coords(dict(iscell=('cell_ids', iscell)))
#
#     # save xarray dataset
#     xrds_traces.to_netcdf(stat_file.with_name('xrds_s2p_traces.nc'))
#
#     # save trial tensor xarray dataset
#     xrds_trials = traces_2_trial_tensors(xrds_traces)
#     xrds_trials.to_netcdf(stat_file.with_name('xrds_s2p_trials.nc'))

# %%
# with open('/local/storage/Remy/natural_mixtures/reports/configs/control1_top2_ramps_stim_str_grid.json', 'r') as f:
#     ramps_stim_gridlist = json.load(f)
# # %%
# from pandas.api.types import CategoricalDtype
#
# heatmap_ord = \
#     ['pfo @ 0.0',
#      '2h @ -7.0',
#      '2h @ -6.0',
#      '2h @ -5.0',
#      '1o3ol @ -5.0',
#      '1o3ol @ -4.0',
#      '1o3ol @ -3.0',
#      '1o3ol @ -5.0, 2h @ -7.0',
#      '1o3ol @ -5.0, 2h @ -6.0',
#      '1o3ol @ -5.0, 2h @ -5.0',
#      '1o3ol @ -4.0, 2h @ -7.0',
#      '1o3ol @ -4.0, 2h @ -6.0',
#      '1o3ol @ -4.0, 2h @ -5.0',
#      '1o3ol @ -3.0, 2h @ -7.0',
#      '1o3ol @ -3.0, 2h @ -6.0',
#      '1o3ol @ -3.0, 2h @ -5.0']
#
# olfcat_ramps = CategoricalDtype(heatmap_ord, ordered=True)
# # %%
# report_data_dir = Path("/local/storage/Remy/natural_mixtures/report_data/xrds_s2p_traces")
# ds_trial_netcdfs = list(report_data_dir.rglob("*__xrds_s2p_trials.nc"))
#
# save_to_pdf = True
# if save_to_pdf:
#     pdf = PdfPages("/local/storage/Remy/natural_mixtures/report_data/"
#                    "xrds_s2p_traces/'corrmats_control_flies.pdf")
#
# for file in ds_trial_netcdfs:
#
#     with xr.open_dataset(file, engine='h5netcdf') as xrds_trials:
#         xrds_trials.sel(time=slice(1, 4)).mean(dim='time')
#         xrds_mean_peak = xrds_trials.sel(time=slice(1, 4)).mean(dim='time')
#         xrds_mean_baseline = xrds_trials.sel(time=slice(-5, 1)).mean(dim='time')
#         xrds_std_baseline = xrds_trials.sel(time=slice(-5, 1)).std(dim='time')
#         xrds_peak_amp = xrds_mean_peak - xrds_mean_baseline
#
#         # corrmat, individual trials
#         # ----------------------------------
#         fig1, axarr = plt.subplots(3, 3, figsize=(24, 24),
#                                    constrained_layout=True,
#                                    gridspec_kw=dict(
#                                        # height_ratios=hratio,
#                                        wspace=0.02,
#                                        hspace=0.02)
#                                    )
#
#         for da_key, ax in zip(xrds_peak_amp.data_vars.keys(), axarr.flat):
#             corrmat = 1 - distance.squareform(distance.pdist(xrds_peak_amp[da_key].to_numpy(), metric='correlation'))
#
#             if 'ramps' in file.name:
#                 df_corrmat = pd.DataFrame(corrmat,
#                                           index=pd.Categorical(xrds_peak_amp.coords['stim'].to_numpy(),
#                                                                categories=heatmap_ord),
#                                           columns=pd.Categorical(xrds_peak_amp.coords['stim'].to_numpy(),
#                                                                  categories=heatmap_ord),
#                                           )
#
#                 df_corrmat.sort_index(axis=0, inplace=True)
#                 df_corrmat.sort_values(by=0, axis=1, inplace=True)
#             else:
#                 df_corrmat = pd.DataFrame(corrmat,
#                                           index=xrds_peak_amp.coords['stim'].to_numpy(),
#                                           columns=xrds_peak_amp.coords['stim'].to_numpy())
#
#             sns.heatmap(df_corrmat, cmap='RdBu_r', ax=ax,
#                         square=True,
#                         vmin=-0.8, vmax=0.8,
#                         cbar_kws=dict(shrink=0.5))
#             ax.set_title(da_key, fontsize=12)
#         #fig1.suptitle(file.relative_to(report_data_dir), fontsize=14)
#         plt.show()
#
#         # corrmat, averaged across stimulus presentations
#         # --------------------------------------------------
#         fig2, axarr = plt.subplots(3, 3, figsize=(24, 24), constrained_layout=True)
#         xrds_peak_amp_grpmean = xrds_peak_amp.groupby('stim').mean(dim='trials')
#         for da_key, ax in zip(xrds_peak_amp_grpmean.data_vars.keys(), axarr.flat):
#             corrmat = 1 - distance.squareform(
#                 distance.pdist(xrds_peak_amp_grpmean[da_key].to_numpy(), metric='correlation'))
#             df_corrmat = pd.DataFrame(corrmat,
#                                       index=xrds_peak_amp_grpmean.coords['stim'].to_numpy(),
#                                       columns=xrds_peak_amp_grpmean.coords['stim'].to_numpy(), )
#
#             sns.heatmap(df_corrmat, cmap='RdBu_r', ax=ax,
#                         square=True,
#                         vmin=-0.8, vmax=0.8,
#                         cbar_kws=dict(shrink=0.5))
#             ax.set_title(da_key, fontsize=12)
#         #fig2.suptitle(file.relative_to(report_data_dir), fontsize=14)
#         plt.show()
#
#     if save_to_pdf:
#         pdf.savefig(fig1)
#         pdf.savefig(fig2)
#
# if save_to_pdf:
#     pdf.close()
# # %% tensor decomposition
# import tensorly as tl
# from tensorly.decomposition import parafac, non_negative_parafac_hals, non_negative_parafac
# from tensorly import metrics
# import tcautils
#
# xrds_mean_baseline = xrds_trials.sel(time=slice(-5, -1)).mean(dim='time')
# xrds_trials_zeroed = xrds_trials - xrds_mean_baseline
# da = xrds_trials['F_bc_zscored']
# # %%
# # factors: trials x cells x time
#
# X = da.to_numpy()
# rank = 15
#
# weights, factors = non_negative_parafac(X, rank=rank)
# M = tcautils.reconstruct(factors, rank)
# rec_error = np.mean((X - M) ** 2)
#
# rmse = metrics.RMSE(X, M)
# mse = metrics.MSE(X, M)
# # %%
# fig_tca, axarr = tcautils.plot_factors(factors, d=3)
# fig_tca.subplots_adjust(top=0.95)
# plt.suptitle(f'tensorly (non_negative_parafac) on F_bc_zscored\nmse={mse:.03f}, rmse={rmse:.03f}')
# plt.show()

# %%
