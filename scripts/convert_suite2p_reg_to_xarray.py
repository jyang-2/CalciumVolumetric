import copy
from pathlib import Path

import numpy as np
import tifffile
import xarray as xr
from scipy.ndimage import gaussian_filter

import natmixconfig
# from PyPDF2 import PdfFileMerger
from pydantic_models import FlatFlyAcquisitions
from s2p import suite2p_helpers, s2p_to_xarray


# from matplotlib.backends.backend_pdf import PdfPages


# %%


# image_labels = [f'{i}' for i in range(volume.shape[0] * volume.shape[1])]
# imwrite(
#      'temp.tif',
#      volume,
#      imagej=True,
#      resolution=(1./2.6755, 1./2.6755),
#      metadata={
#          'spacing': 3.947368,
#          'unit': 'um',
#          'finterval': 1/10,
#          'fps': 10.0,
#          'axes': 'TZYX',
#          'Labels': image_labels,
#      }
#  )

def write_dff_trial_images(ds_dffs, out_dir=None):
    """Save individual trial dff images to tiff files in .../combined/dff_images/trials"""
    attrs = ds_dffs.attrs
    # make output directory named dff_images
    if out_dir is None:
        stat_file = natmixconfig.NAS_PROC_DIR / Path(ds_dffs.attrs['s2p_stat_file'])
        out_dir = stat_file.with_name('dff_images')

    trial_tiff_dir = out_dir.joinpath('trials')
    trial_tiff_dir.mkdir(parents=True, exist_ok=True)

    n_trials = ds_dffs.dims['trials']
    stim_list = ds_dffs.stim.to_numpy().tolist()
    stim_str_list = [item.replace(' ', '_').replace('@', 'at') for item in stim_list]

    for k, da in ds_dffs.items():
        folder = trial_tiff_dir.joinpath(k)
        folder.mkdir(exist_ok=True)
        print(f'\t- saving {k} trial images to tiff stacks.')

        for itrial, stim in enumerate(stim_str_list):
            # tiff string : {image type}__trial{trial number}__{stimulus}.tif
            filename = f"{k}__trial{itrial:02d}__{stim}.tif"

            # get trial array
            img_arr = da.isel(trials=itrial).to_numpy()
            tifffile.imwrite(folder.joinpath(filename),
                             img_arr,
                             imagej=True,
                             metadata=dict(axes='ZYX')
                             )

    return trial_tiff_dir


def write_dff_images_to_tiff_hyperstack(ds_dffs, out_dir=None, filename_prefix=None):
    attrs = ds_dffs.attrs

    # make output directory named dff_images
    if out_dir is None:
        stat_file = natmixconfig.NAS_PROC_DIR / Path(ds_dffs.attrs['s2p_stat_file'])
        out_dir = stat_file.with_name('dff_images')
    out_dir.mkdir(exist_ok=True)

    if filename_prefix is None:
        filename_prefix = f"{attrs['date_imaged']}__{attrs['fly_num']}__{attrs['thorimage']}"

    print('\n---saving dff images to tiff hyperstacks---')
    print(out_dir.as_uri())

    # save dff images to tiff hyperstacks
    for data_var, da in ds_dffs.data_vars.items():
        filename = f"{filename_prefix}__{data_var}.tif"
        img_arr = da.to_numpy()

        print(f'- saving {filename}')
        tifffile.imwrite(out_dir.joinpath(filename),
                         img_arr,
                         imagej=True,
                         metadata=dict(axes='TZYX')
                         )
    print('done!')
    return out_dir


def compute_mean_trial_image(da_reg, stim_ict, baseline_win=(-5.0, -0.5), peak_win=(0.05, 3.0),
                             # sigma=1.0, smooth=True,
                             # clip_quantiles=True, min_clip_quantile=0.05, max_clip_quantile=0.95
                             ):
    """Compute baseline_mean, baseline_std, and peak_mean for a single trial."""

    t, z, y, x = da_reg.shape

    # make time windows stimulus-centered
    stim_b0_win = [stim_ict + x for x in baseline_win]
    # print(stim_b0_win)
    stim_peak_win = [stim_ict + x for x in peak_win]
    # print(stim_peak_win)

    # da_baseline = da_reg.sel(time=slice(*stim_b0_win)).clip()
    # da_peak = da_reg.sel(time=slice(*stim_peak_win))

    # compute baseline images
    baseline_mean = da_reg.sel(time=slice(*stim_b0_win)).mean(dim='time').astype('float32')
    baseline_std = da_reg.sel(time=slice(*stim_b0_win)).std(dim='time').astype('float32')

    # compute mean images
    peak_mean = da_reg.sel(time=slice(*stim_peak_win)).mean(dim='time').astype('float32')

    # # compute df images
    # peak_df = (peak_mean - baseline_mean)
    # peak_dff = (peak_mean - baseline_mean) / baseline_std
    #
    # # smooth images
    # if smooth:
    #     gaussian_sigma = [0, 0, sigma, sigma]
    #
    #     arr_baseline_mean_smoothed = gaussian_filter(baseline_mean.to_numpy(), sigma=gaussian_sigma)
    #     arr_peak_mean_smoothed = gaussian_filter(peak_mean.to_numpy(), sigma=gaussian_sigma)
    #     baseline_mean = xr.DataArray(arr_baseline_mean_smoothed,
    #                                  dims=['trial', 'z', 'y', 'x'])
    #     peak_mean = xr.DataArray(arr_peak_mean_smoothed, dims=['trial', 'z', 'y', 'x'])
    #
    # # construct ds_dff
    # data_vars = dict(peak_dff=peak_dff,
    #                  peak_df=peak_df,
    #                  baseline_mean=baseline_mean,
    #                  baseline_std=baseline_std,
    #                  peak_mean=peak_mean,
    #                  )
    # construct ds_trial_images
    data_vars = dict(baseline_mean=baseline_mean,
                     baseline_std=baseline_std,
                     peak_mean=peak_mean,
                     )

    ds_dff_images = xr.Dataset(data_vars=data_vars, )

    return ds_dff_images


def compute_mean_trial_images(da_reg, olf_ict=None, stim_list=None,
                              baseline_win=(-5.0, -0.5), peak_win=(0.05, 3.0)):
    """Compute baseline_mean, baseline_std, and peak_mean for all trials trial."""
    if olf_ict is None:
        olf_ict = da_reg.attrs['olf_ict']
    if stim_list is None:
        stim_list = da_reg.attrs['stim']

    ds_dff_list = [compute_mean_trial_image(da_reg,
                                            stim_ict=ict,
                                            baseline_win=baseline_win,
                                            peak_win=peak_win,
                                            ) for ict in olf_ict]

    ds_dffs = xr.concat(ds_dff_list, dim='trials')
    ds_dffs = ds_dffs.assign_attrs(da_reg.attrs)
    ds_dffs = ds_dffs.assign_coords(
            trials=np.arange(len(olf_ict)),
            stim=('trials', stim_list))
    return ds_dffs


def compute_dff_images(ds_trial_images, sigma_baseline=0.5, sigma_peak=0.5,
                       smooth_baseline=True, smooth_peak=False, ):
    """Compute df and dff images from baseline_mean, baseline_std, and peak_mean images."""

    # smooth baseline images
    unsmoothed_baseline_mean = copy.deepcopy(ds_trial_images.baseline_mean)
    unsmoothed_peak_mean = copy.deepcopy(ds_trial_images.peak_mean)

    if smooth_baseline:
        arr_baseline_mean_smoothed = gaussian_filter(ds_trial_images.baseline_mean.to_numpy(),
                                                     sigma=[0, 0, sigma_baseline, sigma_baseline])
        baseline_mean = xr.DataArray(arr_baseline_mean_smoothed,
                                     dims=['trials', 'z', 'y', 'x'])
    else:
        baseline_mean = ds_trial_images.baseline_mean

    # smooth peak images
    if smooth_peak:
        arr_peak_mean_smoothed = gaussian_filter(ds_trial_images.peak_mean.to_numpy(),
                                                 sigma=[0, 0, sigma_peak, sigma_peak])
        peak_mean = xr.DataArray(arr_peak_mean_smoothed, dims=['trials', 'z', 'y', 'x'])

    else:
        peak_mean = ds_trial_images.peak_mean

    # compute df images
    peak_df = peak_mean - baseline_mean
    peak_dff = peak_df / baseline_mean

    # construct ds_dff
    data_vars = dict(peak_dff=peak_dff,
                     peak_df=peak_df,
                     baseline_mean=baseline_mean,
                     baseline_std=ds_trial_images.baseline_std,
                     peak_mean=peak_mean,
                     unsmoothed_baseline_mean=unsmoothed_baseline_mean,
                     unsmoothed_peak_mean=unsmoothed_peak_mean
                     )
    ds_dffs = xr.Dataset(data_vars=data_vars, )
    ds_dffs = ds_dffs.assign_attrs(copy.deepcopy(ds_trial_images.attrs))

    # ds_dffs = ds_dffs.assign_coords(
    #         trials=np.arange(ds_dffs.dims['trial']),
    #         stim=('trials', ds_trial_images.stim))
    return ds_dffs


# %%
#     olf_ict = da_reg.attrs['olf_ict']
#     n_trials = len(olf_ict)
#
#     b0_wins = [(ict + x for x in baseline_win) for ict in olf_ict]
#     peak_wins = [(ict + x for x in peak_win) for ict in olf_ict]
#
#     baseline_mean_list = [da_reg.sel(time=slice(*win)).mean(dim='time') for win in b0_wins]
#     baseline_std_list = [da_reg.sel(time=slice(*win)).std(dim='time') for win in b0_wins]
#     peak_mean = [da_reg.sel(time=slice(*win)).mean(dim='time') for win in peak_wins]
#     peak_dff =
#
#     _, z, y, x = da_reg.shape
#
#     attrs = copy.deepcopy(da_reg.attrs)
#
#     data_vars = dict(peak_dff=peak_dff,
#                      baseline_mean=baseline_mean,
#                      baseline_std=baseline_std,
#                      peak_mean=peak_mean,
#                      )
#     ds_dff = xr.Dataset(data_vars=data_vars,
#                         dims=['trial', 'z', 'y', 'x'],
#                         )
#     return peak_dff


def load_registered_suite2p_movie_3d(stat_file):
    """Load registered suite2p movie from combined/stat.npy as an xarray DataArray

    Args:
        stat_file (pathlib.Path): path to stat.npy file

    Returns:
        da_reg (xarray.DataArray): registered movie xr.DataArray w/ dims ('time, 'z', 'y', 'x')
    """
    reg_stack = suite2p_helpers.load_combined_reg_tiffs(stat_file, drop_flyback=False)
    t, z, y, x = reg_stack.shape
    da_reg = xr.DataArray(reg_stack//2,
                          dims=['time', 'z', 'y', 'x'],
                          attrs=dict(
                                  stat_file=str(stat_file)
                          )
                          )
    da_reg = da_reg.clip(min=0, keep_attrs=True) + 1
    da_reg = da_reg.astype('float32')
    return da_reg


def load_registered_suite2p_movie_2d(stat_file):
    """Load registered suite2p movie from plane*/stat.npy"""
    bin_file = stat_file.with_name('data.bin')
    reg_stack = suite2p_helpers.load_plane_from_bin(bin_file)
    t, y, x = reg_stack.shape
    da_reg = xr.DataArray(reg_stack,
                          dims=['time', 'y', 'x'],
                          )
    return da_reg


def load_registered_suite2p_movie_as_xarray_from_flacq(flat_acq: FlatFlyAcquisitions):
    """Load registered suite2p movie from plane*/stat.npy, and add metadata from flat_acq

    Args:
        flat_acq (FlatFlyAcquisitions): flat acq object

    Returns:
        da_flacq_reg (xarray.DataArray): registered xr.DataArray movie, w/ flat_acq attrs
    """
    MOV_DIR = flat_acq.mov_dir()
    stat_file = natmixconfig.NAS_PROC_DIR / flat_acq.stat_file()

    nonflyback_planes = suite2p_helpers.get_nonflyback(stat_file)

    # get flat_acq metadata
    attrs = s2p_to_xarray.flacq_2_xarray_attrs(flat_acq)

    # load registered movie
    da_reg = load_registered_suite2p_movie_3d(stat_file)

    t, z, y, x = da_reg.shape

    # add dimension info
    da_flacq_reg = da_reg.assign_coords(
            time=attrs['stack_times'],
            z=np.arange(z),
            y=np.arange(y),
            x=np.arange(x),
    )

    da_flacq_reg = da_flacq_reg.sel(z=nonflyback_planes)
    da_flacq_reg = da_flacq_reg.assign_attrs(attrs)
    return da_flacq_reg


def clip_movie_by_quantiles(da_reg, q=(0.01, 0.995)):
    """Clip movie by quantiles"""
    q0, q1 = da_reg.quantile(q=q)
    da_reg_clip = da_reg.clip(q0, q1)
    da_reg_clip = da_reg_clip.assign_attrs(copy.deepcopy(da_reg.attrs))
    return da_reg_clip


def main(flat_acq, baseline_win=(-5.0, -0.5), peak_win=(0.05, 3.0), save_netcdf=False):
    da_reg = load_registered_suite2p_movie_as_xarray_from_flacq(flat_acq)

    # da_reg_clip = clip_movie_by_quantiles(da_reg, q=(0.01, 0.995))
    ds_trial_images = compute_mean_trial_images(da_reg,
                                                baseline_win=baseline_win,
                                                peak_win=peak_win)
    ds_dffs = compute_dff_images(ds_trial_images,
                                 sigma_baseline=1,
                                 sigma_peak=0.5,
                                 smooth_baseline=True,
                                 smooth_peak=False)

    dff_images_dir = write_dff_images_to_tiff_hyperstack(ds_dffs)
    dff_trial_images_dir = write_dff_trial_images(ds_dffs)

    if save_netcdf:
        ds_trial_images.to_netcdf(dff_images_dir.joinpath('trial_images.nc'))
        ds_dffs.to_netcdf(dff_images_dir.joinpath('dff_images.nc'))

    return dff_images_dir


def main_b(flat_acq, save_netcdf=False):
    MOV_DIR = flat_acq.mov_dir()
    stat_file = natmixconfig.NAS_PROC_DIR / flat_acq.stat_file()
    ops_file = stat_file.with_name('ops.npy')

    attrs = s2p_to_xarray.flacq_2_xarray_attrs(flat_acq)

    if stat_file.parent.name == 'combined':
        reg_stack = suite2p_helpers.load_reg_stack_from_combined_ops(ops_file)
        t, z, y, x = reg_stack.shape
        da_reg = xr.DataArray(reg_stack,
                              dims=['time', 'z', 'y', 'x'],
                              coords=dict(
                                      time=attrs['stack_times'],
                                      z=np.arange(z),
                                      y=np.arange(y),
                                      x=np.arange(x),
                              ),
                              attrs=attrs
                              )
        filename = stat_file.with_name('reg_combined.nc')

    elif 'plane' in stat_file.parent.name:
        bin_file = stat_file.with_name('data.bin')
        reg_stack = suite2p_helpers.load_plane_from_bin(bin_file)
        t, y, x = reg_stack.shape
        da_reg = xr.DataArray(reg_stack,
                              dims=['time', 'y', 'x'],
                              coords=dict(
                                      time=attrs['stack_times'],
                                      y=np.arange(y),
                                      x=np.arange(x),
                              ),
                              attrs=attrs
                              )
        filename = stat_file.with_name('reg_plane.nc')

    if save_netcdf:
        da_reg.to_netcdf(filename)

    return da_reg
