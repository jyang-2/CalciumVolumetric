"""
Load registered tiff stack results, and save as:
 - a tiff stack in ../combined/reg_stacks/reg_stack.tif, and
 - a (time, z, y, x) netcdf xr.DataArray in ../combined/reg_stacks/xrda_reg_stack.nc, and
 - a (trials, time, z, y, x) netcdf xr.DataArray in ../combined/reg_stacks/xrda_reg_stack_trials.nc w/ good planes only

Then, sum over good z-planes, and save to reg_stack_zsum.tif

"""
import json
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pydantic
import utils2p
import xarray as xr

from pydantic_models import FlatFlyAcquisitions
from s2p import suite2p_helpers

NAS_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")

#%%


def xrda_reg_stack_2_xrda_trial_stacks(xrda_file, trial_ts=None):
    """
    Split xrda_reg_stack into olf_ict centered trial movies.

    Args:
        xrda_file (Path): file path to .../suite2p/combined/xrda_reg_stack.nc
        trial_ts (): timestamps tor trial (centered around stimulus onset time, defined in attrs['olf_ict']

    Returns:
        da_trial (xr.DataArray): data array containing good planes only, with interpolated trials
    """
    if isinstance(xrda_file, str):
        xrda_file = Path(xrda_file)

    xrda_reg_stack = xr.open_dataarray(xrda_file)
    attrs = xrda_reg_stack.attrs

    # select good planes
    da = xrda_reg_stack.where(xrda_reg_stack.good_planes, drop=True)

    if trial_ts is None:
        trial_ts = np.arange(-5, 20, 0.5)

    trial_stacks = []

    # loop through trials and add interpolated (t, z, y, x) planes to `trial_stacks`
    for i, (stim_ict, stim) in enumerate(zip(xrda_reg_stack.attrs['olf_ict'],
                                             xrda_reg_stack.attrs['stim'])):

        da_trial = da.interp(time=trial_ts + stim_ict)
        trial_stacks.append(da_trial.to_numpy())

    # concatenate trial stacks
    cat_trial_stacks = np.stack(trial_stacks, axis=0).astype('float32')

    da_trial_stacks = xr.DataArray(
        data=cat_trial_stacks,
        dims=['trials', 'time', 'z', 'y', 'x'],
        coords=dict(
            trials=range(len(xrda_reg_stack.attrs['stim'])),
            time=trial_ts,
            z=da.z.to_numpy(),
            y=da.y.to_numpy(),
            x=da.x.to_numpy(),
            stim=('trials', xrda_reg_stack.attrs['stim'])
        ),
        attrs=attrs
    )
    return da_trial_stacks


# def xrda_reg_stack_trials_2_trial_tiffs(xrda_file):
#     """Load xrda_reg_stack_trials.nc, and write each trial to a tiff file in a subdirectory. """
#
#     if isinstance(xrda_file, str):
#         xrda_file = Path(xrda_file)
#
#     da = xr.load_dataarray(xrda_file)   # should have dims (trials, time, z, y, x)

def xrda_reg_stack_trials_2_trial_tiffs(xrda_trials):
    """ From xrda_reg_stack_trials.nc, creates olf_ict centered trial movies.

    Args:
        xrda_trials ():
        xrda_file (Path): file path to .../suite2p/combined/reg_stacks/xrda_reg_stack_trials.nc

    Returns:
        saved_tiff_files (List[Path]): list of saved trial tiffs
    """


def xrda_reg_stack_2_trial_tif(xrda_file, filename_base=None):
    """ From xrda_reg_stack.nc, creates olf_ict centered trial movies
    Args:
        xrda_file (Path): file path to .../suite2p/combined/reg_stacks/xrda_reg_stack.nc
        filename_base (str): prefix for trial tiff files saved to .../suite2p/combined/reg_stacks/raw_zsum_trial_tiffs

    Returns:
        saved_tiff_files (List[Path]): list of saved trial tiffs
    """
    if isinstance(xrda_file, str):
        xrda_file = Path(xrda_file)

    xrda_reg_stack = xr.load_dataarray(xrda_file)

    # select good planes
    da = xrda_reg_stack.where(xrda_reg_stack.good_planes, drop=True)
    (t, z, y, x) = da.shape

    # sum planes
    da_zsum = da.sum(dim='z')
    da_zsum_b0_mean = da_zsum.sel()

    # iterate through trials
    saved_tiff_files = []

    # time interval around olf_ict
    trial_ts = np.arange(-5, 20, 0.5)

    trial_tiff_dir = xrda_file.with_name('raw_zsum_trial_tiffs')
    trial_tiff_dir.mkdir(parents=True, exist_ok=True)

    #trial_stacks = []

    # loop through trials
    print('saving z-summed trial tiffs:')
    for i, (stim_ict, stim) in enumerate(zip(xrda_reg_stack.attrs['olf_ict'],
                                             xrda_reg_stack.attrs['stim'])):
        stim_str = stim.replace(' ', '_').replace('@', 'at')

        # save z-projected trial
        if filename_base is None:
            save_name = f"raw_zsum__trial_{i:03d}__{stim_str}.tif"
        else:
            save_name = f"{flacq.filename_base()}__raw_zsum__trial_{i:03d}__{stim_str}.tif"

        da_trial = da_zsum.interp(time=trial_ts + stim_ict)
        save_file = trial_tiff_dir.joinpath(save_name)
        utils2p.save_img(save_file,
                         da_trial.to_numpy().astype('int16'))

        #trial_stacks.append(da_trial.to_numpy())
        saved_tiff_files.append(save_file)
        print(f"\t{save_file}")

    return saved_tiff_files


def write_suite2p_registration_results(stat_file):
    """ Loads suite2p registration results, and saves to **/combined/reg_stacks.

    Args:
        stat_file (Path): file path to .../suite2p/combined/stat.npy

    Returns:
        reg_stack_file (Path): path to 4d tiff hyperstack (made of registered suite2p tiffs)
        da_file (Path):
        """
    s2p_dir = suite2p_helpers.get_suite2p_folder(stat_file)

    # load suite2p xarray (for attrs)
    xrds_suite2p_outputs = xr.load_dataset(stat_file.with_name('xrds_suite2p_outputs.nc'))
    attrs = xrds_suite2p_outputs.attrs

    # load registered movies
    reg_stack = suite2p_helpers.load_bin_files_from_suite2p(s2p_dir)
    (t, z, y, x) = reg_stack.shape

    # make directory for suite2p-registered tiff stack: save_dir
    save_dir = s2p_dir.joinpath('combined', 'reg_stacks')
    save_dir.mkdir(parents=True, exist_ok=True)

    ##########################################
    # save registered stack to reg_stack_file
    ##########################################
    reg_stack_file = save_dir.joinpath('reg_stack.tif')
    utils2p.save_img(reg_stack_file, reg_stack)
    print(f'tiff saved to {reg_stack_file}')

    # add location of the registered tiff stack to attrs
    attrs['reg_stack_file'] = str(reg_stack_file.relative_to(NAS_PROC_DIR))

    # load ops file
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()
    good_planes = [item not in ops['ignore_flyback'] for item in range(reg_stack.shape[1])]

    ###############
    # xr.DataArray
    ###############

    # make dataarray from image data - use da.where(da.good_planes drop=True)
    da = xr.DataArray(
        data=reg_stack,
        dims=['time', 'z', 'y', 'x'],
        coords=dict(
            time=attrs['stack_times'],
            z=range(z),
            y=range(y),
            x=range(x),
            good_planes=('z', good_planes)
        ),
        attrs=attrs
    )

    # save data
    da_file = save_dir.joinpath('xrda_reg_stack.nc')
    da.to_netcdf(da_file)
    print(f'xr.DataArray saved to {da_file}')

    return reg_stack_file, da_file




#
# def copy_trial_tiffs_to_report_data(flat_acq):
#     trial_tiff_dir = flat_acq.s2p_stat_file.with_name('reg_stacks').joinpath('')
#     save_file = f"{flat_acq.filename_base()}__{}"



# %%
if __name__ == '__main__':
    # load flat linked acquisitions from manifestos
    with open("/local/storage/Remy/natural_mixtures/manifestos/flat_linked_thor_acquisitions.json", 'r') as f:
        flat_acqs = json.load(f)

    flat_acqs = pydantic.parse_obj_as(List[FlatFlyAcquisitions], flat_acqs)

    for flacq in filter(lambda x: x.movie_type is not None, flat_acqs):
        print('---')
        print(f"\n{flacq}")

        stat_file = NAS_PROC_DIR.joinpath(flacq.s2p_stat_file)

        ###################################################################
        # convert split tiff stacks to 4D tiff hyperstack, and xr.DataArray
        ###################################################################
        if not stat_file.with_name('reg_stacks').joinpath('xrda_reg_stack.nc').exists():
            print('xrda_reg_stack.nc file does not exist.')
            reg_tiff_file, xrda_img_file = write_suite2p_registration_results(stat_file)
        else:
            xrda_img_file = stat_file.with_name('reg_stacks').joinpath('xrda_reg_stack.nc')
            print('xrda_reg_stack.nc file already exists.')

        ###################################################################
        # convert xrda_img_file to xr.DataArray w/ dims (trials, time, z, y, x)
        ###################################################################
        if not xrda_img_file.with_name('xrda_reg_stack_trials.nc').exists():
            print('xrda_reg_stack_trials.nc does not exist.')
            da_trials = xrda_reg_stack_2_xrda_trial_stacks(xrda_img_file)
            da_trials.to_netcdf(xrda_img_file.with_name('xrda_reg_stack_trials.nc'))
            print(f"{xrda_img_file.with_name('xrda_reg_stack_trials.nc')} saved")

        else:
            print('xrda_reg_stack_trials.nc already exists.')

        ############################
        # make z-summed trial tiffs
        ############################
        if not stat_file.with_name('reg_stacks').joinpath('raw_zsum_trial_tiffs').is_dir():
            saved_trial_tiffs = xrda_reg_stack_2_trial_tif(xrda_img_file, filename_base=flacq.filename_base())
            for item in saved_trial_tiffs:
                print(f"\tsaved {item.name}")
        else:
            report_trial_tiff_dir = NAS_PROC_DIR.with_name('report_data').joinpath('trial_tiffs')
            src_dir = stat_file.with_name('reg_stacks').joinpath('raw_zsum_trial_tiffs')
            dest_dir = report_trial_tiff_dir.joinpath(f"{flacq.filename_base()}__raw_zsum_trial_tiffs")

            if not dest_dir.is_dir():
                shutil.copytree(src_dir, dest_dir)
                print(f"files copied to {dest_dir}")

        # convert xrda_reg_stack.nc to xrda_reg_stack_trials.nc



