""" Split xrds_suite2p_outputs_xid0.nc into trial traces w/ dims """
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import natmixconfig
# import suite2p
# import rastermap
# from rastermap.mapping import Rastermap
from s2p import s2p_to_xarray

plt.rcParams.update({'pdf.fonttype': 42,
                     'text.usetex': False})


def main(flacq):
    mov_dir = flacq.mov_dir()
    print('---')
    print(flacq)
    print(f"\nMOV_DIR: {mov_dir}\n")

    stat_file = natmixconfig.NAS_PROC_DIR / flacq.stat_file()
    print(f"\n\tstat_file: {stat_file}")

    ####################################
    # load first xarray dataset
    ####################################
    xrds_file = stat_file.with_name('xrds_suite2p_outputs_xid0.nc')
    print(f"\n\tloading: {xrds_file}")

    xrds_suite2p_outputs_xid0 = xr.load_dataset(xrds_file)

    ###########################
    # split to trials
    ###########################
    xrds_suite2p_trials = s2p_to_xarray.xrds_suite2p_outputs_xid0_2_xrds_suite2p_output_trials_xid0(
        xrds_suite2p_outputs_xid0, np.arange(-5, 20, 0.25))

    # save trial xarray
    save_file = stat_file.with_name('xrds_suite2p_trials_xid0.nc')
    print(f"\n\tsave to: {save_file}")
    xrds_suite2p_trials.to_netcdf(save_file)
    print(f"\n\tdone!")

    return xrds_suite2p_trials

#%%


