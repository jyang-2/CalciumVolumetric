import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

import natmixconfig
# import suite2p
# import rastermap
# from rastermap.mapping import Rastermap
from s2p import rasterplot


#%%
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
    print(f"\n\tloading: {stat_file.with_name('xrds_suite2p_outputs.nc')}")
    ds_suite2p_outputs = xr.load_dataset(stat_file.with_name('xrds_suite2p_outputs.nc'))

    # win = 90.0 # seconds
    # fps = 3.0

    # add smoothed, zscored traces to dataset

    # Fc = ds_suite2p_outputs['Fc'].to_numpy()

    # F0 = percentile_filter(Fc_smooth, size=win*fps, percentile=5)
    # Fc_bc = Fc - F0
    # print('F0 computed.')

    #############################
    # process fluorescences
    #############################
    Fc_smooth = gaussian_filter1d(ds_suite2p_outputs['Fc'].to_numpy(), sigma=1)
    Fc_zscore = zscore(ds_suite2p_outputs['Fc'].to_numpy(), axis=-1)
    Fc_zscore_smoothed = gaussian_filter1d(Fc_zscore, sigma=1)
    Fc_normed = rasterplot.norm_traces(ds_suite2p_outputs['Fc'].to_numpy())
    Fc_normed_smoothed = gaussian_filter1d(Fc_normed, sigma=1)

    ds_suite2p_outputs['Fc_smooth'] = (('cells', 'time'), Fc_smooth)
    ds_suite2p_outputs['Fc_zscore'] = (('cells', 'time'), Fc_zscore)
    ds_suite2p_outputs['Fc_zscore_smoothed'] = (('cells', 'time'), Fc_zscore_smoothed)
    ds_suite2p_outputs['Fc_normed'] = (('cells', 'time'), Fc_normed)
    ds_suite2p_outputs['Fc_normed_smoothed'] = (('cells', 'time'), Fc_normed_smoothed)

    #######################################
    # add initial rastermap embedding info
    #######################################
    embedding_allcells = np.load(stat_file.with_name('embedding_allcells.npy'),
                                 allow_pickle=True).item()
    ds_suite2p_outputs = ds_suite2p_outputs.assign_coords(
            dict(xid0=('cells', embedding_allcells['model'].xid)))
    ds_suite2p_outputs = ds_suite2p_outputs.assign_coords(
            dict(embedding0=('cells', embedding_allcells[
                'model'].embedding.squeeze())))
    ds_suite2p_outputs.attrs['good_xid'] = embedding_allcells['good_xid']
    print(ds_suite2p_outputs)

    save_file = stat_file.with_name('xrds_suite2p_outputs_xid0.nc')
    print(f"\n\tsave to: {save_file}")
    ds_suite2p_outputs.to_netcdf(save_file)
    print(f"\n\tdone!")

    goodxid_pdf_file = rasterplot.plot_split_rastermap_goodxids_only(save_file)
    return ds_suite2p_outputs
