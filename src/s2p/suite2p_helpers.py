"""
Useful functions for processing suite2p outputs.

Functions involving suite2p rois were copied & slightly altered from the
`suite2p.extraction.masks` module (see https://suite2p.readthedocs.io )
"""
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import parse
# import suite2p
# from suite2p.extraction.masks import create_masks
import utils2p

scalar_stat_keys = ['footprint', 'mrs', 'mrs0', 'compact', 'solidity', 'npix', 'npix_soma', 'radius', 'aspect_ratio',
                    'npix_norm_no_crop', 'npix_norm', 'skew', 'std', 'iplane']

nonscalar_stat_keys = ['ypix', 'xpix', 'lam', 'med', 'soma_crop', 'overlap', 'neuropil_mask']


def scalar_stats_to_dataframe(stats):
    """
    Convert cell statistics in stats to a pd.DataFrame (scalar fields only)

    Args:
        stats (List[dict]):

    Return:
        df (pd.DataFrame): table w/ one row per cell, and columns = scalar_stat_keys.
                            (also added are columns 'cell_id' and 'label'.
    """
    scalar_stats = [{k: v for k, v in stat.items() if k in scalar_stat_keys} for stat in stats]
    df = pd.DataFrame(scalar_stats)
    return df


def stats_to_df_com(stats, Lx, Ly, Lz, allow_overlap=False):
    """Compute center of mass (com) for stat in 'stats.npy'. """

    com_list = [get_com_3d(item, Lx, Ly, Lz,
                           allow_overlap=allow_overlap)
                for item in stats]
    df_com = pd.DataFrame(com_list, columns=['com_z', 'com_y', 'com_x'])
    return df_com


def stats_to_df_med(stats, Lx, Ly, allow_overlap=False):
    med_list = [get_med_3d(stat, Lx, Ly) for stat in stats]
    df_med = pd.DataFrame(med_list, columns=['med_z', 'med_y', 'med_x'])
    return df_med


# %%
def get_suite2p_folder(file):
    """ Returns base suite2p directory in file path

    Args:
        file (Union[str, Path]): filepath to anything in suite2p directory

    Returns:
        (Path): Path to top 'suite2p' folder.
    """
    if isinstance(file, str):
        file = Path(file)

    for folder in file.parents:
        if folder.name == 'suite2p':
            return folder
    return None


def path_to_plane(file):
    """ Extracts plane number from file path containing **/suite2p/plane{int}/...

    Args:
        file (Union[str, Path]): Path like "/local/storage/Remy/natural_mixtures/processed_data/
                                    2022-02-11/3/kiwi_ea_eb_only/downsampled_3/suite2p/plane6"

    Returns:
        int: plane # in file path
    """
    fpath = "{parent}/suite2p/plane{plane:d}/{file}"
    r = parse.search(fpath, str(file))
    plane_idx = r['plane']
    return plane_idx


def load_reg_stack_from_combined_ops(ops_file):
    """ Loads registered movie from the plane folders in /suite2p """

    ops = np.load(ops_file, allow_pickle=True).item()
    #s2p_path = suite2p.io.utils.get_suite2p_path(ops_file)
    s2p_path = get_suite2p_folder(ops_file)
    bin_files = list(s2p_path.rglob("plane*/*.bin"))
    bin_files.sort(key=lambda x: path_to_plane(x))
    stack = np.stack([load_plane_from_bin(item).data for item in bin_files])
    return stack


def load_plane_from_bin(bin_file):
    """ Loads registered binary from file, and returns np.ndarray. """
    ops = np.load(bin_file.with_name('ops.npy'), allow_pickle=True).item()
    # img = suite2p.io.BinaryFile(Lx=ops['Lx'], Ly=ops['Ly'], read_filename=bin_file)

    with open(bin_file, mode='rb') as f:
        img = np.fromfile(f, np.int16).reshape(-1, ops['Ly'], ops['Lx'])

    return img


def load_bin_files_from_suite2p(s2p_path):
    """ Load and combine all registered .bin files from suite2p"""
    bin_files = list(s2p_path.rglob("plane*/*.bin"))
    bin_files.sort(key=lambda x: path_to_plane(x))
    reg_stack = np.stack([load_plane_from_bin(item).data for item in bin_files], axis=1)

    return reg_stack


def get_n_pixels_per_cell(stat_file):
    stat = np.load(stat_file, allow_pickle=True)
    pix_per_cell = np.zeros(stat.size)
    for n in range(stat.size):
        pix_per_cell[n] = stat[n]['ypix'][~stat[n]['overlap']].size
    return pix_per_cell


def overwrite_iscell(iscell_file, iscell_new):
    iscell = np.load(iscell_file, allow_pickle=True)
    n_cells = iscell.shape[0]
    if iscell_new.dtype == np.bool_:
        iscell_new = iscell_new * 1.0
    iscell_to_save = np.zeros_like(iscell)
    iscell_to_save[:, 0] = iscell_new
    np.save(iscell_file, iscell_to_save)
    return iscell_file


# %%

def get_cell_mask_2d(stat, Lx, Ly, allow_overlap=False):
    mask = ... if allow_overlap else ~stat['overlap']
    ypix = np.remainder(stat['ypix'], Ly)
    xpix = np.remainder(stat['xpix'], Lx)
    cell_mask = np.ravel_multi_index((ypix, xpix), (Ly, Lx))
    cell_mask = cell_mask[mask]
    lam = stat['lam'][mask]
    lam_normed = lam / lam.sum() if lam.size > 0 else np.empty(0)
    return cell_mask, lam_normed


# %%
def get_mask_px_3d(stat, Lx, Ly, Lz, allow_overlap=False):
    mask = ... if allow_overlap else ~stat['overlap']
    ypix = np.remainder(stat['ypix'][mask], Ly)
    xpix = np.remainder(stat['xpix'][mask], Lx)
    zpix = np.ones_like(xpix) * stat['iplane']

    lam = stat['lam'][mask]
    lam_normed = lam / lam.sum() if lam.size > 0 else np.empty(0)

    return zpix, ypix, xpix, lam_normed


def get_cell_mask_3d(stat, Lx, Ly, Lz, allow_overlap=False):
    """
    Creates cell masks for rois in stat.npy
    Copied & slightly altered from suite2p.extraction.masks.

    See suite2p.readthedocs.io for more details.

    Args:
        stat (dict):
        Lx (int):
        Ly (int):
        Lz (int):
        allow_overlap (bool=True):

    Returns:
        cell_mask (np.array): flat index into array of size (Lz, Ly, Lx)
        lam_normed (np.array): pixel weights of cell_masks (sum to 1)
    """
    if allow_overlap:
        mask = ...
    else:
        mask = ~stat['overlap']

    ypix = np.remainder(stat['ypix'], Ly) % Ly
    xpix = np.remainder(stat['xpix'], Lx) % Lx
    zpix = np.ones_like(xpix) * stat['iplane'] % Lz

    cell_mask = np.ravel_multi_index((zpix, ypix, xpix), (Lz, Ly, Lx))
    cell_mask = cell_mask[mask]
    lam = stat['lam'][mask]
    lam_normed = lam / lam.sum() if lam.size > 0 else np.empty(0)
    return cell_mask, lam_normed


def get_med_3d(stat, Lx, Ly):
    """Return cell mask median in (z, y, x) coordinates.

    Here, Ly and Lx are the original dimensions of the movie, most likely (256, 256).
    """
    med_y, med_x = stat['med']
    med_y = np.remainder(med_y, Ly)
    med_x = np.remainder(med_x, Ly)
    med_z = stat['iplane']
    return med_z, med_y, med_x


def get_com_3d(stat, Lx, Ly, Lz, allow_overlap=False):
    zpix, ypix, xpix, lam_normed = get_mask_px_3d(stat, Lx, Ly, Lz,
                                                  allow_overlap=allow_overlap)
    wc_x = np.dot(xpix, lam_normed)
    wc_y = np.dot(ypix, lam_normed)
    wc_z = stat['iplane']
    return wc_z, wc_y, wc_x


def get_mask_labels(stats, Lx, Ly, Lz, allow_overlap=False):
    """Return 2D or 3D label matrix. Each label corresponds to the nth+1 cell in `stat`. """

    if Lz == 1:
        label_img = np.zeros((Ly, Lx), dtype=np.uint16).ravel()
        for n_cell, stat in enumerate(stats):
            cell_mask, lam_normed = get_cell_mask_2d(stat, Lx, Ly, allow_overlap=allow_overlap)
            label_img[cell_mask] = n_cell + 1
    else:
        label_img = np.zeros((Lz, Ly, Lx), dtype=np.uint16).ravel()
        for n_cell, stat in enumerate(stats):
            cell_mask, lam_normed = get_cell_mask_3d(stat, Lx, Ly, Lz, allow_overlap=allow_overlap)
            label_img[cell_mask] = n_cell + 1
    label_img = label_img.reshape((Lz, Ly, Lx))
    return label_img


def parse_stats_to_tables(stats, Lx, Ly, Lz, allow_overlap=False):
    df_scalar_stats = scalar_stats_to_dataframe(stats)
    df_med = stats_to_df_med(stats, Lx, Ly, allow_overlap=allow_overlap)
    df_com = stats_to_df_com(stats, Lx, Ly, Lz, allow_overlap=allow_overlap)
    return df_scalar_stats, df_med, df_com


def add_extra_suite2p_outputs(stat_file):
    stats = np.load(stat_file, allow_pickle=True)
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True)


    Lx, Ly, Lz = (256, 256, 16)
    df_scalar_stats, df_med, df_com = parse_stats_to_tables(stats, Lx, Ly, Lz)
    limg = get_mask_labels(stats, Lx, Ly, Lz)

    # make subdirectory in same folder as stat.npy (probably will be suite2p/combined/cell_stats)
    cell_stat_dir = stat_file.with_name('cell_stats')
    cell_stat_dir.mkdir(exist_ok=True)

    # save cell stats as tables in `cell_stat_dir`
    df_scalar_stats.to_csv(cell_stat_dir.joinpath('df_scalar_stats.csv'), index=False)
    df_med.to_csv(cell_stat_dir.joinpath('df_med.csv'), index=False)
    df_com.to_csv(cell_stat_dir.joinpath('df_com.csv'), index=False)

    # save label image as tiff and .npy file
    np.save(cell_stat_dir.joinpath('label_img.npy'), limg)
    utils2p.save_img(cell_stat_dir.joinpath('label_img.tif'), limg)

    return cell_stat_dir


def get_spatial_footprints_2d(stat_file):
    """
    Return CNMF-style array of spatial components w/ dimensions (n_pix, n_cells), where each cell's
    pixel weightings are stored in a column.

    Args:
        stat_file (Path): path to `stat.npy`

    Returns:
        A (np.ndarray): spatial components, has dims = (n_pixels, n_cells)
    """
    stats = np.load(stat_file, allow_pickle=True)
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()

    # F = np.load(stat_file.with_name('F.npy'), allow_pickle=True)
    # n_cells, n_frames = F.shape
    n_cells = len(stats)

    Lx = ops['Lx']
    Ly = ops['Ly']

    A = np.zeros((Lx * Ly, n_cells))
    for i, stat in enumerate(stats):
        mask = ~stat['overlap']
        im = np.zeros((Lx, Ly))
        ypix = stat['ypix'][mask]
        xpix = stat['xpix'][mask]
        lam = stat['lam'][mask]
        print(lam.max())
#        lam = lam / lam.max()
        im[ypix, xpix] = lam
        A[:, i] = im.flatten()
    return A


def get_spatial_footprints_3d(stat_file, Lx=None, Ly=None, Lz=None):
    stats = np.load(stat_file, allow_pickle=True)
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()

    if Lz is None:
        Lz = ops['nplanes']
    if Ly is None or Ly is None:
        Ly, Lx = ops['refImg'].shape

    n_cells = len(stats)

    A = np.zeros((Lx*Ly*Lz, n_cells))
    for i, stat in enumerate(stats):
        cell_mask, lam_normed = get_cell_mask_3d(stat, Lx, Ly, Lz, allow_overlap=ops['allow_overlap'])
        a = np.zeros(Lx*Ly*Lz)
        a[cell_mask] = lam_normed
        A[:, i] = a
    return A


def spatial_footprint_2_label_img(A, Lx, Ly, Lz):
    A_label = (A > 0) * 1
    A_limg = A_label * (np.arange(0, A_label.shape[1]) + 1)
    A_limg = A_limg.sum(axis=1).reshape(Lz, Ly, Lx)
    return A_limg


def get_label_image(stat_file):
    stats = np.load(stat_file, allow_pickle=True)
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()
    Ly, Lx = ops['refImg'].shape

    # determine if single-plane or not
    if (Lx == ops['Lx']) and (Ly == ops['Ly']):
        Lz = 1
        print('single plane')
    else:
        Lz = ops['nplanes']
        print('multi plane')

    if Lz == 1:
        A = get_spatial_footprints_2d(stat_file)
    elif Lz > 1:
        A = get_spatial_footprints_3d(stat_file, Lx, Ly, Lz)
    label_image = spatial_footprint_2_label_img(A, Lx, Ly, Lz)
    return label_image
# %%


if __name__ == '__main__':
    import xarray as xr
    DATA_DIR = Path("/local/storage/Remy/for_dhruvs_paper")
    stat_files = sorted(list(DATA_DIR.rglob('8/**/stat.npy')))

    for stat_file in stat_files:
        stats = np.load(stat_file, allow_pickle=True)
        ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()

        F = np.load(stat_file.with_name('F.npy'))
        Ly = ops['Ly']
        Lx = ops['Lx']

        n_trials = 3
        n_cells, T = F.shape

        #################################
        # process F.npy --> df/f
        #################################
        F_trial = np.stack(np.split(F, 3, axis=1))
        da_F = xr.DataArray(F_trial,
                            dims=['trial', 'cell', 'time'],
                            coords=dict(trial=range(3),
                                        time=range(165),
                                        cell=range(n_cells)))
        da_baseline = da_F.sel(time=slice(20, 60)).mean(dim='time')
        da_dff = (da_F - da_baseline) / da_baseline

        arr_dff_trials = da_dff.to_numpy()
        arr_dff = np.concatenate(tuple(arr_dff_trials), axis=1)

        #################################################
        # save spatial footprints to tifs
        #################################################
        mov_dir = get_suite2p_folder(stat_file).parent

        limg = get_mask_labels(stats, Lx, Ly, 1)
        limg = limg.squeeze()
        utils2p.save_img(mov_dir.joinpath('suite2p_roi_labels.tif'), limg.astype('uint16'))

        A = get_spatial_footprints_2d(stat_file)
        lam_arr = A.reshape(Ly, Lx, A.shape[1]).T
        utils2p.save_img(mov_dir.joinpath('suite2p_roi_lam.tif'), lam_arr.astype('float32'))

        lam_maxnormed = (A/A.max(axis=0)).reshape((Ly, Lx, A.shape[1])).T
        utils2p.save_img(mov_dir.joinpath('suite2p_roi_lam_maxnormed.tif'), lam_maxnormed.astype('float32'))

        lam_sumnormed = (A/A.sum(axis=0)).reshape((Ly, Lx, A.shape[1])).T
        utils2p.save_img(mov_dir.joinpath('suite2p_roi_lam_sumnormed.tif'), lam_sumnormed.astype('float32'))
    #
    # A_maxnorm = A/A.max(axis=0)
    # denoised = (A_maxnorm @ C).T.reshape(495, Ly, Lx)
    # utils2p.save_img(stat_file.with_name('dff_recons_Amaxnorm_x_C.tif'), denoised.astype('float32'))
    #
    # denoised = (A @ F).T.reshape(495, Ly, Lx)
    # utils2p.save_img(stat_file.with_name('movie_reconstructed_A_x_F.tif'), denoised.astype('float32'))
    #
    #
        #######################################################
        # generate denoised movie from suite2p outputs
        ########################################################
        denoised = (A / A.max(axis=0)) @ F
        denoised = denoised.T.reshape(495, Ly, Lx)
        utils2p.save_img(mov_dir.joinpath('dff_recons_Amaxnormed_x_F.tif'), denoised.astype('float32'))

        C = arr_dff
        A_limg = A > 0
        denoised = (A_limg @ C).T.reshape(495, Ly, Lx)
        utils2p.save_img(mov_dir.joinpath('dff_recons_Apix_x_C.tif'), denoised.astype('float32'))
    #
        df_dff = pd.DataFrame(C.T)
        df_dff.to_csv(mov_dir.joinpath('suite2p_dff.csv'), index_label=False)
#%%


#
# A = suite2p_helpers.get_spatial_footprints_2d(stat_file)
# # plt.imshow(A.sum(axis=1).reshape(224, 224), cmap=gray)
# # plt.show()
# # %%
# denoised = (A @ C).T.reshape(495, Ly, Lx)
# utils2p.save_img(stat_file.with_name('dff_recons.tif'), denoised.astype('float32'))
#
# # %%
#f
# F = np.load(stat_file.with_name('F.npy'), allow_pickle=True)
# F_trial = np.stack(np.split(F, 3, axis=1))
#
# da_F = xr.DataArray(F_trial,
#                     dims=['trial', 'cell', 'time'],
#                     coords=dict(trial=range(3),
#                                 time=range(165),
#                                 cell=range(10)))
# da_baseline = da_F.sel(time=slice(20, 60)).mean(dim='time')
#
# norm_by_baseline = True
# if norm_by_baseline:
#     da_dff = (da_F - da_baseline) / da_baseline
# else:
#     da_dff = (da_F - da_baseline)
#
# dff_full = da_dff.to_numpy()
# C = np.concatenate(tuple(dff_full), axis=1)
#
# # %%
# import tensorly as tl
# from tensorly.decomposition import parafac
#
# tensor = tl.tensor(dff_full)
# weights, factors = parafac(tensor, rank=5)
# reconstruction = tl.cp_to_tensor((weights, factors))
# reconstruction_trace = np.concatenate(tuple(reconstruction), axis=1)
# # %%
# from scipy.ndimage import gaussian_filter1d
#
# fig, axarr = plt.subplots(10, 1, figsize=(8.5, 11))
# for i, ax in enumerate(axarr.flat):
#     y = dff_full_trace[i, :]
#     ax.plot(y, 'r-')
#     ys = gaussian_filter1d(y, 3)
#     ax.plot(ys, 'k-')
#
# plt.show()
#
# import tensorly as tl
