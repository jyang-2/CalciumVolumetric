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
import suite2p
import tifffile
# from suite2p.extraction.masks import create_masks
import utils2p
from scipy import sparse

# from scipy.sparse import csc_array

scalar_stat_keys = ['footprint', 'mrs', 'mrs0', 'compact', 'solidity', 'npix', 'npix_soma',
                    'radius', 'aspect_ratio',
                    'npix_norm_no_crop', 'npix_norm', 'skew', 'std', 'iplane']

nonscalar_stat_keys = ['ypix', 'xpix', 'lam', 'med', 'soma_crop', 'overlap', 'neuropil_mask']


# %% useful functions for getting suite2p info and paths

def get_suite2p_folder(file):
    """ Returns base suite2p directory in file path

    Args:
        file (Union[str, Path]): filepath to anything in suite2p directory

    Returns:
        (Path): Path to top 'suite2p' folder.
    """
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


def is_3d(stat_file, method='filepath'):
    """Check if stat.npy file is from a 3D recording.

    Args:
        stat_file (Path): path to stat.npy file
        method (str): 'filepath' or 'iplane'.
                       If 'filepath', then check if 'parent folder name' is in the filepath.
                       If 'iplane', then load `stat_file` and check if 'iplane' is a key.
    """
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    if method == 'filepath':
        is_multiplane = not 'plane' in stat_file.parent.name
    elif method == 'iplane':
        # load stat.npy file into a n_cells x 1 array of dicts
        stat = np.load(stat_file, allow_pickle=True)
        is_multiplane = 'iplane' in stat[0].keys()
    return is_multiplane


def get_flyback(stat_file):
    """Get flyback frames from stat.npy file.

    Args:
        stat_file (Path): path to stat.npy file
    """
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()
    return ops['ignore_flyback']


def get_nonflyback(stat_file):
    """Get non-flyback frames from combined/stat.npy file.

    Args:
        stat_file (Path): path to stat.npy file
    """
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    flyback = get_flyback(stat_file)
    n_planes = get_nplanes(stat_file)
    nonflyback = set(range(n_planes)) - set(flyback)
    nonflyback = sorted(list(nonflyback))
    return nonflyback


def get_frames(stat_file):
    """Get # of timepoints in movie (ops['nframes']).

    Args:
        stat_file (Path): path to stat.npy file
    """
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()
    return ops['nframes']


def get_nplanes(stat_file):
    """Get # of planes in recording from stat.npy file.

    If stat.npy file is in a `plane**` folder from a 3D recording, returns `nplanes`, not 1.
    This differs from get_dims() which returns Lz = 1 for a single plane.

    Args:
        stat_file (Path): path to stat.npy file

    Returns:
        nplanes (int): # of planes in recording
    """
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()
    return ops['nplanes']


def get_dims(stat_file):
    """Get dimensions of recording from stat.npy file.

    Note: if stat.npy file is in a `plane**` folder from a 3D recording, dimensions returned are
    for the single plane (i.e. Lz = 1).
    """
    if is_3d(stat_file):
        Lx, Ly, Lz = get_3d_shape(stat_file)
    else:
        Lx, Ly = get_2d_shape(stat_file)
        Lz = 1

    return Lx, Ly, Lz


def get_3d_shape(stat_file):
    """
    Returns Lx, Ly, Lz for suite2p movie

    Args:
        stat_file (Path): file path to suite2p/combined/stat.npy

    Returns:
        Lx
        Ly
        Lz
    """

    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    s2p_dir = get_suite2p_folder(stat_file)
    plane_stat_files = list(s2p_dir.rglob('plane*/stat.npy'))
    plane_ops = np.load(plane_stat_files[0].with_name('ops.npy'), allow_pickle=True).item()

    Lx = plane_ops['Lx']
    Ly = plane_ops['Ly']
    Lz = plane_ops['nplanes']
    return Lx, Ly, Lz


def get_2d_shape(stat_file):
    """
    Get dimensions of imaging plane if suite2p movie is single plane.

    Args:
        stat_file (Path): file path to suite2p/plane*/stat.npy (NOT COMBINED/STAT.NPY!!!!)

    Returns:
        Lx: # pixels along x
        Ly: # pixels along y
    """
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    plane_ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()

    Lx = plane_ops['Lx']
    Ly = plane_ops['Ly']
    return Lx, Ly


# %% Functions for loading/saving suite2p cell stats

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


# %% Functions for extracting & saving roi medians


def _stats_2_df_med0_xy(stats):
    """Converts `stats` (loaded from stat.npy) to a dataframe with columns ('med_y', 'med_x').

    Note: if stat.npy is from a 3D recording, this does not correct for combined image sizes.

    Args:
        stats (np.ndarray): loaded from "combined/stat.npy" or "plane*/stat.npy"

    Returns:
        df_med0 (pd.DataFrame): dataframe w/ cols ('med_y', 'med_x') from `stats`
    """
    med_list = [s['med'] for s in stats]
    df_med0_xy = pd.DataFrame(med_list, columns=['med_y', 'med_x'])
    return df_med0_xy


def _stat_file_2_df_med0_xyz(stat_file):
    """Converts 'stat.npy' to a dataframe with columns ('med_y', 'med_x', 'med_z').

    Note: `df_med0` means coordinates are uncorrected for the combined image view.

    Args:
        stat_file (Path): file path to "combined/stat.npy" or "plane*/stat.npy"

    Returns:
        df_med0 (pd.DataFrame): dataframe w/ cols ('med_y', 'med_x', 'med_z') from `stats`
    """

    stats = np.load(stat_file, allow_pickle=True)
    df_med0_xy = _stats_2_df_med0_xy(stats)

    if is_3d(stat_file):
        med_z = [s['iplane'] for s in stats]
        df_med0_xyz = df_med0_xy.assign(med_z=med_z)
    else:  # if single plane
        iplane = path_to_plane(stat_file)
        df_med0_xyz = df_med0_xy.assign(med_z=iplane)
    df_med0_xyz = df_med0_xyz.loc[:, ['med_x', 'med_y', 'med_z']]
    return df_med0_xyz


def _stat_file_2_df_med_xyz(stat_file):
    """Corrects dataframe of medians from stat.npy to correct for combined image view.

    Args:
        stat_file (Path): file path to "combined/stat.npy" or "plane*/stat.npy"

    Returns:
        df_med (pd.DataFrame): dataframe w/ cols ('med_y', 'med_x', 'med_z') from `stats`
    """
    df_med_xyz = _stat_file_2_df_med0_xyz(stat_file)
    Lx, Ly, Lz = get_dims(stat_file)

    if is_3d(stat_file):
        df_med_xyz['med_x'] = df_med_xyz['med_x'].mod(Lx)
        df_med_xyz['med_y'] = df_med_xyz['med_y'].mod(Ly)
    return df_med_xyz


def stat_file_2_df_med(stat_file):
    """Converts `stat_file` into `df_med`, and writes it to 'cell_stats/df_med.csv'.

    Args:
        stat_file (Path): file path to "combined/stat.npy" or "plane*/stat.npy"
        save_file (bool): if True, writes df_med to 'cell_stats/df_med.csv'

    Returns:
        df_med (pd.DataFrame): dataframe w/ cols ('med_x', 'med_y', 'med_z') from `stats`,
                                 corrected for combined image view.
    """
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    df_med = _stat_file_2_df_med0_xyz(stat_file)

    # if save_file:
    #     save_dir = stat_file.with_name('cell_stats')
    #     save_dir.mkdir(exist_ok=True)
    #     df_med.to_csv(save_dir.joinpath('df_med.csv'), index=False)

    return df_med


# %% Functions for loading suite2p registration results


def load_single_plane_reg_tiffs(reg_dir):
    """Load registered tiff stacks from suite2p/plane** folder.

    Args:
        reg_dir (Path): Path to `reg_tif`, contains tiffs named file{:03d}_chan0.tif

    Returns:
        np.array: registered movie
    """
    if isinstance(reg_dir, str):
        reg_dir = Path(reg_dir)

    files = sorted(list(reg_dir.glob("file*_chan0.tif")))

    # Load multipage tiff files, and combine into a single movie

    stacks = []
    for file in files:
        with tifffile.TiffFile(file) as tif:
            img = np.stack([page.asarray() for page in tif.pages], axis=0)
            stacks.append(img)

    reg_plane = np.concatenate(stacks, axis=0)
    return reg_plane


def load_combined_reg_tiffs(stat_file, drop_flyback=False):
    """ Load and combine all registered suite2p .tiff files from `stat.npy`.

    Args:
        stat_file (Path): path to stat.npy file
        drop_flyback (bool): whether to drop flyback planes from registered movie  (default: False)

    Returns:
        np.array: registered movie, w/ dtype=np.int16
    """
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    Lx, Ly, Lz = get_dims(stat_file)
    T = get_frames(stat_file)

    # create empty array to hold registered movie
    reg_stack = np.zeros((T, Lz, Ly, Lx), dtype=np.int16)

    # loop through plane folders and load registered planes
    for iplane in range(Lz):
        plane_dir = get_suite2p_folder(stat_file) / f'plane{iplane:d}' / 'reg_tif'
        if plane_dir.is_dir():
            reg_stack[:, iplane, :, :] = load_single_plane_reg_tiffs(plane_dir)

    if drop_flyback:
        nonflyback_planes = get_nonflyback(stat_file)
        reg_stack = reg_stack[:, nonflyback_planes, :, :]

    return reg_stack


def load_reg_stack_from_combined_ops(ops_file):
    """ Loads registered movie from the plane folders in /suite2p

    Args:
        ops_file (object):

    Returns:
        stack (np.ndarray): registered suite2p stack
    """

    ops = np.load(ops_file, allow_pickle=True).item()
    s2p_path = get_suite2p_folder(ops_file)
    bin_files = list(s2p_path.rglob("plane*/*.bin"))
    bin_files.sort(key=lambda x: path_to_plane(x))
    stack = np.stack([load_plane_from_bin(item).data for item in bin_files], axis=1)
    return stack


def load_plane_from_bin(bin_file):
    """ Loads registered binary from file, and returns np.ndarray. """
    ops = np.load(bin_file.with_name('ops.npy'), allow_pickle=True).item()
    img = suite2p.io.BinaryFile(Lx=ops['Lx'], Ly=ops['Ly'], read_filename=bin_file)

    # with open(bin_file, mode='rb') as f:
    #     img = np.fromfile(f, np.int16).reshape(-1, ops['Ly'], ops['Lx'])

    return img


def load_bin_files_from_suite2p(s2p_path):
    """ Load and combine all registered .bin files from suite2p folder."""
    bin_files = list(s2p_path.rglob("plane*/*.bin"))
    bin_files.sort(key=lambda x: path_to_plane(x))
    reg_stack = np.stack([load_plane_from_bin(item).data for item in bin_files], axis=1)

    return reg_stack


# %% Functions for loading suite2p cell masks

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
def get_plane_patches(stat_file):
    """Gets locations of each plane in the combined FOV image for multiplane movies.

    Args:
        stat_file (Path): path to stat.npy file
    Returns:
        list: (y, x) coordinates of the top left corner of each plane in the combined FOV image.
    """
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()
    combined_Ly = ops['Ly']
    combined_Lx = ops['Lx']

    Lx, Ly, Lz = get_dims(stat_file)
    n_planes = get_nplanes(stat_file)

    y_patches = np.arange(0, combined_Ly, Ly)
    x_patches = np.arange(0, combined_Lx, Lx)
    patch_indices = [(iy, ix) for iy in y_patches for ix in x_patches]
    patch_indices = patch_indices[:n_planes]
    return patch_indices


def stat_file_2_label_img0(stat_file):
    """Convert suite2p stat.npy file to label image. Not corrected for combined FOV.

    Args:
        stat_file (Path): path to stat.npy file

    Returns:
        np.array: label image
    """
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    stats = np.load(stat_file, allow_pickle=True)
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()

    im = np.zeros((ops['Ly'], ops['Lx']))
    n_cells = stats.size

    for n in range(0, n_cells):
        ypix = stats[n]['ypix'][~stats[n]['overlap']]
        xpix = stats[n]['xpix'][~stats[n]['overlap']]
        im[ypix, xpix] = n + 1
    im = im.astype(np.float32)
    return im


def stat_file_2_label_img(stat_file):
    """Convert suite2p stat.npy file to label image. Corrected for combined FOV."""
    if isinstance(stat_file, str):
        stat_file = Path(stat_file)

    label_img0 = stat_file_2_label_img0(stat_file)

    is_multiplane = is_3d(stat_file)

    if is_multiplane:
        Lx, Ly, Lz = get_dims(stat_file)
        patch_idx_list = get_plane_patches(stat_file)

        label_img_list = []

        # extract label image patch for each plane
        for y0, x0 in patch_idx_list:
            plane_limg = label_img0[y0:y0+Ly, x0:x0+Lx]
            label_img_list.append(plane_limg)

        # combine label images for each plane
        label_img = np.stack(label_img_list, axis=0)
    else:
        label_img = label_img0
    label_img = label_img.astype(np.float32)
    return label_img


def get_cell_mask_2d(stat, Lx, Ly, allow_overlap=False):
    """

    Args:
        stat (dict): loaded from `stat.npy` in suite2p folder
                        (either .../suite2p/plane<#>/stat.npy, or
                        .../suite2p/combined/stat.npy)
        Lx ():
        Ly ():
        allow_overlap ():

    Returns:
        cell_mask (np.array): flat index into array of size (Lz, Ly, Lx)
        lam_normed (np.array): pixel weights of cell_masks (sum to 1)

    """
    mask = ... if allow_overlap else ~stat['overlap']
    ypix = np.remainder(stat['ypix'], Ly)
    xpix = np.remainder(stat['xpix'], Lx)
    cell_mask = np.ravel_multi_index((ypix, xpix), (Ly, Lx))
    cell_mask = cell_mask[mask]
    lam = stat['lam'][mask]
    lam_normed = lam / lam.sum() if lam.size > 0 else np.empty(0)
    return cell_mask, lam_normed


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


# %% Functions for computing center of mass of cell masks

def get_com(stat, allow_overlap=False):
    """Get uncorrected center of `stat` (for one cell)."""
    mask = ... if allow_overlap else ~stat['overlap']
    ypix = stat['ypix'][mask]
    xpix = stat['xpix'][mask]
    lam = stat['lam'][mask]
    lam_normed = lam / lam.sum() if lam.size > 0 else np.empty(0)
    com_x = np.dot(xpix, lam_normed)
    com_y = np.dot(ypix, lam_normed)
    return com_y, com_x


def get_com_corrected(stat, Lx, Ly, allow_overlap=False):
    """Get center of `stat` (for one cell), corrected for combined FOV."""
    com_y0, com_x0 = get_com(stat, allow_overlap=allow_overlap)
    com_y = com_y0 % Ly
    com_x = com_x0 % Lx
    return com_y, com_x


def stat_file_2_df_com0(stat_file):
    """Convert  stat.npy file to dataframe w/ center of mass coordinates, uncorrected for
    combined FOV.

    Args:
        stat_file (Path): path to stat.npy file

    Returns:
        pd.DataFrame: dataframe of cell centers of mass, with columns ('com_x', 'com_y')

    Note: if "combined/stat.npy", centers of mass are not corrected for combined FOV.

    """
    stats = np.load(stat_file, allow_pickle=True)
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()

    com_list = [get_com(s, allow_overlap=ops['allow_overlap']) for s in stats]
    df_com0 = pd.DataFrame(com_list, columns=['com_y', 'com_x'])

    return df_com0


def stat_file_2_df_com(stat_file):
    """Convert suite2p stat.npy file to dataframe of cell centers of mass.

    Args:
        stat_file (Path): path to stat.npy file

    Returns:
        pd.DataFrame: dataframe of cell centers of mass, with columns ('com_x', 'com_y', 'com_z')

    Note: if "combined/stat.npy", centers of mass are not corrected for combined FOV.
    """
    stats = np.load(stat_file, allow_pickle=True)
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()

    is_multiplane = is_3d(stat_file)
    Lx, Ly, Lz = get_dims(stat_file)

    com_list = [get_com_corrected(s, Lx=Lx, Ly=Ly, allow_overlap=ops['allow_overlap']) for s in
                stats]
    df_com = pd.DataFrame(com_list, columns=['com_y', 'com_x'])

    if is_multiplane:
        com_z = [s['iplane'] for s in stats]
        df_com['com_z'] = com_z
    else:
        df_com['com_z'] = path_to_plane(stat_file)
    return df_com


# %%
def get_mask_px_3d(stat, Lx, Ly, Lz, allow_overlap=False):
    mask = ... if allow_overlap else ~stat['overlap']
    ypix = np.remainder(stat['ypix'][mask], Ly)
    xpix = np.remainder(stat['xpix'][mask], Lx)
    zpix = np.ones_like(xpix) * stat['iplane']

    lam = stat['lam'][mask]
    lam_normed = lam / lam.sum() if lam.size > 0 else np.empty(0)

    return zpix, ypix, xpix, lam_normed


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


# %%
def parse_stats_to_tables(stats, Lx, Ly, Lz, allow_overlap=False):
    df_scalar_stats = scalar_stats_to_dataframe(stats)
    df_med = stats_to_df_med(stats, Lx, Ly, allow_overlap=allow_overlap)
    df_com = stats_to_df_com(stats, Lx, Ly, Lz, allow_overlap=allow_overlap)
    return df_scalar_stats, df_med, df_com


def add_extra_suite2p_outputs(stat_file):
    """Add extra outputs in folder containing 'stat.npy', under subdirectory 'cell_stats'.

    """
    stats = np.load(stat_file, allow_pickle=True)
    ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()

    is_multiplane = is_3d(stat_file)
    Lx, Ly, Lz = get_dims(stat_file)

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

    # save spatial footprints
    A = get_spatial_footprints_3d(stat_file, Lx, Ly, Lz)
    # A_sp = sparse.csc_array(A)
    A_sp = sparse.csc_matrix(A)
    sparse.save_npz(cell_stat_dir.joinpath('A.npz'), A_sp)

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

    A = np.zeros((Lx * Ly * Lz, n_cells))
    for i, stat in enumerate(stats):
        cell_mask, lam_normed = get_cell_mask_3d(stat, Lx, Ly, Lz,
                                                 allow_overlap=ops['allow_overlap'])
        a = np.zeros(Lx * Ly * Lz)
        a[cell_mask] = lam_normed
        A[:, i] = a
    A = A.astype('float32')
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
#
#
# if __name__ == '__main__':
#     import xarray as xr
#
#     DATA_DIR = Path("/local/storage/Remy/for_dhruvs_paper")
#     stat_files = sorted(list(DATA_DIR.rglob('8/**/stat.npy')))
#
#     for stat_file in stat_files:
#         stats = np.load(stat_file, allow_pickle=True)
#         ops = np.load(stat_file.with_name('ops.npy'), allow_pickle=True).item()
#
#         F = np.load(stat_file.with_name('F.npy'))
#         Ly = ops['Ly']
#         Lx = ops['Lx']
#
#         n_trials = 3
#         n_cells, T = F.shape
#
#         #################################
#         # process F.npy --> df/f
#         #################################
#         F_trial = np.stack(np.split(F, 3, axis=1))
#         da_F = xr.DataArray(F_trial,
#                             dims=['trial', 'cell', 'time'],
#                             coords=dict(trial=range(3),
#                                         time=range(165),
#                                         cell=range(n_cells)))
#         da_baseline = da_F.sel(time=slice(20, 60)).mean(dim='time')
#         da_dff = (da_F - da_baseline) / da_baseline
#
#         arr_dff_trials = da_dff.to_numpy()
#         arr_dff = np.concatenate(tuple(arr_dff_trials), axis=1)
#
#         #################################################
#         # save spatial footprints to tifs
#         #################################################
#         mov_dir = get_suite2p_folder(stat_file).parent
#
#         limg = get_mask_labels(stats, Lx, Ly, 1)
#         limg = limg.squeeze()
#         utils2p.save_img(mov_dir.joinpath('suite2p_roi_labels.tif'), limg.astype('uint16'))
#
#         A = get_spatial_footprints_2d(stat_file)
#         lam_arr = A.reshape(Ly, Lx, A.shape[1]).T
#         utils2p.save_img(mov_dir.joinpath('suite2p_roi_lam.tif'), lam_arr.astype('float32'))
#
#         lam_maxnormed = (A / A.max(axis=0)).reshape((Ly, Lx, A.shape[1])).T
#         utils2p.save_img(mov_dir.joinpath('suite2p_roi_lam_maxnormed.tif'),
#                          lam_maxnormed.astype('float32'))
#
#         lam_sumnormed = (A / A.sum(axis=0)).reshape((Ly, Lx, A.shape[1])).T
#         utils2p.save_img(mov_dir.joinpath('suite2p_roi_lam_sumnormed.tif'),
#                          lam_sumnormed.astype('float32'))
#         #
#         # A_maxnorm = A/A.max(axis=0)
#         # denoised = (A_maxnorm @ C).T.reshape(495, Ly, Lx)
#         # utils2p.save_img(stat_file.with_name('dff_recons_Amaxnorm_x_C.tif'), denoised.astype('float32'))
#         #
#         # denoised = (A @ F).T.reshape(495, Ly, Lx)
#         # utils2p.save_img(stat_file.with_name('movie_reconstructed_A_x_F.tif'), denoised.astype('float32'))
#         #
#         #
#         #######################################################
#         # generate denoised movie from suite2p outputs
#         ########################################################
#         denoised = (A / A.max(axis=0)) @ F
#         denoised = denoised.T.reshape(495, Ly, Lx)
#         utils2p.save_img(mov_dir.joinpath('dff_recons_Amaxnormed_x_F.tif'),
#                          denoised.astype('float32'))
#
#         C = arr_dff
#         A_limg = A > 0
#         denoised = (A_limg @ C).T.reshape(495, Ly, Lx)
#         utils2p.save_img(mov_dir.joinpath('dff_recons_Apix_x_C.tif'), denoised.astype('float32'))
#         #
#         df_dff = pd.DataFrame(C.T)
#         df_dff.to_csv(mov_dir.joinpath('suite2p_dff.csv'), index_label=False)


# %%
# Lx, Ly, Lz = get_3d_shape(stat_file)
# A = get_spatial_footprints_3d(stat_file, Lx, Ly, Lz)
# #
# # A = suite2p_helpers.get_spatial_footprints_2d(stat_file)
# # # plt.imshow(A.sum(axis=1).reshape(224, 224), cmap=gray)
# # # plt.show()
# # # %%
# denoised = (A @ C).T.reshape(495, Ly, Lx)
# %%
# utils2p.save_img(stat_file.with_name('dff_recons.tif'), denoised.astype('float32'))
#
# # %%
# f

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
