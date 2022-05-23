""" Module w/ functions for extracting summary images from `ops.npy` and `stat.npy` files,
 and saving them to `.tif` files for easy viewing.

See suite2p.readthedocs.io/en/latest/outputs.html for further info.

Summary images include:
- refImg
- meanImg
- meanImgE
- Vcorr
"""
from pathlib import Path
from typing import List

import numpy as np
import tifffile

from s2p import suite2p_helpers


def extract_from_plane_ops_files(ops_key, ops_file_list):
    """ Loads ops values from list of ops.npy file paths, and returns them in a list.

    Args:
        ops_key (str): ops key
        ops_file_list (List[Path]): List of ops.npy files
    Returns:
        plane_items (List): List of ops values
    """
    plane_items = []
    for ops_file in ops_file_list:
        ops = np.load(ops_file, allow_pickle=True).item()

        if ops_key == 'Vcorr':
            x = np.arange(ops['Lx'])
            y = np.arange(ops['Ly'])
            xgrid, ygrid = np.meshgrid(x, y)
            ymin, ymax = ops['yrange']
            xmin, xmax = ops['xrange']
            mask = (xgrid >= xmin) & (xgrid < xmax) & (ygrid >= ymin) & (ygrid < ymax)
            Vcorr = np.zeros((ops['Ly'], ops['Lx']))
            Vcorr[np.flatnonzero(mask)] = ops['Vcorr']
            plane_items.append(Vcorr)
        if ops_key != 'Vcorr':
            plane_items.append(ops[ops_key])
    return plane_items


def load_summary_images(ops_file):
    """ Extracts refImg, meanImg, meanImgE, and Vcorr images from ops.npy, and saves them as tiff files.
    Args:
        ops_file (Path): Path to ops.npy file
    Returns:
        saved_files (List[Path]): List of .tif files saved.
    """
    if isinstance(ops_file, str):
        ops_file = Path(ops_file)
    if ops_file.parent.name == 'combined':
        # print('combined')
        combined_ops = np.load(ops_file, allow_pickle=True).item()
        s2p_path = suite2p_helpers.get_suite2p_folder(ops_file)

        n_planes = combined_ops['nplanes']
        ignore_flyback = combined_ops['ignore_flyback']

        # list of non-ignored flyback plane folders
        plane_ops_files = [s2p_path.joinpath(f"plane{z}", "ops.npy") for z in range(n_planes) if
                           z not in ignore_flyback]

        # load summary images

        ref_img_list = extract_from_plane_ops_files('refImg', plane_ops_files)
        mean_img_list = extract_from_plane_ops_files('meanImg', plane_ops_files)
        mean_imge_list = extract_from_plane_ops_files('meanImgE', plane_ops_files)
        # vcorr_list = extract_from_plane_ops_files('Vcorr', plane_ops_files)

        ref_img = np.stack(ref_img_list, axis=0)
        mean_img = np.stack(mean_img_list, axis=0)
        mean_imge = np.stack(mean_imge_list, axis=0)
        # vcorr_img = np.stack(vcorr_list, axis=0)

    else:
        # print('single plane')
        ops = np.load(ops_file, allow_pickle=True).item()
        ref_img = ops['refImg']
        mean_img = ops['meanImg']
        mean_imge = ops['meanImgE']

    return ref_img, mean_img, mean_imge


def extract_summary_images_to_tiff_files(ops_file):
    if isinstance(ops_file, str):
        ops_file = Path(ops_file)
    ref_img, mean_img, mean_imge = load_summary_images(ops_file)
    mean_img = mean_img.astype('float32')

    tifffile.imsave(ops_file.with_name('ref_img.tif'), ref_img, 'imagej')
    tifffile.imsave(ops_file.with_name('mean_img.tif'), mean_img, 'imagej')
    tifffile.imsave(ops_file.with_name('mean_imge.tif'), mean_imge, 'imagej')
    return [ops_file.with_name(item) for item in ['ref_img.tif', 'mean_img.tif', 'mean_imge.tif']]


#if __name__ == '__main__':
    #cli()
