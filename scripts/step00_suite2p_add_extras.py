#!/usr/bin/env python3

import sys
sys.path.append("/home/remy/PycharmProjects/CalciumVolumetric/src")

from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# import suite2p
# import rastermap
# from rastermap.mapping import Rastermap
from s2p import rasterplot, suite2p_helpers, extract_summary_images

plt.rcParams.update({'pdf.fonttype': 42,
                     'text.usetex': False})

# Set project directory
#NAS_PRJ_DIR = Path("/local/matrix/Remy-Data/projects/natural_mixtures")
#NAS_PROC_DIR = NAS_PRJ_DIR.joinpath("processed_data")


def main(stat_file):
    """

    Args:
        stat_file (Path): path to .../suite2p/combined/stat.npy

    Returns:

    """
    # save cellstats to .../suite2p/combined/cell_stats
    #    - df_com.csv
    #    - df_med.csv
    #    - df_scalar_stats.csv
    #    - label_img.(tif, npy)

    suite2p_helpers.add_extra_suite2p_outputs(stat_file)

    # saves 'ref_img.tif', 'mean_img.tif', 'mean_imge.tif' in .../suite2p/combined
    extract_summary_images.extract_summary_images_to_tiff_files(stat_file.with_name('ops.npy'))

    # runs initial rastermap embedding, and saves to .../suite2p/combined/embedding_allcells.npy
    model = rasterplot.run_initial_rastermap_embedding(stat_file, save=True)

    # # generate denoised movie
    # Lx, Ly, Lz = suite2p_helpers.get_3d_shape(stat_file)
    # A = suite2p_helpers.get_spatial_footprints_3d(stat_files[0], Lx=256, Ly=256, Lz=16)
    # denoised =  (A @ C).T.reshape(495, Ly, Lx)
    return model


if __name__ == '__main__':
    USAGE = 'Adds some extra files to suite2p output folder..'
    parser = argparse.ArgumentParser(description=USAGE)
    parser.add_argument('stat_file', type=str,
                        help='path to .../suite2p/combined/stat.npy')
    args = parser.parse_args()

    main(Path(args.stat_file))
    print('suite2p extras saved successfully.')
