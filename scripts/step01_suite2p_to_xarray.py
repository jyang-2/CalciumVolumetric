#!/usr/bin/env python3
"""
Convert suite2p outputs (in .../suite2p/combined, contains file "stat.npy") to:
  - xrds_suite2p_outputs.nc
  - embedding_allcells.npy
  - embedding_allcells_asdict.npy
  - rastermap_embedding_allcells
  - NOT xrds_suite2p_trials.nc

Afterwards, pick the good rastermap cluster embeddings in src/s2p/rastermap_allcells_pick_clusters.py
"""
import sys
sys.path.append("/home/remy/PycharmProjects/CalciumVolumetric/src")

import matplotlib.pyplot as plt
import argparse

from pydantic_models import FlatFlyAcquisitions
# import suite2p
# import rastermap
# from rastermap.mapping import Rastermap
from s2p import rasterplot, s2p_to_xarray
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({'pdf.fonttype': 42,
                     'text.usetex': False})

# Set project directory
import natmixconfig
from natmixconfig import *
# NAS_PRJ_DIR = Path("/local/matrix/Remy-Data/projects/natural_mixtures")
# NAS_PROC_DIR = NAS_PRJ_DIR.joinpath("processed_data")


def main(flacq):
    mov_dir = flacq.mov_dir()
    print(flacq)
    print(f"\nMOV_DIR: {mov_dir}\n")

    stat_file = flacq.stat_file(relative_to=natmixconfig.NAS_PROC_DIR)
    # stat_file = list(mov_dir.rglob("combined/stat.npy"))[0]
    print(f"\n\tstat_file: {stat_file}")

    #####################################
    # create xrds_suite2p_outputs.nc
    #####################################
    # xrds_suite2p_outputs = s2p_to_xarray.flacq_2_xrds_suite2p_outputs(flacq.dict(),
    # save_netcdf=True)
    xrds_suite2p_outputs = s2p_to_xarray.flacq_2_xrds_suite2p_outputs(flacq,
                                                                      save_netcdf=True)
    xrds_file = stat_file.with_name('xrds_suite2p_outputs.nc')
    print(f"\n\t- saved {xrds_file.relative_to(mov_dir)}")

    #####################################
    # plot initial rastermap embedding
    #####################################
    fig1, axarr1 = rasterplot.plot_initial_rastermap_clustering(xrds_file)
    # plt.show()

    # save initial embedding to .../suite2p/combined/rastermap_embedding_allcells.pdf
    pdf_file = xrds_file.with_name('rastermap_embedding_allcells.pdf')
    if ~pdf_file.is_file():
        with PdfPages(pdf_file) as pdf:
            pdf.savefig(fig1)
            print('\trastermap_embedding_allcells.pdf saved.')
    else:
        print(f"\tpdf file already exists: {pdf_file.relative_to(NAS_PROC_DIR)}")

    return xrds_suite2p_outputs


if __name__ == '__main__':
    USAGE = 'Brief description of what the script does.'
    parser = argparse.ArgumentParser(description=USAGE)
    parser.add_argument('flacq_txt', type=str,
                        help='text format of json file, in FlatFlyAcquisition format')
    args = parser.parse_args()
    flat_fly_acq = FlatFlyAcquisitions.parse_raw(args.flacq_txt)
    main(flat_fly_acq)
    print('xrds_suite2p_outputs.nc saved successfully.')

