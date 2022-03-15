from pathlib import Path
from typing import Union

import numpy as np
import parse
# import suite2p
import utils2p


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
    s2p_path = suite2p.io.utils.get_suite2p_path(ops_file)
    bin_files = list(s2p_path.rglob("plane*/*.bin"))
    bin_files.sort(key=lambda x: path_to_plane(x))
    stack = np.stack([load_plane_from_bin(item).data for item in bin_files])
    return stack


def load_plane_from_bin(bin_file):
    """ Loads registered binary from file, and returns np.ndarray. """
    ops = np.load(bin_file.with_name('ops.npy'), allow_pickle=True).item()
    img = suite2p.io.BinaryFile(Lx=ops['Lx'], Ly=ops['Ly'], read_filename=bin_file)
    return img.data


def load_bin_files_from_suite2p(s2p_path):
    """ Load and combine all registered .bin files from suite2p"""
    bin_files = list(s2p_path.rglob("plane*/*.bin"))
    bin_files.sort(key=lambda x: path_to_plane(x))
    reg_stack = np.stack([load_plane_from_bin(item).data for item in bin_files])
    save_dir = s2p_path.with_name('s2p_reg_tif')
    save_dir.mkdir(parents=True, exist_ok=True)
    utils2p.save_img(s2p_path.joinpath('reg_stack.tif'), reg_stack)
    return reg_stack
#%%

NAS_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data/2022-02-11")
combined_ops_files = list(NAS_PROC_DIR.rglob("combined/ops.npy"))

for ops_file in combined_ops_files:
    c_ops = np.load(ops_file, allow_pickle=True).item()

    # get suite2p folder
    s2p_path = suite2p.io.utils.get_suite2p_path(ops_file)

    print('')
    print(ops_file)
    print(s2p_path)

    # get binary files for registered movies, and sort by plane
    bin_files = list(s2p_path.rglob("plane*/*.bin"))
    bin_files.sort(key=lambda x: path_to_plane(x))

    # load and combine registered planes
    reg_stack = np.stack([load_plane_from_bin(item).data for item in bin_files])
    print('stack created')
    print(f"shape: {reg_stack.shape}")

    save_dir = s2p_path.with_name('s2p_reg_tif')
    save_dir.mkdir(parents=True, exist_ok=True)
    utils2p.save_img(save_dir.joinpath('reg_stack.tif'), reg_stack)
    print("registed tiff stack saved.")






#%% save registered tiff to save_path0

c_ops = np.load(ops_file, allow_pickle=True).item()
save_file = Path(c_ops['save_path']).joinpath('s2p_reg_tif', 'reg_stack.tif')
save_file.parent.mkdir(parents=True, exist_ok=True)
utils2p.save_img(save_file, reg_stack)
#%%


#%%%
folder = Path("/local/storage/Remy/natural_mixtures/processed_data/2022-02-10/1/kiwi/downsampled_3")

fpath = "{parent}/suite2p/plane{plane:d}/{tiff_file}"
tiff_files = list(folder.rglob("*/reg_tif/*.tif"))
tiff_files.sort(key=lambda x: parse.search(fpath, str(x))['plane'])

movies = cm.load_movie_chain(tiff_files, is3D=True)



