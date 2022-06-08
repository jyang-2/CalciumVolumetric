""" This file contains manually chosen cluster identities for responding cell groups."""
from pathlib import Path

import numpy as np

NAS_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/processed_data")

good_xid_lookup = {"2022-02-10/1/kiwi/downsampled_3/suite2p/combined/embedding_allcells.npy":
                       list(range(14, 19)) + list(range(30, 38)),
                   "2022-02-10/2/kiwi/downsampled_3/suite2p/combined/embedding_allcells.npy":
                       list(range(0, 4)) + [7, 10, 35, 36] + list(range(20, 29)),
                   "2022-02-10/2/kiwi_ea_eb_only/downsampled_3/suite2p/combined/embedding_allcells.npy":
                       list(range(17, 22)) + list(range(24, 30)),
                   "2022-02-11/1/kiwi/downsampled_3/suite2p/combined/embedding_allcells.npy":
                       [9, 11, 36, 39] + list(range(13, 23)) + list(range(27, 34)),
                   "2022-02-11/1/kiwi_ea_eb_only/downsampled_3/suite2p/combined/embedding_allcells.npy":
                       [10, 11] + list(range(18, 25)) + list(range(33, 40)),
                   "2022-02-11/2/kiwi_ea_eb_only/downsampled_3/suite2p/combined/embedding_allcells.npy":
                       list(range(4, 12)) + list(range(13, 17)),
                   "2022-02-11/3/kiwi/downsampled_3/suite2p/combined/embedding_allcells.npy":
                       [0, 23] + list(range(11, 15)) + [15, 16, 17] + list(range(30, 40)),
                   "2022-02-11/3/kiwi_ea_eb_only/downsampled_3/suite2p/combined/embedding_allcells.npy":
                       [5, 20, 21] + list(range(30, 39)),
                   "2022-03-29/1/control1/source_extraction_s2p/suite2p/combined/embedding_allcells.npy":
                       list(range(0, 6)) + list(range(7, 16)) + list(range(28, 40)),
                   "2022-03-29/1/control1_top2_ramps/source_extraction_s2p/suite2p/combined/embedding_allcells.npy":
                       list(range(0, 4)) + list(range(23, 35)) + [38, 39],
                   "2022-03-29/2/control1/source_extraction_s2p/suite2p/combined/embedding_allcells.npy":
                       list(range(8, 18)) + list(range(22, 28)) + list(range(29, 40)),
                   "2022-03-29/2/control1_top2_ramps_001/source_extraction_s2p/suite2p/combined/embedding_allcells.npy":
                       list(range(4, 8)) + list(range(9, 19)) + list(range(24, 30)),
                   "2022-04-04/1/control1_001/stk/suite2p/combined/embedding_allcells.npy":
                       list(range(9)) + [17] + list(range(20, 24)) + list(range(28, 32)) + list(range(33, 40)),
                   "2022-04-04/1/control1_top2_ramps/caiman_mc_els/suite2p/combined/embedding_allcells.npy":
                       list(range(7)) + [21] + list(range(23, 35)),
                   "2022-04-09/1/kiwi/source_extraction_s2p/suite2p/combined/embedding_allcells.npy":
                       list(range(2, 16)) + [25] + list(range(29, 38)),
                   "2022-04-09/1/kiwi_ea_eb_only/source_extraction_s2p/suite2p/combined/embedding_allcells.npy":
                       [0, 5, 6, 26] + list(range(12, 21)) + list(range(33, 40)),
                   "2022-04-09/2/kiwi/source_extraction_s2p/suite2p/combined/embedding_allcells.npy":
                       list(range(0, 4)) + [5, 6, 13] + list(range(18, 25)) + list(range(30, 40)),
                   "2022-04-09/2/kiwi_ea_eb_only/source_extraction_s2p/suite2p/combined/embedding_allcells.npy":
                       list(range(8, 11)) + list(range(12, 17)) + list(range(18, 25)),
                   "2022-04-10/1/kiwi/source_extraction_s2p/suite2p/combined/embedding_allcells.npy":
                       list(range(4, 20)) + list(range(28, 37)),  # 1834/3428 cells
                   "2022-04-10/1/kiwi_ea_eb_only/source_extraction_s2p/suite2p/combined/embedding_allcells.npy":
                       [0] + list(range(2, 9)) + list(range(25, 30))
                   }


# %%

def write_good_xid_to_file(file, good_xid):
    if isinstance(file, str):
        file = Path(file)

    if file.is_file():
        embedding = np.load(file, allow_pickle=True).item()
        embedding['good_xid'] = good_xid
        np.save(file, embedding)
        print(f"\t- good_xid added to {file}")

        embedding_asdict = np.load(file.with_name('embedding_allcells_asdict.npy'), allow_pickle=True).item()
        embedding_asdict['good_xid'] = good_xid
        np.save(file.with_name('embedding_allcells_asdict.npy'), embedding_asdict)

        return True

    else:
        return False


# %%

for i, (fname, good_clusters) in enumerate(good_xid_lookup.items()):
    fname_embedding = NAS_PROC_DIR / Path(fname)

    print(f"\n{i}.\t{fname_embedding}")
    print(f"\t- good_xid = {good_clusters}")
    print(f"\t- embedding_allcells.npy exists: {fname_embedding.is_file()}")

    if fname_embedding.is_file():
        print(write_good_xid_to_file(fname_embedding, sorted(good_clusters)))

