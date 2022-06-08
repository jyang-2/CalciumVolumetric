import numpy as np
from typing import List

import numpy as np
import pydantic
import xarray as xr
from rastermap.mapping import Rastermap

from config import *
from pydantic_models import FlatFlyAcquisitions


#%%


def load(xrds_file):
    ds = xr.load_dataset(stat_file.with_name('xrds_suite2p_outputs_xid0.nc'))
    ds = ds.sortby('cells')
    good_xid = ds.attrs['good_xid']
    ds = ds.where(ds.xid.isin(good_xid), drop=True)
    return ds


def transform(ds):
    ##########################
    # Run rastermap clustering
    ##########################
    ops = {'n_components': 1,
           'n_X': 40,
           'alpha': 1.,
           'K': 1.,
           'nPC': 200, 'constraints': 2,
           'annealing': True, 'init': 'pca'}

    model = Rastermap(**ops)
    embedding = model.fit_transform(ds.Fc_normed.to_numpy())
    return model, ops, embedding


def save_embedding(filepath, ops, model, data_description, cells):
    np.save(filepath,
            dict(ops=ops,
                 model=model,
                 data_description=data_description,
                 cells=cells))
    return filepath


def save_embedding_as_dict(filepath, ops, model, data_description, cells):
    np.save(filepath,
            dict(ops=ops,
                 model=model.__dict__,
                 data_description=data_description,
                 cells=cells))
    return filepath


if __name__ == '__main__':

    movie_types = ['kiwi', 'control1', 'control1_top2_ramps', 'kiwi_ea_eb_only']

    manifest_json = Path("/local/storage/Remy/natural_mixtures/manifestos/flat_linked_thor_acquisitions.json")
    flat_acqs = pydantic.parse_file_as(List[FlatFlyAcquisitions], manifest_json)

    for flacq in filter(lambda x: x.movie_type in movie_types, flat_acqs):
        print('---')
        print(f"\n{flacq}")

        stat_file = NAS_PROC_DIR.joinpath(flacq.s2p_stat_file)

        ###################################
        # load data, make sure it's sorted
        ###################################
        xrds_suite2p_outputs = xr.load_dataset(stat_file.with_name('xrds_suite2p_outputs_xid0.nc'))
        xrds_suite2p_outputs = xrds_suite2p_outputs.sortby('cells')

        ###########################################
        # drop all cells not in a good_xid cluster
        ###########################################
        good_xid = xrds_suite2p_outputs.attrs['good_xid']
        ds = xrds_suite2p_outputs.where(xrds_suite2p_outputs.xid.isin(good_xid), drop=True)

        ##########################
        # Run rastermap clustering
        ##########################
        ops = {'n_components': 1,
               'n_X': 40,
               'alpha': 1.,
               'K': 1.,
               'nPC': 200, 'constraints': 2,
               'annealing': True, 'init': 'pca'}

        model = Rastermap(**ops)
        embedding = model.fit_transform(ds.Fc_normed.to_numpy())

        ###############
        # save results
        ###############
        save_embedding(stat_file.with_name('embedding_goodxids.npy'), ops=ops, model=model,
                       data_description='Fc_normed', cells=ds.cells.to_numpy())

        save_embedding_as_dict(stat_file.with_name('embedding_goodxids.npy'), ops=ops, model=model,
                               data_description='Fc_normed', cells=ds.cells.to_numpy())

        ds = ds.assign_coords(dict(xid1=('cells', model.xid)))
        ds = ds.assign_coords(dict(embedding1=('cells', model.embedding.squeeze())))
        ds.to_netcdf(stat_file.with_name('xrds_suite2p_outputs_xid1.nc'))


