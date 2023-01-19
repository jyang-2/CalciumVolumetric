from pathlib import Path
from typing import List

import pydantic

import aggregate
import natmixconfig
from pydantic_models import FlatFlyAcquisitions
from s2p import suite2p_helpers
from scripts import aggregate_trial_rdms


def get_manuscript_flat_acqs(prj='odor_space_collab', imaging_type='kc_soma', panel='megamat'):
    """Helper function for loading manuscript datasets"""
    all_flat_acqs = pydantic.parse_file_as(List[FlatFlyAcquisitions], natmixconfig.MANIFEST_FILE)

    # filter flat_acqs
    movie_panel = aggregate.PanelMovies(prj=prj, panel=panel)
    flat_acq_list = aggregate.filter_flacq_list(all_flat_acqs,
                                                allowed_movie_types=movie_panel.movies,
                                                has_s2p_output=True,
                                                allowed_imaging_type=imaging_type,
                                                )

    if prj == 'odor_space_collab':
        flat_acq_list = list(filter(lambda x: suite2p_helpers.is_3d(x.stat_file(
                relative_to=natmixconfig.NAS_PROC_DIR)),
                                    flat_acq_list, ))
        if imaging_type == 'kc_soma':
            flat_acq_list = [flat_acq_list[i] for i in [0, 1, 2, 4]]

    return flat_acq_list


def combine_respvec_datasets(imaging_types=['kc_soma', 'pn_boutons']):
    """For each imaging region, load flat_acqs, aggregate respvecs w/ different settings,
    and save ds_agg_respvec to a .netcdf file.

    There are two parameter dicts: `panel_params`, which defines the imaging region and odor
    panel, and `respvec_params`, which contains the options for pooling the respvec datasets (
    whether or not to drop bad cell clusters, average trials within odor, and standardization
    method).

    See `aggregate_trial_rdms.get_respvec_param_combinations` or
    `aggregate_trial_rdms.aggregate_respvecs` for more information.
    """
    filename_list = []

    # loop over imaging regions
    for imaging_type in imaging_types:
        panel_params = dict(imaging_type=imaging_type, panel='megamat')

        flat_acq_list = get_manuscript_flat_acqs(prj='odor_space_collab',
                                                 imaging_type=panel_params['imaging_type'],
                                                 panel=panel_params['panel'])

        print('\nPANEL PARAMS:')
        print(panel_params)
        print('------------')

        # make list of respvec param combinations
        respvec_param_combos = aggregate_trial_rdms.get_respvec_param_combinations(
                respvec=['mean_peak'],
                trialavg=[True],
                good_xids_only=[False, True],
        )

        # loop through parameter options
        for i, respvec_params in enumerate(respvec_param_combos):
            print(f"{i}:")

            # aggregate respvec datasets
            # --------------------------
            ds_respvec_agg = aggregate_trial_rdms.aggregate_respvecs(
                    flat_acq_list,
                    **respvec_params
            )

            # add panel info to dataset
            ds_respvec_agg.attrs.update(panel_params)

            # save to netcdf file
            # --------------------
            # get output directory
            RESPVEC_OUTPUT_DIR = Path(aggregate_trial_rdms.RESPVEC_OUTPUT_DIR.format(
                    **panel_params))
            print(f"\t{RESPVEC_OUTPUT_DIR}")

            # make sure output directory is valid
            RESPVEC_OUTPUT_DIR = natmixconfig.NAS_PRJ_DIR / RESPVEC_OUTPUT_DIR
            if not RESPVEC_OUTPUT_DIR.is_dir():
                RESPVEC_OUTPUT_DIR.mkdir(parents=True)

            # get filename according to respvec aggregation parameters
            RESPVEC_OUTPUT_FILENAME = aggregate_trial_rdms.RESPVEC_OUTPUT_FILENAME.format(
                    **respvec_params)
            print(f"\t{RESPVEC_OUTPUT_FILENAME}")

            filename = RESPVEC_OUTPUT_DIR.joinpath(RESPVEC_OUTPUT_FILENAME)

            # fix dtypes in attrs
            for k, v in ds_respvec_agg.attrs.items():
                if isinstance(v, tuple):
                    ds_respvec_agg.attrs[k] = list(v)
                elif isinstance(v, bool):
                    ds_respvec_agg.attrs[k] = str(v)
                elif v is None:
                    ds_respvec_agg.attrs[k] = str(v)

            # for k, v in ds_respvec_agg.attrs.items():
            #     print(f"{k}, {v}, {type(v)}")

            ds_respvec_agg.to_netcdf(filename)
            print(f'\tds_respvec_agg saved to:')
            print(f'\t{filename.as_uri()}')
            filename_list.append(filename)

    return filename_list
