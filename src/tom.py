"""Functions for making natural mixture data compatible with tom's analysis code.

Required inputs:
- 'pin_odor_mixture_list.json'
"""
import copy
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import pydantic
import xarray as xr
from sklearn import preprocessing

import natmixconfig
import ryeutils
from pydantic_models import FlatFlyAcquisitions, PinOdorMixture

tom_abbrevs = {'kiwi approx.': '~kiwi',
               'ethyl acetate': 'EA',
               'ethyl butyrate': 'EB',
               'isoamyl alcohol': 'IAol',
               'isoamyl acetate': 'IAA',
               'ethanol': 'EtOH',
               '1-octen-3-ol': 'OCT',
               '2-heptanone': '2H',
               'methyl salicylate': 'MS',
               'valeric acid': 'VA',
               'furfural': 'FUR',
               'control mix': 'control mix',
               'paraffin': 'pfo',
               'pfo': 'pfo',
               # 'trans-2-hexenal': 'T2H',
               # '3-methylthio-1-propanol': '3MT1P'
               }


def fix_abbrevs(pin_odor_mixlist, abbrevs):
    """ Loop through all PinOdors in pin_odor_mixlist, and update the abbreviations (specified in
    `abbrev`) """
    fixed_odor_mixlist = copy.deepcopy(pin_odor_mixlist)

    for odor_mix in fixed_odor_mixlist:
        for pin_odor in odor_mix.pin_odor_list:
            if pin_odor.name in abbrevs.keys():
                pin_odor.abbrev = abbrevs[pin_odor.name]
    return fixed_odor_mixlist


def make_panel_info_dict(flacq, panel_id):
    return None


def pin_odor_mixlist_2_odor_dim(pin_odor_mixlist):
    # for odor_mix in pin_odor_mixlist:
    #     for pin_odor in odor_mix.pin_odor_list:
    #         pin_odor.abbrev = tom_abbrevs[pin_odor.name]

    # is_pair_ = [len(item.pin_odor_list) == 2 for item in pin_odor_mixlist]

    odor_pair_list_ = [item.as_str(round_whole_conc=True) for item in pin_odor_mixlist]

    for item in odor_pair_list_:
        if len(item) == 1:
            item.append('solvent')

    odor1_, odor2_ = zip(*odor_pair_list_)

    stim_list_ = [item.as_flat_str(round_whole_conc=True) for item in pin_odor_mixlist]
    repeat_ = ryeutils.occurrence(stim_list_)
    return odor1_, odor2_, repeat_


def init_df_activation_strength(flacq):
    """
    Makes initial dataframe containing relevant info for computing trial-by-trial activation
    strength.

    Args:
        flacq (FlatFlyAcquisitions): natural mixture dataset.
                                        see manifestors/flat_fly_acquisitions.json

    Returns:
        pd.DataFrame: contains columns ["panel", "is_pair", "date", "fly_num", "odor1", "odor2"]

    """
    odor_mixlist = pydantic.parse_file_as(List[PinOdorMixture],
                                          flacq.mov_dir().joinpath('pin_odor_mixture_list.json'))

    odor_mixlist_tomformat = fix_abbrevs(odor_mixlist, abbrevs=tom_abbrevs)
    odor1, odor2, repeat = pin_odor_mixlist_2_odor_dim(odor_mixlist_tomformat)

    df_activation_strength = pd.DataFrame(dict(panel=flacq.panel(),
                                               is_pair=flacq.is_pair(),
                                               date=np.datetime64(flacq.date_imaged, 'ns'),
                                               fly_num=flacq.fly_num,
                                               odor1=odor1,
                                               odor2=odor2))
    return df_activation_strength


def make_mi_odors_for_corr(flacq, astype='multiindex'):
    """
    Makes pd.MultiIndex for `da_corr`, a datastructure compatible with Tom's analysis code

    Args:
        flacq (FlatFlyAcquisitions): Entry from
                                    natural_mixtures/manifestos/flat_linked_thor_acquisitions.json
        astype (Union[str, None]): default value 'multiindex'; if otherwise, return as pd.DataFrame

    Returns:
        df_odor (pd.DataFrame): pd.MultiIndex or pd.DataFrame with columns
                                    ["is_pair", "odor1", "odor2", "repeat"]

        df_odor_b (pd.DataFrame): pd.MultiIndex or pd.DataFrame with columns
                                    ["is_pair_b", "odor1_b", "odor2_b", "repeat"]
                                  Should be the same as df_odor, used for computing odor-odor
                                  similarities.

    Examples:
        >>> import natmixconfig
        >>> flat_acqs = pydantic.parse_file_as(List[FlatFlyAcquisitions], natmix.MANIFEST_FILE)
        >>> df_odors, df_odors_b = make_mi_odors(flat_acqs[-1], astype='dataframe')
        >>> df_odors
            is_pair	odor1	odor2	repeat
                0	False	t2h @ -6	solvent	0
                1	False	t2h @ -6	solvent	1
                2	False	t2h @ -6	solvent	2
                3	False	3mt1p @ -6	solvent	0
                4	False	3mt1p @ -6	solvent	1


    """
    odor_mixlist = pydantic.parse_file_as(List[PinOdorMixture],
                                          flacq.mov_dir().joinpath('pin_odor_mixture_list.json'))

    fixed_odor_mixlist = fix_abbrevs(odor_mixlist, abbrevs=tom_abbrevs)

    odor1, odor2, repeat = pin_odor_mixlist_2_odor_dim(fixed_odor_mixlist)

    df_odor = pd.DataFrame(dict(is_pair=flacq.is_pair(),
                                odor1=odor1,
                                odor2=odor2,
                                repeat=repeat))

    df_odor_b = df_odor.add_suffix('_b')

    if astype == 'multiindex':
        df_odor = pd.MultiIndex.from_frame(df_odor)
        df_odor_b = pd.MultiIndex.from_frame(df_odor_b)
    return df_odor, df_odor_b


# %%

def process_toms_respvecs(da_ori_respvec):
    da_respvec = da_ori_respvec.where(da_ori_respvec.panel == 'megamat', drop=True)

    df_datefly = pd.DataFrame.from_dict(
            dict(date=da_respvec.date.to_numpy(),
                 fly_num=da_respvec.fly_num.to_numpy())
    )
    df_acq = df_datefly.drop_duplicates()
    df_acq = df_acq.reset_index(drop=True)
    df_acq['iacq'] = range(df_acq.shape[0])
    df_acq = df_acq.set_index(['date', 'fly_num'])

    datefly_tuples = [(a, b) for a, b in zip(da_respvec.date.to_numpy(),
                                             da_respvec.fly_num.to_numpy())]
    datefly = [f"{a}/{b}" for a, b in zip(da_respvec.date.to_numpy(),
                                          da_respvec.fly_num.to_numpy())]
    abbrev_list = [item.split('@')[0].strip() for item in da_respvec[
        'odor1'].to_numpy()]
    da_respvec = da_respvec.drop_vars('is_pair').drop_vars('odor2')
    da_respvec = da_respvec.assign_coords(
            iacq=('col', df_acq.loc[datefly_tuples, 'iacq'].to_numpy()),
            datefly=('col', datefly),
            abbrev=('row', abbrev_list),
    )
    return da_respvec


# %%

def make_dataframes_from_xlsx():
    file = Path("/local/matrix/Remy-Data/projects/odor_space_collab"
                "/analysis_outputs/from_tom/data_from_tom(4).xlsx")

    df_orns = pd.read_excel(file, sheet_name='orn_terminals', header=[0, 1, 2],
                            index_col=[0, 1, 2, 3, 4])
    df_orns.columns.names = ['date', 'fly_num', 'roi']
    df_pns = pd.read_excel(file, sheet_name='pn_boutons', header=[0, 1, 2],
                           index_col=[0, 1, 2, 3, 4])
    df_pns.columns.names = ['date', 'fly_num', 'roi']
    return df_orns, df_pns


def df_ori_2_dataarray(df_ori):
    # fixed_index = df_ori.index.to_frame().convert_dtypes()
    # mi_row = pd.MultiIndex.from_frame(fixed_index)
    #
    # fixed_columns = df_ori.columns.to_frame().convert_dtypes()
    # mi_col = pd.MultiIndex.from_frame(fixed_columns)

    da_ori = xr.DataArray(df_ori.to_numpy(),
                          dims=['row', 'col'],
                          coords=dict(
                                  row=df_ori.index,
                                  col=df_ori.columns
                          ))
    da_ori = da_ori.reset_index('row')
    da_ori = da_ori.reset_index('col')
    return da_ori


def fix_da_ori(da_ori):
    da_fixed = da_ori.drop_vars(['is_pair', 'odor2'])
    da_fixed = da_fixed.where(da_fixed.panel == 'megamat', drop=True)
    for coord_name in ['panel', 'odor1', 'roi']:
        da_fixed[coord_name] = da_fixed[coord_name].astype(str)
    da_fixed['repeat'] = da_fixed['repeat'].astype('int')
    da_fixed['fly_num'] = da_fixed['fly_num'].astype('int')
    return da_fixed


# %%

def convert_xlsx_to_netcdfs():
    df_orn_ori, df_pn_ori = make_dataframes_from_xlsx()

    da_orn_ori = df_ori_2_dataarray(df_orn_ori)
    da_pn_ori = df_ori_2_dataarray(df_pn_ori)

    da_orn_fixed = fix_da_ori(da_orn_ori)
    da_pn_fixed = fix_da_ori(da_pn_ori)

    da_orn_fixed.name = 'orn_terminals'
    da_pn_fixed.name = 'pn_dendrites'

    da_orn_fixed.to_netcdf(natmixconfig.NAS_PRJ_DIR.joinpath(
            'analysis_outputs',
            'by_imaging_type',
            'orn_terminals',
            'xrds_orn_terminals.nc'
    ))
    da_pn_fixed.to_netcdf(natmixconfig.NAS_PRJ_DIR.joinpath(
            'analysis_outputs',
            'by_imaging_type',
            'pn_dendrites',
            'xrds_pn_dendrites.nc'
    ))
    return da_orn_fixed, da_pn_fixed


da_orn, da_pn = convert_xlsx_to_netcdfs()

# %%
da_orn = xr.load_dataarray(natmixconfig.NAS_PRJ_DIR.joinpath(
        'analysis_outputs',
        'by_imaging_type',
        'orn_terminals',
        'xrds_orn_terminals.nc'
))
da_pn = xr.load_dataarray(natmixconfig.NAS_PRJ_DIR.joinpath(
        'analysis_outputs',
        'by_imaging_type',
        'pn_dendrites',
        'xrds_pn_dendrites.nc'
))
#%%
da_orn_trialavg = da_orn.groupby('odor1').mean().rename(dict(odor1='stim'))
da_pn_trialavg = da_pn.groupby('odor1').mean().rename(dict(odor1='stim'))
# %%
print('orn_terminals')
da_list = []
for iacq, (grp, da) in enumerate(da_orn_trialavg.set_index(col=['date', 'fly_num']).groupby('col')):
    print(f"{iacq}: {grp}")
    da_list.append(da.assign_coords(iacq=iacq).reset_index('col'))
da_orn_respvec = xr.concat(da_list, 'col')
da_orn_respvec = da_orn_respvec.assign_coords(abbrev=('stim',
                                     [item.split('@')[0].strip() for item in da_orn_respvec[
                                         'stim'].to_numpy()])
                             )
da_orn_respvec = da_orn_respvec.rename(dict(col='roiacq'))
da_orn_respvec = da_orn_respvec.transpose('roiacq', 'stim')
#%%
print('pn_dendrites')
da_list = []
for iacq, (grp, da) in enumerate(da_pn_trialavg.set_index(col=['date', 'fly_num']).groupby('col')):
    print(f"{iacq}: {grp}")
    da_list.append(da.assign_coords(iacq=iacq).reset_index('col'))
da_pn_respvec = xr.concat(da_list, 'col')
da_pn_respvec = da_pn_respvec.assign_coords(abbrev=('stim',
                                     [item.split('@')[0].strip() for item in da_pn_respvec[
                                         'stim'].to_numpy()])
                             )
da_pn_respvec = da_pn_respvec.rename(dict(col='roiacq'))
da_pn_respvec = da_pn_respvec.transpose('roiacq', 'stim')
#%%

da_orn_respvec.to_netcdf(natmixconfig.NAS_PRJ_DIR.joinpath(
        'analysis_outputs',
        'by_imaging_type',
        'orn_terminals',
        'megamat',
        'xrds_orn_terminals__megamat__respvec_agg.nc'
))

da_pn_respvec.to_netcdf(natmixconfig.NAS_PRJ_DIR.joinpath(
        'analysis_outputs',
        'by_imaging_type',
        'pn_dendrites',
        'megamat',
        'xrds_pn_dendrites__megamat__respvec_agg.nc'
))

#%%
ufunc_names = [None, 'robust_scale', 'quantile_transform', 'scale']
ufunc_kwargs_lookup = {
    'robust_scale': dict(with_centering=True,
                         with_scaling=True,
                         quantile_range=(2.5, 97.5)
                         ),
    'quantile_transform': dict(axis=0, n_quantiles=10,
                               output_distribution='normal'),
    'scale': None
}

#%%
respvec_data = dict(
        orn_terminals=da_orn_respvec,
        pn_dendrites=da_pn_respvec
                    )
#%%
stdized_respvecs_by_region = {}

for imaging_type in ['orn_terminals', 'pn_dendrites']:

    stdized_respvecs = {}

    for ufunc_name in ufunc_names:
        da_stdized_list = []

        for iacq, da in respvec_data[imaging_type].groupby('iacq'):
            if ufunc_name is not None:
                ufunc = getattr(preprocessing, ufunc_name)
                ufunc_kwargs = ufunc_kwargs_lookup[ufunc_name]

                da_standardized = xr.apply_ufunc(
                        ufunc,
                        da,
                        input_core_dims=[['roiacq', 'stim']],
                        output_core_dims=[['roiacq', 'stim']],
                        # vectorize=True,
                        keep_attrs=True,
                        kwargs=ufunc_kwargs
                )
            else:
                da_standardized = da.transpose('roiacq', 'stim')

            da_standardized['roiacq'] = da['roiacq']
            da_stdized_list.append(da_standardized)

        da_respvec_cat = xr.concat(da_stdized_list, 'roiacq')
        attrs = dict(imaging_type='pn_dendrites',
                     ufunc=str(ufunc_name)
                     )
        da_respvec_cat = da_respvec_cat.assign_attrs(attrs)
        stdized_respvecs[ufunc_name] = da_respvec_cat
    stdized_respvecs_by_region[imaging_type] = stdized_respvecs
#%%
for imaging_type in ['orn_terminals', 'pn_dendrites']:
    SAVE_DIR = natmixconfig.NAS_PRJ_DIR.joinpath(
        'analysis_outputs',
        'by_imaging_type',
        imaging_type,
        'megamat',
        'respvec_concat')
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for ufunc_name in ufunc_names:
        filename = f"xrds_{imaging_type}__megamat__respvec_agg__stdize_{ufunc_name}.nc"
        stdized_respvecs_by_region[imaging_type][ufunc_name].to_netcdf(
                SAVE_DIR.joinpath(filename)
        )
    #%%
# def make_preprocessed_tom_respvecs():
#     da_stdized_list = []
#
#
#     for imaging_type in ['orn_terminals', 'pn_boutons']:
#         da_respvec_agg =
#         SAVE_DIR = natmixconfig.NAS_PRJ_DIR.joinpath(
#                 'analysis_outputs',
#                 'by_imaging_type',
#                 'orn_'
#         )
#
#     ds_stdized_list = []
#     for ufunc_name in ufunc_names:
#         if ufunc_name is not None:
#             ufunc = getattr(preprocessing, ufunc_name)
#             ufunc_kwargs = ufunc_kwargs_lookup[ufunc_name]
#
#             ds_standardized = xr.apply_ufunc(
#                     ufunc,
#                     ds,
#                     input_core_dims=[['glomeruli', 'odors']],
#                     output_core_dims=[['glomeruli', 'odors']],
#                     # vectorize=True,
#                     keep_attrs=True,
#                     kwargs=ufunc_kwargs
#             )
#         else:
#             ds_standardized = ds.transpose('glomeruli', 'odors')
#
#         abbrev_ord = ['1-5ol', '1-6ol', '1-8ol', '2-but', '2h', '6al', 'B-cit', 'IaA',
#                               'Lin',
#                         'aa', 'benz', 'eb', 'ep', 'ms', 'pa', 't2h', 'va']
#
#         hallem_ord = ['1-pentanol', '1-hexanol', '1-octanol', '2-butanone', '2-heptanone',
#                         'hexanal', 'b-citronellol', 'isopentyl acetate', 'linalool', 'acetic acid',
#                         'benzaldehyde', 'ethyl butyrate', 'ethyl propionate', 'methyl salicylate',
#                         'pentyl acetate', 'E2-hexenal', 'pentanoic acid']
#         # add attrs
#         attrs = dict(abbrev_ord=abbrev_ord, hallem_ord=hallem_ord, ufunc=str(ufunc_name))
#         ds_standardized = ds_standardized.assign_attrs(attrs)
#
#         ds_standardized_megamat = ds_standardized.sel(odors=hallem_ord)
#         ds_standardized_megamat = ds_standardized_megamat.assign_coords(
#                 dict(abbrev=('odors', abbrev_ord))
#         )
#
#         # save datasets
#         filename = f"xrds_hallem_respvec__all_odors__stdize_{ufunc_name}.nc"
#         ds_standardized.to_netcdf(SAVE_DIR.joinpath(filename))
#
#         filename = f"xrds_hallem_respvec__megamat__stdize_{ufunc_name}.nc"
#         ds_standardized_megamat.to_netcdf(SAVE_DIR.joinpath(filename))
#
#     return ds_stdized_list
