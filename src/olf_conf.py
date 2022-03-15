""" Functions for parsing .yaml files in natural_mixtures/olfactometer_configs

Attributes:
        OLF_PROC_DIR (Path):
"""
from itertools import chain
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import yaml

from pydantic_models import PinOdor

OLF_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/olfactometer_configs")
"""Path: path to directory holding olfactometer config files."""

relative_file_path = True
"""boolean: whether or not files should be loaded relative to OLF_PROC_DIR"""

balance_pins = {2, 42}
"""set: pins used as flow balance pins on olfactometer (discount as an odor stimulus) """

abbreviations = {'kiwi approx.': 'kiwi',
                 'ethyl acetate': 'ea',
                 'ethyl butyrate': 'eb',
                 'isoamyl alcohol': 'IaOH',
                 'isoamyl acetate': 'IaA',
                 'ethanol': 'EtOH'}
"""dict: odor name abbreviations"""


def load_olf_config(yaml_file, relative_path=relative_file_path):
    """ Loads olfactometer config files as dict.

     Args:
        yaml_file (Union[Path, str]): config yaml filename
        relative_path (bool): whether to load relative to OLF_PROC_DIR

    Returns:
        olf_config (dict): dict with keys ['pin_sequence', 'pins2odors', 'settings']
     """
    if '\\' in yaml_file:
        yaml_file = Path(*yaml_file.split('\\'))
    if relative_path:
        yaml_file = OLF_PROC_DIR.joinpath(yaml_file)

    with open(yaml_file, 'r') as f:
        olf_config = yaml.safe_load(f)

    return olf_config


def parse_pin_odors(olf_config, lookup_abbrev=True):
    """ Converts olf_config['pins2odors']  into a dictionary mapping {<pin #> : PinOdor}

    Args:
        olf_config (dict): dictionary loaded from olfactometer config .yaml
        lookup_abbrev (bool): whether to lookup odor abbreviations from `abbreviations`

    Returns:
        pins2odors (dict): key = pin #, value = PinOdor(...)

    Examples::
        olf_config = olf_conf.load_olf_config('20220209_210002_stimuli.yaml')\n
        pins2odors = olf_conf.parse_pin_odors(olf_config)\

    """
    pins2odors = {k: PinOdor.parse_obj(v) for k, v in olf_config['pins2odors'].items()}
    if lookup_abbrev:
        for k, v in pins2odors.items():
            if v.name in abbreviations.keys():
                v.abbrev = abbreviations[v.name]

    return pins2odors


def pin_sequence_as_list(olf_config):
    """Parses pins in olf_config['pin_sequence'], and returns as a list of lists, with balance_pins removed."""
    pin_groups = olf_config['pin_sequence']['pin_groups']
    pin_list = [item['pins'] for item in pin_groups]
    pin_list = [[pin for pin in pin_grp if pin not in balance_pins] for pin_grp in pin_list]
    return pin_list


def pin_list_to_odors(pin_list: List[List[int]], pins2odors: dict):
    """Maps list(list[pins]] intp List[List[PinOdors]]"""
    stim_list = [[pins2odors[pin] for pin in pins if pin not in balance_pins] for pins in pin_list]
    return stim_list


def get_unique_pin_odors(flat_stim_list):
    """
    Returns unique PinOdor instances from a flat list.

    Args:
        flat_stim_list (List[PinOdor])

    Returns:
        unique_pin_odors (List[PinOdor]) :


    Examples::

    >>>> from itertools import chain
    >>>> flat_stim_list = list(chain(*stim_list))
    >>>> u_pin_odors = get_unique_pin_odors(flat_stim_list)

    """
    unique_odors = {item.json() for item in flat_stim_list}
    unique_odors = list(unique_odors)

    unique_pin_odors = [PinOdor.parse_raw(item) for item in unique_odors]
    return unique_pin_odors


def get_name_to_log10conc(flat_stim_list):
    """
    Returns dict mapping { <odor name> : <list of concentrations used> }

    Args:
        flat_stim_list (List[PinOdor]): use `flat_stim_list = list(chain(*stim_list))`

    Returns:
        odor2conc (dict)

    Examples::
    name2conc = get_name_to_log10conc(flat_stim_list)

    """
    u_pin_odors = get_unique_pin_odors(flat_stim_list)

    odor_names = list({item.name for item in flat_stim_list})
    d = {}
    for name in odor_names:
        d[name] = list({item.log10_conc for item in u_pin_odors if item.name == name})
    return d


def pin_odors_to_dataframe(pin_odors):
    """
    Converts nested lists of PinOdor items into a dataframe w/ odor names as columns, and concentrations of each odor
    (if delivered) for each stimulus trial.

    Args:
        pin_odors (List[List[PinOdor]]): get w/ function parse_pin_odors(...)

    Returns:
        df_pins2odors (pd.DataFrame):
    """

    odor_names = list({item.name for item in chain(*pin_odors)})

    n_stim, n_channels = np.array(pin_odors).shape
    df_stimuli = pd.DataFrame([[None] * len(odor_names)] * n_stim, columns=sorted(odor_names))

    for i, odors in enumerate(pin_odors):
        for odor in odors:
            if odor.log10_conc is None:
                df_stimuli.loc[i, odor.name] = float('-inf')
            #if odor.log10_conc is not None:
            else:
                df_stimuli.loc[i, odor.name] = odor.log10_conc
    return df_stimuli


def pins2odors_to_dataframe(olf_config):
    """Converts olf_config['pins2odors'] into a dataframe, with pin # as the index.

    Args:
        olf_config (dict) :
            loaded from olfactometer config .yaml file \n
            olf_conf.load_olf_config('20220209_210002_stimuli.yaml')

    Returns:
        df (pd.DataFrame): table w/ columns = ['name', 'log10_conc'], index = [{pin #s}]

    Examples::
                      name   log10_conc
            --------------- -----------
        37  isoamyl alcohol        -3.6 \n
        38   ethyl butyrate        -3.5 \n
        39     kiwi approx.        -1.0 \n
        40              pfo         0.0 \n
        41     kiwi approx.        -2.0 \n
        49     kiwi approx.         0.0 \n
        51          ethanol        -2.0 \n
        52    ethyl acetate        -4.2 \n
        53  isoamyl acetate        -3.7 \n

    """
    df = pd.DataFrame(list(olf_config['pins2odors'].values()),
                      index=olf_config['pins2odors'].keys())
    df = df.loc[:, ['name', 'log10_conc']]
    return df


def remove_null_conc(pin_odor_list):
    """ Returns pin odors non-null log10_conc values"""
    return list(filter(lambda x: x.log10_conc is not None, pin_odor_list))

# %%
def main(config_yaml):
    olf_config = load_olf_config(config_yaml)

    pins2odors = parse_pin_odors(olf_config)
    pin_list = pin_sequence_as_list(olf_config)
    pin_odors = pin_list_to_odors(pin_list, pins2odors)

    df_pins2odors = pins2odors_to_dataframe(olf_config)
    df_stimuli = pin_odors_to_dataframe(pin_odors)
    return olf_config, pins2odors, pin_list, pin_odors, df_pins2odors, df_stimuli


# %%

if __name__ == "__main__":
    import manifestos
    fly_acqs, flat_acqs = manifestos.main()
    for lacq in flat_acqs:
        olf_config, pins2odors, pin_list, pin_odors, df_pins2odors, df_stimuli = main(lacq.olf_config)

        SAVE_DIR = NAS_PROC_DIR.joinpath(lacq.date_imaged, str(lacq.fly_num), lacq.thorimage)

        print('')
        print(SAVE_DIR)
        print(lacq.olf_config)
        df_stimuli.to_csv(SAVE_DIR.joinpath("df_stimuli.csv"))
        df_pins2odors.to_csv(SAVE_DIR.joinpath('df_pins2odors.csv'))
        print('csv files saved successfully.')


