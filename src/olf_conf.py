""" Functions for parsing .yaml files in natural_mixtures/olfactometer_configs"""
from itertools import chain
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

from pydantic_models import PinOdor

OLF_PROC_DIR = Path("/local/storage/Remy/natural_mixtures/olfactometer_configs")
relative_file_path = True
balance_pins = {2, 42}

abbreviations = {'kiwi approx.': 'kiwi',
                 'ethyl acetate': 'ea',
                 'ethyl butyrate': 'eb',
                 'isoamyl alcohol': 'IaOH',
                 'isoamyl acetate': 'IaA',
                 'ethanol': 'EtOH'}


def load_olf_config(yaml_file, relative_path=relative_file_path):
    """ Loads olfactometer config files. """
    if '\\' in yaml_file:
        yaml_file = Path(*yaml_file.split('\\'))
    if relative_path:
        yaml_file = OLF_PROC_DIR.joinpath(yaml_file)

    with open(yaml_file, 'r') as f:
        olf_config = yaml.safe_load(f)

    return olf_config


# %%

def parse_pin_odors(olf_config, lookup_abbrev=True):
    """ Converts olf_config['pins2odors'] into a dictionary mapping {<pin #> : PinOdor}"""
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
    """ Returns unique PinOdor instances from a flat list.

    Examples:
        >>>> from itertools import chain
        >>>> flat_stim_list = list(chain(*stim_list))
        >>>> u_pin_odors = get_unique_pin_odors(flat_stim_list)

    Args:
        flat_stim_list:

    Returns:
        unique_pin_odors ( List[PinOdor] )

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

    Examples:
        >>>> name2conc = get_name_to_log10conc(flat_stim_list)
        >>>> print(name2conc)

        {'ethanol': [-2.0],
         'ethyl acetate': [-4.2],
         'ethyl butyrate': [-3.5],
         'isoamyl acetate': [-3.7],
         'isoamyl alcohol': [-3.6],
         'kiwi approx.': [0.0, -1.0, -2.0],
         'pfo': [0.0] }

    """
    u_pin_odors = get_unique_pin_odors(flat_stim_list)

    odor_names = list({item.name for item in flat_stim_list})
    d = {}
    for name in odor_names:
        d[name] = list({item.log10_conc for item in u_pin_odors if item.name == name})
    return d


def pin_odors_to_dataframe(pin_odors):
    """
    Converts a list(list(PinOdor)) into a dataframe with odor names as columns, and the concentration of each odor
    (if delivered) for each stimulus trial.
    """

    odor_names = list({item.name for item in chain(*pin_odors)})

    n_stim, n_channels = np.array(pin_odors).shape
    df_stimuli = pd.DataFrame([[None] * len(odor_names)] * n_stim, columns=sorted(odor_names))

    for i, odors in enumerate(pin_odors):
        for odor in odors:
            if odor.log10_conc is not None:
                df_stimuli.loc[i, odor.name] = odor.log10_conc
    return df_stimuli


def pins2odors_to_dataframe(olf_config: dict):
    """Converts olf_config['pins2odors'] into a dataframe, with pin # as the index."""
    df = pd.DataFrame(list(olf_config['pins2odors'].values()),
                      index=olf_config['pins2odors'].keys())
    df = df.loc[:, ['name', 'log10_conc']]
    return df


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

# odor_names = sorted(list({item.name for item in pins2odors.values()}))


# %%
# pd.DataFrame([tuple(item.dict().values()) for item in flat_stim_list],
#             columns=['name', 'conc', 'abbrev'])

# olf_config = load_olf_config(flat_acqs[0].olf_config)
# pins2odors = parse_pin_odors(olf_config)
