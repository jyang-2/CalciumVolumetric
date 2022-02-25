# code for parsing information in natural_mixtures/manifestos

from typing import List

import pydantic
import yaml

from pydantic_models import FlyWithAcquisitions, FlatFlyAcquisitions


def load_acquisitions_by_fly(lacq_file=None):
    """
    Loads list of linked thor acquisitions from yaml file

    Args:
        lacq_file: path to linked_thor_acquisitions.yaml

    Returns:
        linked_acquisitions ( List[FlyWithAcquisitions] )
    """
    linked_acquisitions = load_linked_acquisitions(lacq_file=None)
    linked_acquisitions = pydantic.parse_obj_as(List[FlyWithAcquisitions], linked_acquisitions)
    return linked_acquisitions


def load_linked_acquisitions(lacq_file=None):
    """ Load data in 'linked_thor_acquisitions.yaml' """
    if lacq_file is None:
        lacq_file = "/local/storage/Remy/natural_mixtures/manifestos/linked_thor_acquisitions.yaml"
    with open(lacq_file, 'r') as f:
        linked_acquisitions = yaml.safe_load(f)
    return linked_acquisitions


def flatten_linked_acquisitions(linked_acquisitions: list):
    """Flattens nested linked acquisition dicts"""

    flat_thor_acquisitions = []
    for fly_acq in linked_acquisitions:
        for lacq in fly_acq['linked_thor_acquisitions']:
            flat_acq = dict(date_imaged=fly_acq['date_imaged'], fly_num=fly_acq['fly_num'])
            for k, v in lacq.items():
                flat_acq[k] = v
            flat_thor_acquisitions.append(flat_acq)
    return flat_thor_acquisitions


def load_flat_fly_acquisitions(lacq_file=None):
    """
    Loads and parses information in linked_thor_acquisitions.yaml to a flat structure

    Args:
        lacq_file: path to linked_thor_acquisitions.yaml

    Returns:
        flat_acqs ( List[FlatFlyAcquisitions] )
    """
    lacq_list = flatten_linked_acquisitions(load_linked_acquisitions(lacq_file))
    flat_acqs = pydantic.parse_obj_as(List[FlatFlyAcquisitions], lacq_list)
    return flat_acqs


def main():
    # fly_acqs = load_acquisitions_by_fly()
    # flat_acqs = load_flat_fly_acquisitions()
    # return fly_acqs, flat_acqs
    return load_acquisitions_by_fly(), load_flat_fly_acquisitions()


if __name__ == '__main__':
    fly_acqs, flat_acqs = main()
