"""
Useful functions/classes for aggregating data from multiple FlatFlyAcquisitions.




"""

import copy
from typing import List

import pydantic

from pydantic_models import FlatFlyAcquisitions


class PanelMovies(pydantic.BaseModel):
    """A class for specifying movie types for different odor panels.

    For more easily filtering/grouping FlatFlyAcquisitions. Use with `filter_flacq_list`.

    An odor panel is a grouping of odor stimuli that are presented together. Sometimes multiple
    olfactometer configs exist for a given odor panel (i.e. there are a core group of common
    odors in different configs, but sometimes there are extra probe odors).

    Thus, a `movie` refers to a unique olfactometer config, and each `panel` can have multiple
    `movies`.

    Attributes:
        prj (str): the project name (either 'natural_mixtures', 'odor_space_collab',
                   or 'odor_unpredictability)
        panel (str): name of the odor panel
        movies (List[str]): list of movie names
        _allowed_projects (List[str]): list of allowed project names
        _allowed_movie_types_by_panel (Dict[str, List[str]]): dictionary of allowed movie types
            by panel type
        _allowed_panels_by_project (Dict[str, List[str]]): dictionary of allowed panel types by
            project.


    Examples:
        >>> megamat_panel = aggregate.PanelMovies(prj='odor_space_collab', panel='megamat')
        >>> megamat_panel
        PanelMovies(prj='odor_space_collab', panel='megamat', movies=['megamat0', 'megamat1'])

        >>> panel.movies
        ['megamat0', 'megamat1']

    """
    # Class attributes
    _allowed_projects = ['natural_mixtures',
                         'odor_space_collab',
                         'odor_unpredictability']

    _allowed_panels_by_project = {
        'natural_mixtures': ['kiwi', 'control'],
        'odor_space_collab': ['validation', 'megamat', 'odorspace'],
        'odor_unpredictability': ['cyanide']
    }
    _allowed_movie_types_by_panel = {
        'kiwi': ['kiwi',
                 'kiwi_components_again',
                 'kiwi_components_again_with_partial',
                 'kiwi_components_again_with_partial_and_probes'],
        'control': ['control1',
                    'control1_components_again_with_partial_and_probes'],
        'validation': ['validation0', 'validation1'],
        'megamat': ['megamat0', 'megamat1'],
        'odorspace': ['odorspace0', 'odorspace1'],
        'cyanide': ['cyanide']
    }
    prj: str
    panel: str
    movies: List[str]

    class Config:
        validate_assignment = True

    # if movies is None, then set movies to the default list of movies for the panel
    @pydantic.root_validator(pre=True, allow_reuse=True)
    def set_movies(cls, values):
        if values.get('movies') is None:
            values['movies'] = cls._allowed_movie_types_by_panel[values['panel']]
        return values


def filter_flacq_list(flacq_list, allowed_movie_types=None, has_s2p_output=True,
                      allowed_imaging_type=None):
    """Filter a list of FlatFlyAcquisitions.

    Args:
        allowed_imaging_type (Union[str, None]): kc_soma, pn_boutons, kc_dendrites, or None (
            default). If None, then all imaging types are allowed.
        flacq_list (List[FlatFlyAcquisition]): the FlatFlyAcquisitions to filter
        allowed_movie_types (List[str]): the allowed movie types (e.g. ['megamat0', 'megamat1'])
        has_s2p_output (bool): whether the FlatFlyAcquisitions must have suite2p output
                               i.e. whether s2p_stat_file must be defined

    Returns:
        flacq_list_ (List[FlatFlyAcquisition]): FlatFlyAcquisitions list, with specified
            movie types and filtered by s2p_output.

    Examples:
        >>> flat_acqs = pydantic.parse_file_as(List[FlatFlyAcquisitions], natmixconfig.MANIFEST_FILE)
        >>> megamat_panel = aggregate.PanelMovies(prj='odor_space_collab', panel='megamat')
        >>> filtered_flat_acqs = filter_flacq_list(flat_acqs, allowed_movie_types=megamat_panel.movies)
    """
    flacq_list_ = copy.deepcopy(flacq_list)

    if allowed_movie_types is not None:
        flacq_list_ = list(filter(lambda x: x.movie_type in allowed_movie_types, flacq_list_))
    if allowed_imaging_type is not None:
        flacq_list_ = list(filter(lambda x: x.imaging_type == allowed_imaging_type, flacq_list_))
    if has_s2p_output:
        flacq_list_ = list(filter(lambda x: x.stat_file() is not None, flacq_list_))

    return flacq_list_
