import typing
from typing import List

import pydantic


class Fly(pydantic.BaseModel):
    date_imaged: str
    fly_num: int
    genotype: str
    date_eclosed: str
    sex: str


class ThorImage(pydantic.BaseModel):
    date_time: str
    utime: int
    name: str
    ori_path: str
    rel_path: str


class ThorSync(pydantic.BaseModel):
    name: str
    ori_path: str
    rel_path: str


class FlatFlyAcquisitions(pydantic.BaseModel):
    date_imaged: str = pydantic.Field(...)
    fly_num: int = pydantic.Field(...)
    thorimage: str = pydantic.Field(...)
    thorsync: str = pydantic.Field(...)
    olf_config: str = pydantic.Field(...)


class LinkedThorAcquisition(pydantic.BaseModel):
    """ Holds info for looking up fly, thorsync, and thorimage file ids"""
    thorimage: str = pydantic.Field(...)
    thorsync: str = pydantic.Field(...)
    olf_config: str = pydantic.Field(...)
    notes: typing.Optional[str]


class FlyWithAcquisitions(pydantic.BaseModel):
    """ Fly info, including list of linked thor acquisitions.

    Used to parse data in {NAS_PRJ_DIR}/manifestos/linked_thor_acquisitions.yaml

    Example:
    -------

        # load linked thorlabs acquisitions (by fly) from yaml manifest
        with open(NAS_PRJ_DIR.joinpath("manifestos/linked_thor_acquisitions.yaml"), 'r') as f:
            linked_acq = yaml.safe_load(f)

        # parse w/ pydantic model
        flies_with_acq = [FlyWithAcquisitions(**item) for item in linked_acq]

    """
    date_imaged: str = pydantic.Field(...)
    fly_num: int = pydantic.Field(...)
    linked_thor_acquisitions: typing.List[LinkedThorAcquisition]


class PinOdor(pydantic.BaseModel):
    name: str
    log10_conc: typing.Optional[float]
    abbrev: typing.Optional[str]

    def str(self, use_abbrev=True):
        if use_abbrev and self.abbrev is not None:
            return f"{self.abbrev} @ {self.log10_conc}"
        else:
            return f"{self.name} @ {self.log10_conc}"

    def as_name_tuple(self):
        return self.name, self.log10_conc

    def as_abbrev_tuple(self):
        return self.abbrev, self.log10_conc

    def __eq__(self, other):
        return (self.name == other.name) & (self.log10_conc == other.log10_conc)


class PinOdorMixture(pydantic.BaseModel):
    """Model for dealing with multi-component pin mixtures (i.e. lists of PinOdor(...) instances)"""
    pin_odor_list: typing.List[PinOdor]
    order: typing.Optional[typing.Union[typing.List[str], typing.List[PinOdor]]] = []

    # n_components : Optional[int] = 0

    @pydantic.validator('pin_odor_list', pre=True, each_item=True)
    def is_list_pin_odors(cls, v):
        if isinstance(v, dict):
            return PinOdor(**v)
        else:
            return v

    # @pydantic.validator('pin_odor_list', pre=False, always=True)
    # def filter_pin_odors(cls, v, values):
    #     """ Drop odors if log10_conc is None"""
    #     filtered_pin_odors = list(filter(lambda x: ~((x.log10_conc is None) or np.isnan(x.log10_conc)), v))
    #
    #     if len(filtered_pin_odors) == 0:
    #         filtered_pin_odors = [PinOdor(name='paraffin', log10_conc=0, abbrev='pfo')]
    #
    #     return filtered_pin_odors

    def as_tuple(self, use_abbrev=True):
        """ Returns each item in self.pin_odor_list as (abbrev, log10_conc)

        Examples:
            >>>> pin_odors = [PinOdor(name='ethyl acetate', log10_conc=-6.2, abbrev='ea'),\
                                PinOdor(name='ethyl butyrate', log10_conc=-5.5, abbrev='eb')]

            >>>> bin_mix = PinOdorMixture(pin_odor_list=pin_odors)

            >>>> print(bin_mix.as_tuple())

                    [('ea', -6.2), ('eb', -5.5)]

        """
        if use_abbrev:
            return [(item.abbrev, item.log10_conc) for item in self.pin_odor_list]
        else:
            return [(item.name, item.log10_conc) for item in self.pin_odor_list]

    def as_str(self, use_abbrev=True):
        """
        Returns:
            List[str] : ['ea @ -6.2', 'eb @ -5.5', ...] if abbrev=False
        """

        return [x.str(use_abbrev=use_abbrev) for x in self.pin_odor_list]

    def as_flat_str(self, use_abbrev=True):
        """ Converts odor mixture into readable string, for plotting stimulus label text

        Returns:
            str :

        """
        if len(self.pin_odor_list) == 1:
            return self.pin_odor_list[0].str(use_abbrev=use_abbrev)
        else:
            return ", ".join([item.str(use_abbrev=use_abbrev) for item in self.pin_odor_list])

    def __eq__(self, other):
        return set(self.as_tuple()) == set(other.as_tuple())


class PinOdorListList(pydantic.BaseModel):
    """ Class for generating nested json files for PinOdors"""
    __root__: List[List[PinOdor]]


class PinOdorMixtureList(pydantic.BaseModel):
    """ Class for generating nested json files for PinOdorMixtures"""
    __root__: List[PinOdorMixture]
