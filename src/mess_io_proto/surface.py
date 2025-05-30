"""Read MESS file format."""

import itertools
from pathlib import Path
from typing import Annotated

import networkx
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    model_validator,
)

from .util import MessBlockParseData, mess


class Well(BaseModel):
    id: int
    energy: float


class UnimolWell(Well):
    name: str


class NMolWell(Well):
    names: Annotated[list[str], AfterValidator(sorted)]
    interacting: bool = False
    fake: bool = False


class Barrier(BaseModel):
    well_ids: Annotated[tuple[int, int], AfterValidator(lambda x: tuple(sorted(x)))]
    name: str
    energy: float
    fake: bool = False


class Surface(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    wells: list[Well]
    barriers: list[Barrier]

    @model_validator(mode="after")
    def _validate_ids(self):
        # Validate well IDs
        well_ids = [n.id for n in self.wells]
        well_id_set = set(well_ids)
        if not len(well_ids) == len(well_id_set):
            raise ValueError(f"Repeated well IDs: {well_ids}")

        # Validate barrier well IDs
        barrier_well_ids_lst = [e.well_ids for e in self.barriers]
        barrier_well_id_set = set(itertools.chain.from_iterable(barrier_well_ids_lst))
        if not barrier_well_id_set <= well_id_set:
            raise ValueError(f"Undefined well IDs for barriers: {barrier_well_ids_lst}")

        return self


def from_mess(mess_inp: str | Path) -> Surface:
    """Read surface.

    :param mess_inp: MESS input
    :return: Surface
    """
    all_datas = mess.parse_blocks(mess_inp)
    well_data = [d for d in all_datas if d.type in ["Well", "Bimolecular"]]
    barrier_data = [d for d in all_datas if d.type == "Barrier"]

    id_dct = {d.label: i for i, d in enumerate(well_data)}

    wells = [well_from_mess_block_parse_data(d, id_dct) for d in well_data]
    barriers = [barrier_from_mess_lock_parse_data(d, id_dct) for d in barrier_data]

    return Surface(wells=wells, barriers=barriers)


def graph(surf: Surface) -> networkx.MultiGraph:
    """Generate NetworkX graph."""
    net = networkx.MultiGraph()
    net.add_nodes_from([(w.id, w.model_dump()) for w in surf.wells])
    net.add_edges_from([(*b.well_ids, b.model_dump()) for b in surf.barriers])
    return net


# Helpers
def well_from_mess_block_parse_data(
    data: MessBlockParseData, id_dct: dict[str, int]
) -> Well:
    """Generate Well object from block.

    :param id_dct: Dictionary mapping labels to IDs
    :return: Well
    """
    if data.type == "Barrier":
        raise ValueError("Cannot create well object from barrier block.")

    assert data.label in id_dct, f"{data.label} not in {id_dct}"
    id_ = id_dct.get(data.label)

    if data.type == "Bimolecular":
        names = data.label.split("+")
        return NMolWell(
            id=id_, energy=data.energy, names=names, interacting=False, fake=False
        )

    if data.label.startswith("FakeW-"):
        names = data.label.removeprefix("FakeW-").split("+")
        return NMolWell(
            id=id_, energy=data.energy, names=names, interacting=True, fake=True
        )

    assert data.type == "Well"
    return UnimolWell(id=id_, energy=data.energy, name=data.label)


def barrier_from_mess_lock_parse_data(
    data: MessBlockParseData, id_dct: dict[str, int]
) -> Barrier:
    """Generate Barrier object from block.

    :param block_data:
    :param id_dct: Dictionary mapping labels to IDs
    :return: Barrier
    """
    if not data.type == "Barrier":
        raise ValueError("Cannot create barrier object from non-barrier block.")

    name, *well_labels = data.label.split()
    fake = name.startswith("FakeB-")
    assert all(
        label in id_dct for label in well_labels
    ), f"{well_labels} not in {id_dct}"
    well_ids = list(map(id_dct.get, well_labels))
    return Barrier(well_ids=well_ids, name=name, energy=data.energy, fake=fake)
