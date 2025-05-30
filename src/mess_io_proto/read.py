"""Read MESS file format."""

import itertools
import re
from pathlib import Path
from typing import Annotated, Literal

import pyparsing as pp
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    model_validator,
)
from pyparsing import pyparsing_common as ppc


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


def surface(mess_inp: str | Path) -> Surface:
    """Read surface.

    :param mess_inp: MESS input
    :return: Surface
    """
    all_blocks = blocks(mess_inp)
    well_blocks = [b for b in all_blocks if b.type in ["Well", "Bimolecular"]]
    barrier_blocks = [b for b in all_blocks if b.type == "Barrier"]

    id_dct = {block.label: id_ for id_, block in enumerate(well_blocks)}

    wells = [block.well_object(id_dct) for block in well_blocks]
    barriers = [block.barrier_object(id_dct) for block in barrier_blocks]

    return Surface(wells=wells, barriers=barriers)


COMMENT = pp.Literal("!") + pp.rest_of_line()
END_FILE = pp.Keyword("End") + pp.StringEnd()

BLOCK_START = pp.Keyword("Well") | pp.Keyword("Bimolecular") | pp.Keyword("Barrier")
BLOCK_END = pp.Keyword("End") + pp.FollowedBy(BLOCK_START | END_FILE)
FILE_HEADER = pp.SkipTo(BLOCK_START)


class Block(BaseModel):
    type: Literal["Well", "Bimolecular", "Barrier"]
    label: str
    energy: float
    energy_unit: str
    contents: str

    def well_object(self, id_dct: dict[str, int]) -> Well:
        """Generate Well object from block.

        :param id_dct: Dictionary mapping labels to IDs
        :return: Well
        """
        if self.type == "Barrier":
            raise ValueError("Cannot create well object from barrier block.")

        assert self.label in id_dct, f"{self.label} not in {id_dct}"
        id_ = id_dct.get(self.label)

        if self.type == "Bimolecular":
            names = self.label.split("+")
            return NMolWell(
                id=id_, energy=self.energy, names=names, interacting=False, fake=False
            )

        if self.label.startswith("FakeW-"):
            names = self.label.removeprefix("FakeW-").split("+")
            return NMolWell(
                id=id_, energy=self.energy, names=names, interacting=True, fake=True
            )

        assert self.type == "Well"
        return UnimolWell(id=id_, energy=self.energy, name=self.label)

    def barrier_object(self, id_dct: dict[str, int]) -> Barrier:
        """Generate Barrier object from block.

        :param id_dct: Dictionary mapping labels to IDs
        :return: Barrier
        """
        if not self.type == "Barrier":
            raise ValueError("Cannot create barrier object from non-barrier block.")

        name, *well_labels = self.label.split()
        fake = name.startswith("FakeB-")
        assert all(
            label in id_dct for label in well_labels
        ), f"{well_labels} not in {id_dct}"
        well_ids = list(map(id_dct.get, well_labels))
        return Barrier(well_ids=well_ids, name=name, energy=self.energy, fake=fake)


def blocks(mess_inp: str | Path) -> list[Block]:
    """Read well/bimol/barrier blocks.

    :param mess_inp: MESS input
    :return: A list of type, contents tuples for each block
    """
    mess_inp = mess_inp.read_text() if isinstance(mess_inp, Path) else mess_inp

    block = BLOCK_START("type") + pp.SkipTo(BLOCK_END)("contents") + BLOCK_END

    expr = pp.Suppress(FILE_HEADER) + pp.OneOrMore(pp.Group(block))
    expr.ignore(COMMENT)

    block_lst = []
    for res in expr.parse_string(mess_inp):
        block = _parse_block(res.get("type"), res.get("contents"))
        block_lst.append(block)

    return block_lst


# Helpers
class Key:
    unit = "unit"
    energy = "energy"


UNIT = pp.Suppress("[") + pp.Word(pp.alphas + "/")(Key.unit) + pp.Suppress("]")
ZERO_ENERGY = pp.Literal("ZeroEnergy") + UNIT + ppc.number(Key.energy)
GROUND_ENERGY = pp.Literal("GroundEnergy") + UNIT + ppc.number(Key.energy)


def _parse_block(
    type: Literal["Well", "Bimolecular", "Barrier"], contents: str
) -> Block:
    """Read an individual block.

    :param type: Block type
    :param contents: Block contents
    :return: Block
    """
    # Parse header
    label, *_ = re.split("!|\n", contents, maxsplit=1)
    # Parse energy
    expr = ... + (GROUND_ENERGY if type == "Bimolecular" else ZERO_ENERGY)
    res = expr.parse_string(contents)

    return Block(
        type=type,
        label=label.strip(),
        energy=res.get(Key.energy),
        energy_unit=res.get(Key.unit),
        contents=contents,
    )


def fake_well_component_names(name: str) -> None | list[str]:
    """Get sorted list of component names from fake well name.

    :param name: Fake well name
    :return: Component names
    """
    if not name.startswith("FakeW-"):
        return None

    return sorted(name.removeprefix("FakeW-").split("+"))
