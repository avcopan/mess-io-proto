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


class Node(BaseModel):
    id: int
    energy: float


class UnimolNode(Node):
    name: str


class NMolNode(Node):
    names: list[str]
    complex: bool = False
    fake: bool = False


class Edge:
    node_ids: Annotated[tuple[int, int], AfterValidator(lambda x: tuple(sorted(x)))]
    name: str
    energy: float
    fake: bool = False


class Network(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: list[Node]
    edges: list[Edge]

    @model_validator(mode="after")
    def _validate_ids(self):
        # Validate node IDs
        node_ids = [n.id for n in self.nodes]
        node_id_set = set(node_ids)
        if not len(node_ids) == len(node_id_set):
            raise ValueError(f"Repeated node IDs: {node_ids}")

        # Validate edge IDs
        edge_node_ids_lst = [e.node_ids for e in self.edges]
        edge_node_id_set = set(itertools.chain.from_iterable(edge_node_ids_lst))
        if not edge_node_id_set <= node_id_set:
            raise ValueError(f"Edges include missing node IDs: {edge_node_ids_lst}")


def network(mess_inp: str | Path) -> Network:
    """Read network.

    :param mess_inp: MESS input
    :return: Network
    """
    all_blocks = blocks(mess_inp)
    node_blocks = [b for b in all_blocks if b[0] in ["Well", "Bimolecular"]]
    edge_blocks = [b for b in all_blocks if b[0] == "Barrier"]

    for id, block in node_blocks:
        pass



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
