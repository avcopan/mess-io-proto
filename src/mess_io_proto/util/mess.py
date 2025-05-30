"""Read MESS file format."""

import re
from pathlib import Path
from typing import Literal

import pyparsing as pp
from pydantic import BaseModel
from pyparsing import pyparsing_common as ppc

COMMENT = pp.Literal("!") + pp.rest_of_line()
END_FILE = pp.Keyword("End") + pp.StringEnd()

BLOCK_START = pp.Keyword("Well") | pp.Keyword("Bimolecular") | pp.Keyword("Barrier")
BLOCK_END = pp.Keyword("End") + pp.FollowedBy(BLOCK_START | END_FILE)
FILE_HEADER = pp.SkipTo(BLOCK_START)


class MessBlockParseData(BaseModel):
    type: Literal["Well", "Bimolecular", "Barrier"]
    label: str
    energy: float
    energy_unit: str
    contents: str


def parse_blocks(mess_inp: str | Path) -> list[MessBlockParseData]:
    """Read well/bimol/barrier blocks.

    :param mess_inp: MESS input
    :return: A list of type, contents tuples for each block
    """
    mess_inp = mess_inp.read_text() if isinstance(mess_inp, Path) else mess_inp

    block_expr = BLOCK_START("type") + pp.SkipTo(BLOCK_END)("contents") + BLOCK_END

    expr = pp.Suppress(FILE_HEADER) + pp.OneOrMore(pp.Group(block_expr))
    expr.ignore(COMMENT)

    block_data_lst = []
    for res in expr.parse_string(mess_inp):
        block_data = _parse_block(res.get("type"), res.get("contents"))
        block_data_lst.append(block_data)

    return block_data_lst


# Helpers
class Key:
    unit = "unit"
    energy = "energy"


UNIT = pp.Suppress("[") + pp.Word(pp.alphas + "/")(Key.unit) + pp.Suppress("]")
ZERO_ENERGY = pp.Literal("ZeroEnergy") + UNIT + ppc.number(Key.energy)
GROUND_ENERGY = pp.Literal("GroundEnergy") + UNIT + ppc.number(Key.energy)


def _parse_block(
    type: Literal["Well", "Bimolecular", "Barrier"], contents: str
) -> MessBlockParseData:
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

    return MessBlockParseData(
        type=type,
        label=label.strip(),
        energy=res.get(Key.energy),
        energy_unit=res.get(Key.unit),
        contents=contents,
    )
