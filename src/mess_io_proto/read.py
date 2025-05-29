"""Read MESS file format."""

from pathlib import Path

import pyparsing as pp

COMMENT = pp.Literal("!") + pp.rest_of_line()
WELL_KEY = pp.Keyword("Well")
BIMOL_KEY = pp.Keyword("Bimolecular")
BARRIER_KEY = pp.Keyword("Barrier")
END_KEY = pp.Keyword("End")
END_FILE = END_KEY + pp.StringEnd()

BLOCK_START = WELL_KEY | BIMOL_KEY | BARRIER_KEY
BLOCK_END = END_KEY + pp.FollowedBy(BLOCK_START | END_FILE)
HEADER = pp.SkipTo(BLOCK_START)


def blocks(mess_inp: str | Path) -> dict[str, str]:
    """Read well/bimol/barrier blocks.

    :param mess_inp: MESS input
    :return: A dictionary mapping block type to contents
    """
    mess_inp = mess_inp.read_text() if isinstance(mess_inp, Path) else mess_inp

    block = BLOCK_START("type") + pp.SkipTo(BLOCK_END)("contents") + BLOCK_END

    expr = pp.Suppress(HEADER) + pp.OneOrMore(pp.Group(block))
    expr.ignore(COMMENT)

    block_dct = {}
    for res in expr.parse_string(mess_inp):
        block_dct[res.get("type")] = res.get("contents")
    
    return block_dct
