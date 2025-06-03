import itertools
from collections.abc import Sequence

import numpy


def longest_common_subsequence(
    seq1: Sequence[object], seq2: Sequence[object]
) -> list[object]:
    """Determine the longest common subsequence of two sequences.

    Algorithm based on https://stackoverflow.com/a/48653758

    :param seq1: First sequence
    :param seq2: Second sequence
    :return: Common subsequence
    """
    n1, n2 = len(seq1), len(seq2)
    comp = numpy.empty((n1, n2), dtype=tuple)
    comp.fill(())
    for i1, i2 in itertools.product(range(n1), range(n2)):
        if seq1[i1] == seq2[i2]:
            if i1 == 0 or i2 == 0:
                comp[i1, i2] = (seq1[i1],)
            else:
                comp[i1, i2] = comp[i1 - 1, i2 - 1] + (seq1[i1],)
        else:
            comp[i1, i2] = max(comp[i1 - 1, i2], comp[i1, i2 - 1], key=len)
    return list(comp[-1, -1])


def common_prefix(seqs: Sequence[Sequence[object]]) -> list[object]:
    """Determine common prefix of multiple sequences."""
    if not seqs:
        return []

    seq1 = min(seqs)
    seq2 = max(seqs)
    for i, obj in enumerate(seq1):
        if obj != seq2[i]:
            return seq1[:i]
    return seq1


def unique_suffixes(seqs: Sequence[Sequence[object]]) -> list[list[object]]:
    """Determine common prefix of multiple sequences."""
    if not seqs:
        return []

    seq1 = min(seqs)
    seq2 = max(seqs)
    for i, obj in enumerate(seq1):
        if obj != seq2[i]:
            return [s[i:] for s in seqs]
    return seqs
