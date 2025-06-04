"""Utility functions acting on sequences."""

import itertools
from collections.abc import Sequence

import more_itertools as mit
import numpy


def path_node_edge_sequence(
    path: Sequence[object], ordered: bool = False
) -> list[object | tuple[object, object] | frozenset[object]]:
    """Determine sequence of nodes and edges traversing a path in order.

    :param path: Path sequence
    :return: Sequence of nodes and edges
    """
    edge_iter = mit.pairwise(path) if ordered else map(frozenset, mit.pairwise(path))
    edge_iter = edge_iter if ordered else map(frozenset, edge_iter)
    return list(mit.interleave_longest(path, edge_iter))


def unique_edge_paths(
    path: Sequence[object], comp_paths: Sequence[Sequence[object]]
) -> list[list[object]]:
    """Split a path to remove edges it shares with other paths.

    "Edges" here are adjacent values.

    :param seq: Path sequence
    :param seqs: Path sequences to compare against
    :return: Path subsequences with unique edges
    """
    edges = set(itertools.chain.from_iterable(map(mit.pairwise, comp_paths)))
    return list(
        filter(lambda s: len(s) > 1, mit.split_when(path, lambda x, y: (x, y) in edges))
    )


def longest_common_substring_bounds(
    seq1: Sequence[object], seq2: Sequence[object]
) -> list[tuple[int, int]]:
    """Determine the longest common substring slices of two sequences.

    Algorithm based on https://stackoverflow.com/a/48653758

    :param seq1: First sequence
    :param seq2: Second sequence
    :return: Common subsequence
    """
    n1, n2 = len(seq1), len(seq2)
    comp = numpy.empty((n1, n2), dtype=numpy.object_)
    max_len = 0
    slices = []
    for i1, i2 in itertools.product(range(n1), range(n2)):
        if seq1[i1] == seq2[i2]:
            comp[i1, i2] = 1 if (i1 == 0 or i2 == 0) else comp[i1 - 1, i2 - 1] + 1
            end = i1 + 1
            start = end - comp[i1, i2]
            if comp[i1, i2] > max_len:
                max_len = end - start
                slices = [(start, end)]
            elif comp[i1, i2] == max_len:
                slices.append((start, end))
        else:
            comp[i1, i2] = 0
    return slices


def longest_common_substrings(
    seq1: Sequence[object], seq2: Sequence[object]
) -> list[list[object]]:
    """Determine the longest common substring of two sequences.

    Algorithm based on https://stackoverflow.com/a/48653758

    :param seq1: First sequence
    :param seq2: Second sequence
    :return: Common subsequence
    """
    slices = [slice(s, e) for s, e in longest_common_substring_bounds(seq1, seq2)]
    return [seq1[s] for s in slices]


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
