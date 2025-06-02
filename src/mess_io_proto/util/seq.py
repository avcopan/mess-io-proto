from collections.abc import Sequence


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
