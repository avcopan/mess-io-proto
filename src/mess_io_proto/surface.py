"""Read MESS file format."""

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import automol
import more_itertools as mit
import networkx
import numpy
import polars
import pyvis
import scipy
import scipy.interpolate
from matplotlib import figure, offsetbox
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    model_validator,
)
from rdkit.Chem import Draw

from .util import MessBlockParseData, mess, seqn


class Feature(BaseModel, ABC):
    energy: float
    fake: bool = False

    @property
    @abstractmethod
    def label(self):
        """Label."""
        pass


class Well(Feature):
    id: int

    @property
    @abstractmethod
    def names_list(self):
        """Names."""
        pass


class UnimolWell(Well):
    type: Literal["unimol"] = "unimol"
    name: str

    @property
    def label(self):
        """Label."""
        return self.name

    @property
    def names_list(self):
        """Label."""
        return [self.name]


class NMolWell(Well):
    type: Literal["nmol"] = "nmol"
    names: Annotated[list[str], AfterValidator(sorted)]
    interacting: bool = False

    @property
    def label(self):
        """Label."""
        label = " + ".join(self.names)
        if self.fake:
            label = f"Fake({label})"

        return label

    @property
    def names_list(self):
        """Label."""
        return self.names


class Barrier(Feature):
    well_ids: Annotated[tuple[int, int], AfterValidator(lambda x: tuple(sorted(x)))]
    name: str
    energy: float
    barrierless: bool = False

    @property
    def label(self):
        """Label."""
        return self.name


class Surface(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    wells: list[Well]
    barriers: list[Barrier]
    amchi_mapping: dict[str, str] | None = None

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


def from_mess(mess_inp: str | Path, spc_inp: str | Path | None = None) -> Surface:
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

    chi_dct = None
    if spc_inp is not None:
        spc_inp = Path(spc_inp) if isinstance(spc_inp, str) else spc_inp
        spc_df = polars.read_csv(spc_inp, quote_char="'")
        spc_df = spc_df.with_columns(
            polars.col("smiles")
            .map_elements(automol.smiles.amchi, return_dtype=str)
            .alias("amchi")
        )
        chi_dct = dict(spc_df.select(["name", "amchi"]).iter_rows())
        names = set(itertools.chain.from_iterable(w.names_list for w in wells))
        assert all(n in chi_dct for n in names), f"{names} !<= {chi_dct}"

    return Surface(wells=wells, barriers=barriers, amchi_mapping=chi_dct)


def with_energies_relative_to(surf: Surface, well_id: int) -> Surface:
    """Get surface with energies relative to one well.

    :param surf: Surface
    :param well_id: Well ID
    :return: Surface
    """
    well0 = next(w for w in surf.wells if w.id == well_id)
    energy0 = well0.energy

    surf = surf.model_copy(deep=True)
    for well in surf.wells:
        well.energy -= energy0
    for barrier in surf.barriers:
        barrier.energy -= energy0

    return surf


def without_fake_wells(surf: Surface) -> Surface:
    """Remove fake wells from a surface.

    Connects the two ends and removes the fake barrier.

    :param surf: Surface
    :return: Surface
    """
    map_dct = fake_well_mapping(surf, full=True)
    wells = [w.model_copy() for w in surf.wells if not w.fake]
    barriers = [
        Barrier(
            well_ids=sorted(map(map_dct.get, b.well_ids)),
            **b.model_dump(exclude="well_ids"),
        )
        for b in surf.barriers
        if not b.fake
    ]
    return Surface(wells=wells, barriers=barriers, amchi_mapping=surf.amchi_mapping)


def fake_well_mapping(surf: Surface, full: bool = False) -> dict[int, int]:
    """Get the mapping of fake wells to real wells.

    :param surf: Surface
    :param full: Whether to map real wells onto themselves
    :return: Mapping of fake well IDs to real well IDs
    """
    fake_wells = [w for w in surf.wells if w.fake]
    fake_barriers = [b for b in surf.barriers if b.fake]
    map_dct = {}
    for fake_well in fake_wells:
        fake_barrier = next(
            (b for b in fake_barriers if fake_well.id in b.well_ids), None
        )
        assert fake_barrier is not None, surf
        fake_well_id = fake_well.id
        (real_well_id,) = set(fake_barrier.well_ids) - {fake_well_id}
        map_dct[fake_well_id] = real_well_id

    if full:
        map_dct.update({w.id: w.id for w in surf.wells if not w.fake})

    return map_dct


def subsurface(surf: Surface, well_ids: Sequence[int]) -> Surface:
    """Extract a sub-surface from wells.

    :param surf: Surface
    :return: Surface
    """
    well_ids = set(well_ids)
    return Surface(
        wells=[w for w in surf.wells if w.id in well_ids],
        barriers=[b for b in surf.barriers if set(b.well_ids) <= well_ids],
    )


def feature_dict(
    surf: Surface, drop_barrierless: bool = False
) -> dict[int | frozenset[int], Feature]:
    """Get features by ID.

    :param surf: Surface
    :param drop_barrierless: Whether to drop barrierless barriers
    :return: Dicture mapping IDs to features
    """
    feat_dct = {w.id: w for w in surf.wells}
    feat_dct.update(
        {frozenset(b.well_ids): b for b in surf.barriers if not b.barrierless}
    )
    return feat_dct


def longest_path(surf: Surface) -> Surface:
    """Longest surface path.

    :param surf: Surface
    :return: Path surface
    """
    nx_gra = graph(surf)
    well_id_seq = max(
        (p for _, d in networkx.all_pairs_shortest_path(nx_gra) for p in d.values()),
        key=len,
    )
    return path_from_well_id_sequence(surf, well_id_seq)


def path_from_well_id_sequence(
    surf: Surface, well_id_seq: Sequence[int]
) -> list[Feature]:
    """Generate ordered sequence of wells and barriers from a well ID sequence.

    :param surf: Surface
    :param path: Path
    :param barrierless: Whether to include barrierless edges
    :return: Wells and barriers in order
    """
    seq = list(
        mit.interleave_longest(well_id_seq, map(frozenset, mit.pairwise(well_id_seq)))
    )
    dct = {w.id: w for w in surf.wells}
    dct.update({frozenset(b.well_ids): b for b in surf.barriers if not b.barrierless})
    return [dct.get(k) for k in seq if k in dct]


COLOR_SEQUENCE = [
    "#000000",
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def plot_paths(
    surf: Surface,
    paths: Sequence[Sequence[int]],
    fig: figure.Figure,
    colors: Sequence[str] | None = None,
    stereo: bool = True,
) -> figure.Figure:
    """Plot multiple paths onto matplotlib figure."""
    npaths = len(paths)
    colors = colors or list(itertools.islice(itertools.cycle(COLOR_SEQUENCE), npaths))

    # Split paths to contain only unique edges, tracking their colors
    paths0 = paths
    colors0 = colors
    paths = []
    colors = []
    for i in range(npaths):
        prev_paths = paths0[:i]
        path = paths0[i]
        color = colors0[i]
        paths_ = seqn.unique_edge_paths(path, comp_paths=prev_paths)
        paths.extend(paths_)
        colors.extend([color] * len(paths_))

    # Sort starting, middle, and ending wells in order of appearance
    mid_pool = set(itertools.chain.from_iterable(p[1:-1] for p in paths))
    start_pool = {p[0] for p in paths} - mid_pool
    end_pool = {p[-1] for p in paths} - mid_pool
    # BUG: Mid-ID ordering is incorrect. We need to make sure that the ordering of
    # mid-IDs is consistent with *all* paths for situations like:
    #   paths = [
    #       [23, 11, 1, 19],
    #       [22, 0, 1, 19],
    #   ]
    # Here, the ordering must be 11-0-1, not 11-1-0
    order = list(mit.unique_everseen(itertools.chain.from_iterable(paths)))
    # This patches the bug, but doesn't give the desired order (giving precedence to
    # earlier paths)
    # order = list(
    #     reversed(
    #         list(
    #             mit.unique_everseen(
    #                 reversed(list(itertools.chain.from_iterable(paths)))
    #             )
    #         )
    #     )
    # )
    start_ids = sorted(start_pool, key=order.index)
    mid_ids = sorted(mid_pool, key=order.index)
    end_ids = sorted(end_pool, key=order.index)

    # Assign coordinates to starting, middle, and ending wells
    coord_min = -1
    coord_max = len(mid_ids)
    coord_dct = {id_: i for i, id_ in enumerate(mid_ids)}
    coord_dct.update(dict.fromkeys(start_ids, coord_min))
    coord_dct.update(dict.fromkeys(end_ids, coord_max))

    # Determine coordinates for path features, including barriers
    feat_dct = feature_dict(surf, drop_barrierless=True)
    feats_lst = []
    coords_lst = []
    for path in paths:
        keys = [id_ for id_ in seqn.path_node_edge_sequence(path) if id_ in feat_dct]

        # Add feature list
        feats = list(map(feat_dct.get, keys))
        feats_lst.append(feats)

        # Add coordinate list
        coords = list(map(coord_dct.get, keys))
        coords = interpolate_missing_coordinates(coords)
        coords_lst.append(coords)

        print(f"keys = {keys}")
        print(f"feats = {feats}")
        print(f"coords = {coords}")

    # Determine energy range
    all_feats = list(itertools.chain.from_iterable(feats_lst))
    energy_min = min(f.energy for f in all_feats)
    energy_max = max(f.energy for f in all_feats)

    # Plot in reverse order to put earlier paths on top
    for feats, coords, color in reversed(
        list(zip(feats_lst, coords_lst, colors, strict=True))
    ):
        fig = _plot_path_features(
            feats=feats,
            fig=fig,
            coords=coords,
            color=color,
            stereo=stereo,
            amchi_mapping=surf.amchi_mapping,
            x_range=(coord_min, coord_max),
            y_range=(energy_min, energy_max),
        )

    return fig


def plot_connected_paths(
    surf: Surface,
    id_seqs: Sequence[Sequence[int]],
    fig: figure.Figure,
    colors: Sequence[str] | None = None,
    stereo: bool = True,
    amchi_mapping: dict[str, str] | None = None,
) -> figure.Figure:
    """Plot multiple paths onto matplotlib figure.

    Currently assumes the paths all start from a common root.
    """
    nseq = len(id_seqs)
    colors = colors or list(itertools.islice(itertools.cycle(COLOR_SEQUENCE), nseq))

    id_seqs_out = []
    for (*prev_id_seqs, id_seq), *_ in itertools.islice(
        mit.windowed_complete(id_seqs, n=0), 1, None
    ):
        # Determine longest shared prefix with previous sequences
        prefix = max(
            (list(mit.longest_common_prefix([id_seq, s])) for s in prev_id_seqs),
            default=(),
        )

        # Start from the last common ID, to continue path
        start = max(len(prefix) - 1, 0)
        id_seqs_out.append(id_seq[start:])

    # 1. Sort unique everseen indices, excluding termini
    srt_ids = list(
        mit.unique_everseen(itertools.chain.from_iterable(s[:-1] for s in id_seqs_out))
    )
    max_coord = len(srt_ids)
    coord_dct = {id_: i for i, id_ in enumerate(srt_ids)}
    coord_dct.update({s[-1]: max_coord for s in id_seqs_out})

    feat_dct = {w.id: w for w in surf.wells}
    feat_dct.update(
        {frozenset(b.well_ids): b for b in surf.barriers if not b.barrierless}
    )

    data_lst = []
    for id_seq in id_seqs_out:
        seq = list(mit.interleave_longest(id_seq, map(frozenset, mit.pairwise(id_seq))))
        seq = [x for x in seq if x in feat_dct]
        path = list(map(feat_dct.get, seq))
        coords = list(map(coord_dct.get, seq))
        coords = interpolate_missing_coordinates(coords)
        data = list(zip(coords, path, strict=True))
        data_lst.append(data)

    coords, feats = zip(*itertools.chain.from_iterable(data_lst), strict=True)
    x_min = min(coords)
    x_max = max(coords)
    y_min = min(f.energy for f in feats)
    y_max = max(f.energy for f in feats)

    for data, color in reversed(list(zip(data_lst, colors, strict=False))):
        _plot_path_from_data(
            data=data,
            fig=fig,
            color=color,
            stereo=stereo,
            amchi_mapping=amchi_mapping,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )

    return fig


def _plot_path_features(
    feats: Sequence[Feature],
    fig: figure.Figure,
    coords: Sequence[float],
    color: str = "black",
    stereo: bool = True,
    amchi_mapping: dict[str, str] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> figure.Figure:
    """Plot features along a path."""
    energies = [f.energy for f in feats]
    x_min, x_max = x_range or (min(coords), max(coords))
    y_min, y_max = y_range or (min(energies), max(energies))
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    grid = numpy.linspace(*x_range, 1000)

    # Get the current axis
    ax = fig.gca()

    # Turn off all but the y axis
    ax.xaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Energy (kcal/mol)")

    # Plot labels
    data = list(zip(coords, feats, strict=True))
    shift_x = True
    for coord, feat in data[::-1]:
        if isinstance(feat, Well):
            x0 = coord
            y0 = feat.energy
            x_ = x0 + 0.15 * x_scale if shift_x else x0
            y_ = y0 if shift_x else y0 - 0.1 * y_scale
            shift_x = False
            if amchi_mapping:
                chi = automol.amchi.join(list(map(amchi_mapping.get, feat.names_list)))
                img = _offset_image_from_amchi(chi, stereo=stereo)
                box = offsetbox.AnnotationBbox(
                    img, (x_, y_), frameon=False, annotation_clip=False
                )
                ax.add_artist(box)
            else:
                ax.annotate(
                    feat.label, (x_, y_), fontsize=10, ha="center", clip_on=False
                )

    # Plot path
    for (coord1, feat1), (coord2, feat2) in mit.pairwise(data):
        print("coord1, coord2", coord1, coord2)
        print("feat1, feat2", feat1, feat2)
        grid12 = grid[numpy.where((grid >= coord1) & (grid <= coord2))]
        interp = scipy.interpolate.BPoly.from_derivatives(
            (coord1, coord2), ((feat1.energy, 0), (feat2.energy, 0))
        )
        ax.plot(grid12, interp(grid12), color=color)

    return fig


def _plot_path_from_data(
    data: Sequence[tuple[float, Feature]],
    fig: figure.Figure,
    color: str = "black",
    stereo: bool = True,
    amchi_mapping: dict[str, str] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> figure.Figure:
    """Plot path onto matplotlib figure.

    :param data: Path data, with coordinates
    :return: Figure
    """
    coords, feats = zip(*data, strict=True)
    grid = numpy.linspace(0, max(coords), 1000)

    energies = [f.energy for f in feats]
    x_min, x_max = x_range or (min(coords), max(coords))
    y_min, y_max = y_range or (min(energies), max(energies))
    x_scale = x_max - x_min
    y_scale = y_max - y_min

    # Get the current axis
    ax = fig.gca()

    # Turn off all but the y axis
    ax.xaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Energy (kcal/mol)")

    # Plot labels
    shift_x = True
    for coord, feat in data[::-1]:
        if isinstance(feat, Well):
            x0 = coord
            y0 = feat.energy
            x_ = x0 + 0.15 * x_scale if shift_x else x0
            y_ = y0 if shift_x else y0 - 0.1 * y_scale
            shift_x = False
            if amchi_mapping:
                chi = automol.amchi.join(list(map(amchi_mapping.get, feat.names_list)))
                img = _offset_image_from_amchi(chi, stereo=stereo)
                box = offsetbox.AnnotationBbox(
                    img, (x_, y_), frameon=False, annotation_clip=False
                )
                ax.add_artist(box)
            else:
                ax.annotate(
                    feat.label, (x_, y_), fontsize=10, ha="center", clip_on=False
                )

    # Plot path
    for (coord1, feat1), (coord2, feat2) in mit.pairwise(data):
        grid12 = grid[numpy.where((grid >= coord1) & (grid <= coord2))]
        interp = scipy.interpolate.BPoly.from_derivatives(
            (coord1, coord2), ((feat1.energy, 0), (feat2.energy, 0))
        )
        ax.plot(grid12, interp(grid12), color=color)

    return fig


def from_graph(nx_gra: networkx.MultiGraph) -> Surface:
    """Generate Surface from NetworkX MultiGraph."""
    wells = [well_from_data(d) for *_, d in nx_gra.nodes.data()]
    barriers = [Barrier.model_validate(d) for *_, d in nx_gra.edges.data()]
    return Surface(wells=wells, barriers=barriers)


def graph(surf: Surface) -> networkx.MultiGraph:
    """Generate NetworkX MultiGraph."""
    nx_gra = networkx.MultiGraph()
    nx_gra.add_nodes_from([(w.id, w.model_dump()) for w in surf.wells])
    nx_gra.add_edges_from([(*b.well_ids, b.model_dump()) for b in surf.barriers])
    return nx_gra


def display_network(
    surf: Surface,
    height: str = "750px",
    out_name: str = "net.html",
    out_dir: str = ".pyvis",
    stereo: bool = True,
    open_browser: bool = True,
) -> None:
    """Display surface as a pyvis Network.

    :param surf: Surface
    :param height: Frame height
    """
    out_dir: Path = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    vis_net = pyvis.network.Network(
        height=height, directed=False, notebook=True, cdn_resources="in_line"
    )
    for well in surf.wells:
        if surf.amchi_mapping is None:
            vis_net.add_node(well.id, label=well.label, title=str(well.id))
        else:
            chi = automol.amchi.join(list(map(surf.amchi_mapping.get, well.names_list)))
            image_path = _image_file_from_amchi(chi, out_dir=out_dir, stereo=stereo)
            vis_net.add_node(
                well.id,
                label=well.label,
                title=str(well.id),
                shape="image",
                image=image_path,
            )
    for barrier in surf.barriers:
        vis_net.add_edge(*barrier.well_ids, title=barrier.label)

    # Generate the HTML file
    vis_net.write_html(str(out_dir / out_name), open_browser=open_browser)


# Helpers
def well_from_data(data: dict[str, object]) -> Well:
    """Generate Well object from data.

    :param data: Data
    :return: Well
    """
    if data.get("type") == "nmol":
        return NMolWell.model_validate(data)

    return UnimolWell.model_validate(data)


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
    barrierless = "PhaseSpaceTheory" in data.contents
    well_ids = list(map(id_dct.get, well_labels))
    return Barrier(
        well_ids=well_ids,
        name=name,
        energy=data.energy,
        fake=fake,
        barrierless=barrierless,
    )


# Helpers
def _image_file_from_amchi(chi, out_dir: str | Path, stereo: bool = True):
    """Create an SVG molecule drawing and return the path."""
    out_dir = Path(out_dir)
    img_dir = Path("img")
    (out_dir / img_dir).mkdir(exist_ok=True)

    gra = automol.amchi.graph(chi, stereo=stereo)
    svg_str = automol.graph.svg_string(gra, image_size=100)

    chk = automol.amchi.amchi_key(chi)
    path = img_dir / f"{chk}.svg"
    with open(out_dir / path, mode="w") as file:
        file.write(svg_str)

    return str(path)


def _offset_image_from_amchi(chi, stereo: bool = True) -> offsetbox.OffsetImage:
    """Get PNG image array from AMChI."""
    rdm = automol.amchi.rdkit_molecule(chi, stereo=stereo)
    rdd = Draw.MolDraw2DCairo(150, 100)
    rdd.drawOptions().setBackgroundColour((1, 1, 1, 0))
    rdd.drawOptions().useBWAtomPalette()
    rdd.DrawMolecule(rdm)
    img = Draw._drawerToImage(rdd)
    return offsetbox.OffsetImage(numpy.asarray(img), zoom=0.4)


def interpolate_missing_coordinates(coords: Sequence[float | None]) -> list[float]:
    """Interpolates missing values in a list using linear interpolation."""
    coords = numpy.array(coords, dtype=float)
    nan = numpy.isnan(coords)
    (miss_idxs,) = numpy.nonzero(nan)
    (good_idxs,) = numpy.nonzero(~nan)
    coords[nan] = numpy.interp(miss_idxs, good_idxs, coords[~nan])
    return coords.tolist()
