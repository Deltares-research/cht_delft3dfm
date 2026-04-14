"""Datashader-based trimesh elevation overlay for Delft3D-FM grids.

Mirrors :py:class:`hydromt_sfincs.workflows.map_overlay.ElevationOverlay`
but operates on unstructured meshes with mixed triangles and quadrilaterals.
Node elevations are assumed to live on ``mesh2d_node_z``; face-to-node
connectivity is fan-triangulated via :py:meth:`xugrid.Ugrid2d.triangulate`.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

__all__ = [
    "ElevationOverlay",
    "GridElevation",
    "MeshOverlay",
    "make_edge_dataframe",
    "make_elevation_trimesh",
    "make_elevation_overlay",
    "make_map_overlay",
]

logger = logging.getLogger(__name__)

try:
    import datashader.transfer_functions as tf
    from datashader import Canvas
    from datashader.utils import export_image

    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False


def make_edge_dataframe(ugrid, source_crs: CRS) -> pd.DataFrame:
    """Build a datashader-ready edge-coordinate DataFrame in EPSG:3857.

    One row per mesh edge with its two endpoints reprojected from
    ``source_crs``. Mirrors the SFINCS implementation.
    """
    x1 = ugrid.edge_node_coordinates[:, 0, 0]
    x2 = ugrid.edge_node_coordinates[:, 1, 0]
    y1 = ugrid.edge_node_coordinates[:, 0, 1]
    y2 = ugrid.edge_node_coordinates[:, 1, 1]

    cross_dateline = False
    if source_crs.is_geographic and (np.max(x1) > 180.0 or np.max(x2) > 180.0):
        cross_dateline = True

    transformer = Transformer.from_crs(source_crs, 3857, always_xy=True)
    x1, y1 = transformer.transform(x1, y1)
    x2, y2 = transformer.transform(x2, y2)
    if cross_dateline:
        x1[x1 < 0] += 40075016.68557849
        x2[x2 < 0] += 40075016.68557849

    return pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))


def make_map_overlay(
    dataframe: pd.DataFrame,
    file_name: Union[str, Path],
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    color: str = "black",
    width: int = 800,
) -> bool:
    """Render a PNG edge overlay using datashader."""
    if not HAS_DATASHADER:
        logger.warning("Datashader is not available. Please install datashader.")
        return False
    if dataframe is None or dataframe.empty:
        return False
    try:
        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        xl0, yl0 = transformer.transform(xlim[0], ylim[0])
        xl1, yl1 = transformer.transform(xlim[1], ylim[1])
        if xl0 > xl1:
            xl1 += 40075016.68557849
        xlim = [xl0, xl1]
        ylim = [yl0, yl1]
        ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        height = int(width * ratio)
        cvs = Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)
        agg = cvs.line(dataframe, x=["x1", "x2"], y=["y1", "y2"], axis=1)
        img = tf.shade(agg, cmap=color)
        path_dir = os.path.dirname(file_name) or os.getcwd()
        name = os.path.splitext(os.path.basename(file_name))[0]
        export_image(img, name, export_path=path_dir)
        return True
    except Exception:
        return False


class MeshOverlay:
    """Lazy, cached grid-edge overlay renderer."""

    def __init__(self) -> None:
        self._df: pd.DataFrame = pd.DataFrame()

    def invalidate(self) -> None:
        self._df = pd.DataFrame()

    def render(
        self,
        ugrid,
        source_crs: CRS,
        file_name: Union[str, Path],
        xlim: Optional[List[float]] = None,
        ylim: Optional[List[float]] = None,
        color: str = "black",
        width: int = 800,
    ) -> bool:
        if self._df.empty:
            self._df = make_edge_dataframe(ugrid, source_crs)
        return make_map_overlay(
            self._df, file_name, xlim=xlim, ylim=ylim, color=color, width=width,
        )


def make_elevation_trimesh(
    ugrid,
    z: np.ndarray,
    source_crs: CRS,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Build datashader trimesh DataFrames from an unstructured mesh.

    Nodes are reprojected from ``source_crs`` to EPSG:3857 (dateline-aware
    for geographic sources). Faces are fan-triangulated using
    :py:meth:`xugrid.Ugrid2d.triangulate`, which preserves node indexing.

    Parameters
    ----------
    ugrid : xugrid.Ugrid2d
        Source mesh exposing ``node_x``, ``node_y`` and
        ``face_node_connectivity``.
    z : np.ndarray
        Node elevations aligned with ``ugrid.node_x`` / ``ugrid.node_y``.
    source_crs : pyproj.CRS
        CRS of the mesh coordinates.

    Returns
    -------
    (vertices, simplices) : two pd.DataFrame
        ``vertices`` has columns ``x, y, z``; ``simplices`` has
        ``v0, v1, v2``. Both empty if no finite z values are present.
    """
    nx = np.asarray(ugrid.node_x, dtype=np.float64)
    ny = np.asarray(ugrid.node_y, dtype=np.float64)
    nz = np.asarray(z, dtype=np.float64)

    if nz.size == 0 or not np.any(np.isfinite(nz)):
        logger.warning(
            "Delft3D-FM elevation trimesh: no finite node z values (n=%d)", nz.size
        )
        return pd.DataFrame(), pd.DataFrame()

    transformer = Transformer.from_crs(source_crs, 3857, always_xy=True)
    vx, vy = transformer.transform(nx, ny)
    if source_crs.is_geographic and np.max(nx) > 180.0:
        vx[vx < 0] += 40075016.68557849

    triangulated = ugrid.triangulate()
    tris = np.asarray(triangulated.face_node_connectivity, dtype=np.int64)

    vertices = pd.DataFrame({"x": vx, "y": vy, "z": nz})
    simplices = pd.DataFrame({"v0": tris[:, 0], "v1": tris[:, 1], "v2": tris[:, 2]})
    return vertices, simplices


def make_elevation_overlay(
    vertices: pd.DataFrame,
    simplices: pd.DataFrame,
    file_name: Union[str, Path],
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    cmap="gist_earth",
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    width: int = 800,
) -> bool:
    """Render an elevation trimesh overlay to PNG using datashader."""
    if not HAS_DATASHADER:
        logger.warning("Datashader is not available. Please install datashader.")
        return False
    if vertices is None or vertices.empty:
        return False

    try:
        import datashader as ds

        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        xl0, yl0 = transformer.transform(xlim[0], ylim[0])
        xl1, yl1 = transformer.transform(xlim[1], ylim[1])
        if xl0 > xl1:
            xl1 += 40075016.68557849
        x_range = (xl0, xl1)
        y_range = (yl0, yl1)
        ratio = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
        height = int(width * ratio)

        cvs = Canvas(
            x_range=x_range, y_range=y_range, plot_width=width, plot_height=height,
        )
        mesh = ds.utils.mesh(vertices, simplices)
        agg = cvs.trimesh(
            vertices, simplices, mesh=mesh, agg=ds.mean("z"), interp=False,
        )

        if isinstance(cmap, str):
            from matplotlib import colormaps

            cmap = colormaps[cmap]
        span = (cmin, cmax) if (cmin is not None and cmax is not None) else None
        img = tf.shade(agg, cmap=cmap, span=span, how="linear")

        path_dir = os.path.dirname(file_name) or os.getcwd()
        name = os.path.splitext(os.path.basename(file_name))[0]
        export_image(img, name, export_path=path_dir)
        return True
    except Exception as exc:
        logger.exception("Delft3D-FM elevation overlay failed: %s", exc)
        return False


class GridElevation:
    """Thin view exposing ``map_overlay`` on a :class:`Delft3DFMGrid`.

    The DDB bathymetry raster layer calls ``data.map_overlay(...)``; this
    proxy forwards to the grid's cached :class:`ElevationOverlay`. Same
    role as ``SfincsQuadtreeElevation`` in hydromt-sfincs.
    """

    def __init__(self, grid) -> None:
        self._grid = grid

    def map_overlay(
        self,
        file_name,
        xlim=None,
        ylim=None,
        cmap="gist_earth",
        cmin=None,
        cmax=None,
        width: int = 800,
        **kwargs,
    ) -> bool:
        g = self._grid
        if g.data is None:
            logger.info("GridElevation.map_overlay: grid.data is None — no overlay")
            return False
        if "mesh2d_node_z" not in g.data:
            logger.info(
                "GridElevation.map_overlay: 'mesh2d_node_z' not in grid.data "
                "(vars: %s)", list(g.data.data_vars)
            )
            return False
        return g._elevation_overlay.render(
            ugrid=g.data.grid,
            z=g.data["mesh2d_node_z"].values[:],
            source_crs=g.model.crs,
            file_name=file_name,
            xlim=xlim,
            ylim=ylim,
            cmap=cmap,
            cmin=cmin,
            cmax=cmax,
            width=width,
        )


class ElevationOverlay:
    """Lazy, cached elevation-trimesh overlay renderer.

    Mirrors :py:class:`hydromt_sfincs.workflows.map_overlay.ElevationOverlay`
    but for unstructured (mixed triangle + quadrilateral) meshes.
    """

    def __init__(self) -> None:
        self._vertices: pd.DataFrame = pd.DataFrame()
        self._simplices: pd.DataFrame = pd.DataFrame()

    def invalidate(self) -> None:
        """Drop the cached trimesh; it will be rebuilt on next render."""
        self._vertices = pd.DataFrame()
        self._simplices = pd.DataFrame()

    def render(
        self,
        ugrid,
        z: np.ndarray,
        source_crs: CRS,
        file_name: Union[str, Path],
        xlim: Optional[List[float]] = None,
        ylim: Optional[List[float]] = None,
        cmap="gist_earth",
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        width: int = 800,
    ) -> bool:
        """Render the elevation overlay, rebuilding the cache if needed."""
        if self._vertices.empty:
            self._vertices, self._simplices = make_elevation_trimesh(
                ugrid, z, source_crs,
            )
        return make_elevation_overlay(
            self._vertices, self._simplices, file_name,
            xlim=xlim, ylim=ylim, cmap=cmap, cmin=cmin, cmax=cmax, width=width,
        )
