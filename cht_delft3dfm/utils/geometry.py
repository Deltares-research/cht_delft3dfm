"""Geometry utility classes for building and inspecting regular grids and polylines."""

import math
from typing import Optional

import geopandas as gpd
import numpy as np
import shapely


class Geometry:
    """Base class for geometry objects."""

    def __init__(self) -> None:
        pass


class RegularGrid(Geometry):
    """A rotated regular (structured) grid.

    Parameters
    ----------
    hw : object
        Parent model or context object (unused, kept for API compatibility).
    x0 : float, optional
        X-coordinate of the grid origin.
    y0 : float, optional
        Y-coordinate of the grid origin.
    dx : float, optional
        Cell size in the x-direction.
    dy : float, optional
        Cell size in the y-direction.
    nmax : int, optional
        Number of rows.
    mmax : int, optional
        Number of columns.
    rotation : float, optional
        Grid rotation angle in degrees (counter-clockwise).
    crs : object, optional
        Coordinate reference system.
    """

    def __init__(
        self,
        hw,
        x0: Optional[float] = None,
        y0: Optional[float] = None,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        nmax: Optional[int] = None,
        mmax: Optional[int] = None,
        rotation: Optional[float] = None,
        crs=None,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.mmax = mmax
        self.nmax = nmax
        self.rotation = rotation
        self.crs = crs
        if x0:
            self.xg, self.yg = self.grid_coordinates_corners()
            self.xz, self.yz = self.grid_coordinates_centres()

    def build(
        self,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        nx: int,
        ny: int,
        rotation: float,
        crs,
    ) -> None:
        """Configure and generate grid coordinate arrays.

        Parameters
        ----------
        x0 : float
            X-coordinate of the grid origin.
        y0 : float
            Y-coordinate of the grid origin.
        dx : float
            Cell size in the x-direction.
        dy : float
            Cell size in the y-direction.
        nx : int
            Number of columns.
        ny : int
            Number of rows.
        rotation : float
            Grid rotation in degrees.
        crs : object
            Coordinate reference system.
        """
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.mmax = nx
        self.nmax = ny
        self.rotation = rotation
        self.xg, self.yg = self.grid_coordinates_corners()
        self.xz, self.yz = self.grid_coordinates_centres()
        self.crs = crs

    def grid_coordinates_corners(self) -> tuple:
        """Compute the x/y coordinates of cell corner nodes.

        Returns
        -------
        xg : np.ndarray
            2-D array of x-coordinates at cell corners.
        yg : np.ndarray
            2-D array of y-coordinates at cell corners.
        """
        cosrot = np.cos(self.rotation * np.pi / 180)
        sinrot = np.sin(self.rotation * np.pi / 180)
        xx = np.linspace(0.0, self.mmax * self.dx, num=self.mmax + 1)
        yy = np.linspace(0.0, self.nmax * self.dy, num=self.nmax + 1)
        xg0, yg0 = np.meshgrid(xx, yy)
        xg = self.x0 + xg0 * cosrot - yg0 * sinrot
        yg = self.y0 + xg0 * sinrot + yg0 * cosrot

        return xg, yg

    def grid_coordinates_centres(self) -> tuple:
        """Compute the x/y coordinates of cell centres.

        Returns
        -------
        xz : np.ndarray
            2-D array of x-coordinates at cell centres.
        yz : np.ndarray
            2-D array of y-coordinates at cell centres.
        """
        cosrot = np.cos(self.rotation * np.pi / 180)
        sinrot = np.sin(self.rotation * np.pi / 180)
        xx = np.linspace(
            0.5 * self.dx, self.mmax * self.dx - 0.5 * self.dx, num=self.mmax
        )
        yy = np.linspace(
            0.5 * self.dy, self.nmax * self.dy - 0.5 * self.dy, num=self.nmax
        )
        xg0, yg0 = np.meshgrid(xx, yy)
        xz = self.x0 + xg0 * cosrot - yg0 * sinrot
        yz = self.y0 + xg0 * sinrot + yg0 * cosrot

        return xz, yz

    def plot(self, ax) -> None:
        """Placeholder for grid plotting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot onto.
        """
        pass

    def to_gdf(self) -> gpd.GeoDataFrame:
        """Export grid edges as a GeoDataFrame of MultiLineString geometry.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with a single row containing the grid edge lines.
        """
        lines = []

        cosrot = math.cos(self.rotation * math.pi / 180)
        sinrot = math.sin(self.rotation * math.pi / 180)

        for n in range(self.nmax):
            for m in range(self.mmax):
                xa = self.x0 + m * self.dx * cosrot - n * self.dy * sinrot
                ya = self.y0 + m * self.dx * sinrot + n * self.dy * cosrot
                xb = self.x0 + (m + 1) * self.dx * cosrot - n * self.dy * sinrot
                yb = self.y0 + (m + 1) * self.dx * sinrot + n * self.dy * cosrot
                line = shapely.geometry.LineString([[xa, ya], [xb, yb]])
                lines.append(line)
                xb = self.x0 + m * self.dx * cosrot - (n + 1) * self.dy * sinrot
                yb = self.y0 + m * self.dx * sinrot + (n + 1) * self.dy * cosrot
                line = shapely.geometry.LineString([[xa, ya], [xb, yb]])
                lines.append(line)
        geom = shapely.geometry.MultiLineString(lines)
        gdf = gpd.GeoDataFrame(crs=self.crs, geometry=[geom])

        return gdf


class Point:
    """A single 2-D point with optional name and CRS.

    Parameters
    ----------
    x : float
        X-coordinate.
    y : float
        Y-coordinate.
    name : str, optional
        Point identifier.
    crs : object, optional
        Coordinate reference system.
    """

    def __init__(
        self,
        x: float,
        y: float,
        name: Optional[str] = None,
        crs=None,
    ) -> None:
        self.x = x
        self.y = y
        self.crs = crs
        self.name = name
        self.data = None


class Polyline(Geometry):
    """An ordered sequence of points forming an open or closed line.

    Parameters
    ----------
    x : list of float, optional
        X-coordinates of the polyline vertices.
    y : list of float, optional
        Y-coordinates of the polyline vertices.
    crs : object, optional
        Coordinate reference system.
    name : str, optional
        Polyline identifier.
    closed : bool, optional
        Whether the polyline is closed (default ``False``).
    """

    def __init__(
        self,
        x=None,
        y=None,
        crs=None,
        name: Optional[str] = None,
        closed: bool = False,
    ) -> None:
        self.point = []
        self.name = name
        self.data = None
        self.closed = closed
        self.crs = crs

        if x is not None:
            for j, xp in enumerate(x):
                pnt = Point(x[j], y[j])
                self.point.append(pnt)

    def add_point(
        self,
        x: float,
        y: float,
        name: Optional[str] = None,
        data=None,
        position: int = -1,
    ) -> None:
        """Append or insert a point into the polyline.

        Parameters
        ----------
        x : float
            X-coordinate of the new point.
        y : float
            Y-coordinate of the new point.
        name : str, optional
            Point identifier.
        data : object, optional
            Arbitrary data to attach to the point.
        position : int, optional
            Index at which to insert the point.  ``-1`` (default) appends
            to the end.
        """
        pnt = Point(x, y, name=name, data=data)
        if position < 0:
            # Add point to the end
            self.point.append(pnt)
        else:
            #
            pass

    def plot(self, ax=None) -> None:
        """Placeholder for polyline plotting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot onto.
        """
        pass
