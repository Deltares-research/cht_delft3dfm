"""Delft3D-FM observation points: read, write, add, and delete point locations."""

import os
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from matplotlib import path


class Delft3DFMObservationPoints:
    """Container for Delft3D-FM observation points.

    Parameters
    ----------
    model : Delft3DFM
        Parent model instance.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """Read observation points from all files listed in the model input.

        Does nothing if no observation-point file is configured.
        """
        if not self.model.input.output.obsfile:
            return

        file_list = []
        for i, v in enumerate(self.model.input.output.obsfile):
            file_list.append(os.path.join(self.model.path, v.filepath))

        gdf_list = []

        for file_name in file_list:
            if not os.path.exists(file_name):
                print(f"Warning : file {file_name} does not exist !")
                return
            df = pd.read_csv(
                file_name,
                index_col=False,
                header=None,
                delim_whitespace=True,
                names=["x", "y", "name"],
            )

            for ind in range(len(df.x.values)):
                name = str(df.name.values[ind])
                x = df.x.values[ind]
                y = df.y.values[ind]
                point = shapely.geometry.Point(x, y)
                d = {"name": name, "long_name": None, "geometry": point}
                gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def write(self, file_name: Optional[str] = None) -> None:
        """Write observation points to file.

        Parameters
        ----------
        file_name : str, optional
            Output file path.  If ``None``, the path is taken from the model
            input configuration.
        """
        if len(self.gdf.index) == 0:
            return

        if not file_name:
            if not self.model.input.output.obsfile:
                return
            file_name = self.model.input.output.obsfile[0].filepath

        with open(file_name, "w") as fid:
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.6f}{y:12.6f}  "{name}"\n'
                fid.write(string)

    def add_point(self, x: float, y: float, name: str) -> None:
        """Add a single observation point.

        Parameters
        ----------
        x : float
            X-coordinate of the point.
        y : float
            Y-coordinate of the point.
        name : str
            Point identifier.
        """
        point = shapely.geometry.Point(x, y)
        gdf_list = []
        d = {"name": name, "long_name": None, "geometry": point}
        gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def add_points(self, gdf: gpd.GeoDataFrame, name: str = "name") -> None:
        """Add multiple observation points from a GeoDataFrame, filtered to the grid extent.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Input points to add.
        name : str, optional
            Column in *gdf* to use as the point name (default ``"name"``).
        """
        if self.model.grid.exterior.empty:
            self.model.grid.get_exterior()
        outline = self.model.grid.exterior.loc[0]["geometry"]
        gdf = gdf.to_crs(self.model.crs)
        x = np.empty((len(gdf)))
        y = np.empty((len(gdf)))
        for index, row in gdf.iterrows():
            x[index] = row["geometry"].coords[0][0]
            y[index] = row["geometry"].coords[0][1]
        inpol = inpolygon(x, y, outline)
        gdf_list = []
        for index, row in gdf.iterrows():
            if inpol[index]:
                d = {
                    "name": row[name],
                    "long_name": None,
                    "geometry": shapely.geometry.Point(x[index], y[index]),
                }
                gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, name_or_index: Union[str, int]) -> None:
        """Remove an observation point by name or index.

        Parameters
        ----------
        name_or_index : str or int
            Name string or integer row index of the point to remove.
        """
        if isinstance(name_or_index, str):
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print(f"Point {name} not found!")
        else:
            index = name_or_index
            if len(self.gdf.index) < index + 1:
                print("Index exceeds length!")
            self.gdf = self.gdf.drop(index).reset_index(drop=True)
            return

    def clear(self) -> None:
        """Remove all observation points."""
        self.gdf = gpd.GeoDataFrame()

    def list_names(self) -> list:
        """Return a list of observation point names.

        Returns
        -------
        list of str
            Names of all observation points in the collection.
        """
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names


def inpolygon(xq: np.ndarray, yq: np.ndarray, p) -> np.ndarray:
    """Test whether query points lie inside a polygon.

    Parameters
    ----------
    xq : np.ndarray
        X-coordinates of query points.
    yq : np.ndarray
        Y-coordinates of query points.
    p : shapely.geometry.Polygon
        Polygon to test against.

    Returns
    -------
    np.ndarray of bool
        Boolean array with the same shape as *xq*, ``True`` where the
        corresponding point is inside *p*.
    """
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
    return p.contains_points(q).reshape(shape)
