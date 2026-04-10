"""Delft3D-FM thin-dam definition: read, write, add, and delete thin dams."""

import os
from typing import Union

import dfm_tools as dfmt
import geopandas as gpd
import hydrolib.core.dflowfm as hcdfm
import pandas as pd
import shapely


class Delft3DFMThinDams:
    """Container for Delft3D-FM thin-dam features.

    Parameters
    ----------
    model : Delft3DFM
        Parent model instance.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """Read thin dams from the file referenced in the model input.

        Does nothing if no thin-dam file is configured.
        """
        if not self.model.input.geometry.thindamfile:
            return

        filename = os.path.join(
            self.model.path, self.model.input.geometry.thindamfile[0].filepath
        )
        data = hcdfm.PolyFile(filename)
        self.gdf = dfmt.PolyFile_to_geodataframe_linestrings(data, crs=self.model.crs)

    def write(self) -> None:
        """Write thin dams to the file referenced in the model input.

        Does nothing if no thin-dam file is configured or the GeoDataFrame is
        empty.
        """
        if not self.model.input.geometry.thindamfile:
            return
        if len(self.gdf.index) == 0:
            return

        filename = os.path.join(
            self.model.path, self.model.input.geometry.thindamfile[0].filepath
        )
        pli_polyfile = dfmt.geodataframe_to_PolyFile(self.gdf)
        pli_polyfile.save(filename)

    def add(self, thin_dam: gpd.GeoDataFrame) -> None:
        """Append a thin dam to the collection.

        Parameters
        ----------
        thin_dam : gpd.GeoDataFrame
            GeoDataFrame row (or rows) representing the thin dam to add.
        """
        thin_dam.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, thin_dam], ignore_index=True)

    def add_xy(self, x: list, y: list) -> None:
        """Add a thin dam from x/y coordinate lists.

        Parameters
        ----------
        x : list of float
            X-coordinates of the thin-dam vertices.
        y : list of float
            Y-coordinates of the thin-dam vertices.

        Raises
        ------
        ValueError
            If *x* and *y* have different lengths.
        """
        if len(x) != len(y):
            raise ValueError("x and y must be the same length")
        thin_dam = gpd.GeoDataFrame(
            geometry=[shapely.geometry.LineString(zip(x, y))], crs=self.model.crs
        )
        self.gdf = pd.concat([self.gdf, thin_dam], ignore_index=True)

    def delete(self, index: int) -> None:
        """Remove a thin dam by row index.

        Parameters
        ----------
        index : int
            Integer row index of the thin dam to remove.
        """
        if len(self.gdf.index) < index + 1:
            print("Index exceeds length!")
        self.gdf = self.gdf.drop(index).reset_index(drop=True)
        return

    def clear(self) -> None:
        """Remove all thin dams."""
        self.gdf = gpd.GeoDataFrame()

    def list_names(self) -> list:
        """Return a list of thin-dam names.

        Returns
        -------
        list of str
            Names of all thin dams in the collection.
        """
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names
