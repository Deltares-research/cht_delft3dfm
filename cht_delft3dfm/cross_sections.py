"""Delft3D-FM cross-section definition: read, write, add, and delete cross sections."""

import os
from typing import Union

import dfm_tools as dfmt
import geopandas as gpd
import hydrolib.core.dflowfm as hcdfm
import pandas as pd


class Delft3DFMCrossSections:
    """Container for Delft3D-FM observation cross sections.

    Parameters
    ----------
    model : Delft3DFM
        Parent model instance.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """Read cross sections from the file referenced in the model input.

        Does nothing if no cross-section file is configured.
        """
        if not self.model.input.output.crsfile:
            return

        filename = os.path.join(
            self.model.path, self.model.input.output.crsfile[0].filepath
        )
        data = hcdfm.PolyFile(filename)
        self.gdf = dfmt.PolyFile_to_geodataframe_linestrings(data, crs=self.model.crs)

    def write(self) -> None:
        """Write cross sections to the file referenced in the model input.

        Does nothing if no cross-section file is configured or the GeoDataFrame
        is empty.
        """
        if not self.model.input.output.crsfile:
            return
        if len(self.gdf.index) == 0:
            return

        filename = os.path.join(
            self.model.path, self.model.input.output.crsfile[0].filepath
        )
        pli_polyfile = dfmt.geodataframe_to_PolyFile(self.gdf)
        pli_polyfile.save(filename)

    def add(self, cross_section: gpd.GeoDataFrame) -> None:
        """Append a cross section to the collection.

        Parameters
        ----------
        cross_section : gpd.GeoDataFrame
            GeoDataFrame row representing the cross section to add.
        """
        cross_section.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, cross_section], ignore_index=True)

    def delete(self, name_or_index: Union[str, int]) -> None:
        """Remove a cross section by name or index.

        Parameters
        ----------
        name_or_index : str or int
            Name string or integer row index of the cross section to remove.
        """
        if isinstance(name_or_index, str):
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print(f"Cross section {name} not found!")
        else:
            index = name_or_index
            if len(self.gdf.index) < index + 1:
                print("Index exceeds length!")
            self.gdf = self.gdf.drop(index).reset_index(drop=True)
            return

    def clear(self) -> None:
        """Remove all cross sections."""
        self.gdf = gpd.GeoDataFrame()

    def list_names(self) -> list:
        """Return a list of cross-section names.

        Returns
        -------
        list of str
            Names of all cross sections in the collection.
        """
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names
