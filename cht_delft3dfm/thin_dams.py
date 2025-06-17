# -*- coding: utf-8 -*-
"""
Created on May 15 2025
@author: roelvink
"""
import os
import geopandas as gpd
import shapely
import pandas as pd
import hydrolib.core.dflowfm as hcdfm
import dfm_tools as dfmt

class Delft3DFMThinDams:
    def __init__(self, model):
        self.model = model
        self.gdf  = gpd.GeoDataFrame()

    def read(self):
        # Read in all thin dams
        if not self.model.input.geometry.thindamfile:
            return

        filename = os.path.join(self.model.path, self.model.input.geometry.thindamfile[0].filepath)
        data = hcdfm.PolyFile(filename) #works with polyfile
        self.gdf = dfmt.PolyFile_to_geodataframe_linestrings(data,crs=self.model.crs) 

    def write(self):

        if not self.model.input.geometry.thindamfile:
            return
        if len(self.gdf.index)==0:
            return

        filename = os.path.join(self.model.path, self.model.input.geometry.thindamfile[0].filepath)
        pli_polyfile = dfmt.geodataframe_to_PolyFile(self.gdf)
        pli_polyfile.save(filename)
        
    def add(self, thin_dam):
        # Thin dam may be a gdf or shapely geometry
        thin_dam.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, thin_dam], ignore_index=True)

    def add_xy(self, x, y):
        # Add a thin dam by providing x and y coordinates
        # x and y are lists of the same length
        # Create linestring geometry
        if len(x) != len(y):
            raise ValueError("x and y must be the same length")
        thin_dam = gpd.GeoDataFrame(geometry=[shapely.geometry.LineString(zip(x, y))],
                                    crs=self.model.crs)
        # Create a new row in the gdf
        self.gdf = pd.concat([self.gdf, thin_dam], ignore_index=True)


    def delete(self, index):
        if len(self.gdf.index) < index + 1:
            print("Index exceeds length!")    
        self.gdf = self.gdf.drop(index).reset_index(drop=True)
        return
        
    def clear(self):
        self.gdf  = gpd.GeoDataFrame()

    def list_names(self):
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names
