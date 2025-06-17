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

class Delft3DFMCrossSections:
    def __init__(self, model):
        self.model = model
        self.gdf  = gpd.GeoDataFrame()

    def read(self):
        # Read in all cross sections
        if not self.model.input.output.crsfile:
            return

        filename = os.path.join(self.model.path, self.model.input.output.crsfile[0].filepath)
        data = hcdfm.PolyFile(filename) #works with polyfile
        self.gdf = dfmt.PolyFile_to_geodataframe_linestrings(data,crs=self.model.crs) 
    
    def write(self):

        if not self.model.input.output.crsfile:
            return
        if len(self.gdf.index)==0:
            return

        filename = os.path.join(self.model.path, self.model.input.output.crsfile[0].filepath)
        pli_polyfile = dfmt.geodataframe_to_PolyFile(self.gdf)
        pli_polyfile.save(filename)

    def add(self, cross_section):
        cross_section.set_crs(self.model.crs)
        self.gdf = pd.concat([self.gdf, cross_section], ignore_index=True)

    def delete(self, name_or_index):
        if type(name_or_index) == str:
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print("Cross section " + name + " not found!")    
        else:
            index = name_or_index
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
