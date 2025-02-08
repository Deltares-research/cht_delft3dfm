# -*- coding: utf-8 -*-
"""
@author: roelvink
"""
import os
import numpy as np
import geopandas as gpd
import shapely
import pandas as pd
# from pandas.tseries.offsets import DateOffset
from tabulate import tabulate
from pyproj import Transformer
from cht_utils.deltares_ini import IniStruct
from cht_tide.tide_predict import predict
import dfm_tools as dfmt
import hydrolib.core.dflowfm as hcdfm


class Delft3DFMBoundaryConditions:

    def __init__(self, model):
        self.model = model
        self.forcing = "timeseries"
        self.gdf = gpd.GeoDataFrame()
        self.times = []

    def generate_bnd(self, bnd_withcoastlines=False, bnd_withpolygon=None):
        from shapely import MultiPolygon, LineString, MultiLineString
        from shapely.ops import linemerge

        if not self.model.grid.mk:
            print('"First generate the grid')
            return

        # Options to delete land area
        if bnd_withcoastlines:
            self.gdf = dfmt.generate_bndpli_cutland(mk=self.model.grid.mk, res='h', buffer=0.01)
        if bnd_withpolygon is not None:
            mesh_bnds = self.model.grid.mk.mesh2d_get_mesh_boundaries_as_polygons()
            if mesh_bnds.geometry_separator in mesh_bnds.x_coordinates:
                raise Exception('use dfmt.generate_bndpli_cutland() on an uncut grid')
            mesh_bnds_xy = np.c_[mesh_bnds.x_coordinates,mesh_bnds.y_coordinates]
            pol_gdf = bnd_withpolygon
    
            meshbnd_ls = LineString(mesh_bnds_xy)
            pol_mp = MultiPolygon(pol_gdf.geometry.tolist())
            bnd_ls = meshbnd_ls.intersection(pol_mp)
    
            #attempt to merge MultiLineString to single LineString
            if isinstance(bnd_ls,MultiLineString):
                print('attemting to merge lines in MultiLineString to single LineString (if connected)')
                bnd_ls = linemerge(bnd_ls)
    
            #convert MultiLineString/LineString to GeoDataFrame
            if isinstance(bnd_ls,MultiLineString):
                bnd_gdf = gpd.GeoDataFrame(geometry=list(bnd_ls.geoms))
            elif isinstance(bnd_ls,LineString):
                bnd_gdf = gpd.GeoDataFrame(geometry=[bnd_ls])
            bnd_gdf.crs = pol_gdf.crs
            self.gdf = dfmt.interpolate_bndpli(bnd_gdf,res=0.06)

    def write_bnd(self, bnd_name= None, file_name=None):
        if not bnd_name:
            bnd_name = 'flow_bnd'
        if not file_name:
            file_name = os.path.join(self.model.path, 'bnd.pli')
        pli_polyfile = dfmt.geodataframe_to_PolyFile(self.gdf, name=bnd_name)
        pli_polyfile.save(file_name)
        
    def load_bnd(self, file_name=None):
        if not file_name:
            file_name = os.path.join(self.model.path, 'bnd.pli')
        polyfile_object = hcdfm.PolyFile(file_name)
        self.gdf = dfmt.PolyFile_to_geodataframe_linestrings(polyfile_object, crs='4326')

    def generate_tide(self, tidemodel = 'FES2014', poly_file=None):
        if not poly_file:
            poly_file = os.path.join(self.model.path, 'bnd.pli')
        ext_file_new = os.path.join(self.model.path, 'bc_new.ext')

        ext_new = hcdfm.ExtModel()

        dfmt.interpolate_tide_to_bc(ext_new=ext_new, tidemodel=tidemodel, file_pli=poly_file)
        ext_new.save(filepath=ext_file_new, path_style='windows')
        return ext_new

