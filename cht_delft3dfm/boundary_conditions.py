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
from cht_utils.pli_file import pli2gdf, gdf2pli
from pathlib import Path
class Delft3DFMBoundaryConditions:

    def __init__(self, model):
        self.model = model
        self.forcing = "timeseries"
        self.gdf = gpd.GeoDataFrame()
        self.times = []
        self.bcafile = 'waterlevel.bc'
        self.gdf_points = gpd.GeoDataFrame() # Contains the tide points

    def generate_bnd(self, bnd_withcoastlines=False, bnd_withpolygon=None, resolution=0.06):
        from shapely import MultiPolygon, LineString, MultiLineString
        from shapely.ops import linemerge

        if not self.model.grid.mk:
            print('"First generate the grid')
            return

        # Options to delete land area
        if bnd_withcoastlines:
            bnd_gdf = dfmt.generate_bndpli_cutland(mk=self.model.grid.mk, res='h', buffer=0.01)
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
            names = pol_gdf.get("name", [None])
            if isinstance(bnd_ls, MultiLineString):
                geoms = list(bnd_ls.geoms)
                names = (list(names) * len(geoms))[:len(geoms)]
                bnd_gdf = gpd.GeoDataFrame({"geometry": geoms, "name": names})
            else:
                bnd_gdf = gpd.GeoDataFrame({"geometry": [bnd_ls], "name": [names[0]]})
            bnd_gdf.crs = pol_gdf.crs

        self.gdf = dfmt.interpolate_bndpli(bnd_gdf,res=resolution)

    def write_bnd(self, file_name=None):
        if not file_name:
            file_name = os.path.join(self.model.path, 'bnd.pli')
        # pli_polyfile = dfmt.geodataframe_to_PolyFile(self.gdf, name=bnd_name)
        # pli_polyfile.save(file_name)
        gdf2pli(self.gdf, file_name, add_point_name=True)

    def load_bnd(self, file_name=None):
        if not file_name:
            file_name = os.path.join(self.model.path, 'bnd.pli')
        # polyfile_object = hcdfm.PolyFile(file_name)
        # self.gdf = dfmt.PolyFile_to_geodataframe_linestrings(polyfile_object, crs='4326')
        self.gdf = pli2gdf(file_name)
        self.gdf.crs = self.model.crs

    def generate_tide(self, tidemodel = 'FES2014', poly_file=None, ext_file_new=None):
        if not poly_file:
            poly_file = os.path.join(self.model.path, 'bnd.pli')
        if not ext_file_new:
            ext_file_new = os.path.join(self.model.path, 'bc_new.ext')

        ext_new = hcdfm.ExtModel()

        dfmt.interpolate_tide_to_bc(ext_new=ext_new, tidemodel=tidemodel, file_pli=poly_file)
        ext_new.save(filepath=ext_file_new, path_style='windows')
        return ext_new

    def write_boundary_conditions_astro(self):

        if not self.bcafile:
            # No file name
            return

        if len(self.gdf_points.index)==0:
            # No points
            return
        
        # WL      
        filename = os.path.join(self.model.path, self.bcafile)

        fid = open(filename, "w")

        for ip, point in self.gdf_points.iterrows():
            astro = point["astro"]
            # name is included in the self.gdf_points
            name = point["name"]
            fid.write(f"[forcing]\n")
            fid.write(f"Name                            = {name}\n")
            fid.write(f"Function                        = astronomic\n")
            fid.write(f"Quantity                        = astronomic component\n")
            fid.write(f"Unit                            = -\n")
            fid.write(f"Quantity                        = waterlevelbnd amplitude\n")
            fid.write(f"Unit                            = m\n")
            fid.write(f"Quantity                        = waterlevelbnd phase\n")
            fid.write(f"Unit                            = deg\n")
            for constituent, row in astro.iterrows():
                fid.write(f"{constituent:6s}{row['amplitude']:10.5f}{row['phase']:10.2f}\n")
            fid.write(f"\n")
        fid.close()


    def write_ext_wl(self, poly_file=None, ext_file_new=None):
        """
        Write the boundary conditions to an external file.
        """
        if not poly_file:
            poly_file = os.path.join(self.model.path, 'bnd.pli')

        if not ext_file_new:
            ext_file_new = os.path.join(self.model.path, 'bnd_new.ext')

        if not self.model.input.external_forcing.extforcefilenew:
                self.model.input.external_forcing.extforcefilenew = hcdfm.ExtModel()
                self.model.input.external_forcing.extforcefilenew.filepath = Path(ext_file_new)                                        

        #add boundary to new ext file
        boundary_object = hcdfm.Boundary(quantity='waterlevelbnd', #the FM quantity for tide is also waterlevelbnd
                                        locationfile=poly_file,
                                        forcingfile=self.bcafile)
        self.model.input.external_forcing.extforcefilenew.boundary.append(boundary_object)

        #Save new ext forcing file
        self.model.input.external_forcing.extforcefilenew.save(filepath=ext_file_new,path_style='windows')
