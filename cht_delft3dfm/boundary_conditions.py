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
        self.bcafile = 'waterlevel_astro.bc'
        self.gdf_points = gpd.GeoDataFrame() # Contains the tide points
        self.bzsfile = 'waterlevel_timeseries.bc' 

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
        self.gdf_points = mline2point(self.gdf) # convert line to points

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
        self.gdf_points = mline2point(self.gdf) # convert line to points

    # def generate_tide(self, tidemodel = 'FES2014', poly_file=None, ext_file_new=None):
    #     if not poly_file:
    #         poly_file = os.path.join(self.model.path, 'bnd.pli')
    #     if not ext_file_new:
    #         ext_file_new = os.path.join(self.model.path, 'bc_new.ext')

    #     ext_new = hcdfm.ExtModel()

    #     dfmt.interpolate_tide_to_bc(ext_new=ext_new, tidemodel=tidemodel, file_pli=poly_file)
    #     ext_new.save(filepath=ext_file_new, path_style='windows')
    #     return ext_new

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

    def write_boundary_conditions_timeseries(self):

        if not self.bzsfile:
            # No file name
            return

        if len(self.gdf_points.index)==0:
            # No points
            return
        
        # WL      
        filename = os.path.join(self.model.path, self.bzsfile)

        tref = pd.to_datetime(str(self.model.input.time.refdate), format='%Y%m%d')
        tref_str = tref.strftime('%Y-%m-%d')  # Only date, no time

        fid = open(filename, "w")

        for ip, point in self.gdf_points.iterrows():
            timeseries = point["timeseries"]
            # name is included in the self.gdf_points
            name = point["name"]
            
            if timeseries is None or timeseries.empty:
                continue

            fid.write(f"[forcing]\n")
            fid.write(f"Name                            = {name}\n")
            fid.write(f"Function                        = timeseries\n")
            fid.write(f"Time-interpolation              = linear\n")
            fid.write(f"Quantity                        = time\n")
            fid.write(f"Unit                            = minutes since {tref_str}\n")
            fid.write(f"Quantity                        = waterlevelbnd\n")
            fid.write(f"Unit                            = m\n")

            for timestamp, row in timeseries.iterrows():
                minutes_since_ref = (timestamp - tref).total_seconds() / 60.0
                fid.write(f"{minutes_since_ref:10.2f}{row['wl']:10.2f}\n")
            
            fid.write(f"\n")
        fid.close()

    def write_ext_wl(self, poly_file=None, ext_file_new=None, forcingtype='bca'):
        """
        Write the boundary conditions to an external file.
        """
        if not poly_file:
            poly_file = os.path.join(self.model.path, 'bnd.pli')

        if not ext_file_new:
            ext_file_new = os.path.join(self.model.path, 'bnd_new.ext')

        if forcingtype == 'bca':
            forcingfile = self.bcafile
        elif forcingtype == 'bzs':
            forcingfile = self.bzsfile
        
        if not self.model.input.external_forcing.extforcefilenew:
                self.model.input.external_forcing.extforcefilenew = hcdfm.ExtModel()
                self.model.input.external_forcing.extforcefilenew.filepath = Path(ext_file_new)                                        

        #add boundary to new ext file
        boundary_object = hcdfm.Boundary(quantity='waterlevelbnd', #the FM quantity for tide is also waterlevelbnd
                                        locationfile=poly_file,
                                        forcingfile= forcingfile)
        self.model.input.external_forcing.extforcefilenew.boundary.append(boundary_object)

        #Save new ext forcing file
        self.model.input.external_forcing.extforcefilenew.save(filepath=ext_file_new,path_style='windows')

    def set_timeseries(self,
                       shape="constant",
                       timestep=600.0,
                       offset=0.0,
                       amplitude=1.0,
                       phase=0.0,
                       period=43200.0,
                       peak=1.0,
                       tpeak=86400.0,
                       duration=43200.0):

        # Applies time series boundary conditions for each point
        # Step 1: Reference time from refDate
        tref = pd.to_datetime(str(self.model.input.time.refdate), format='%Y%m%d')

        # Step 2: Start and Stop
        if self.model.input.time.startdatetime:
            t_start = pd.to_datetime(str(self.model.input.time.startdatetime), format='%Y%m%d%H%M%S')
            t_stop = pd.to_datetime(str(self.model.input.time.stopdatetime), format='%Y%m%d%H%M%S')
        else:
            # Use tstart and tstop relative to tref, in model time units
            tunit = self.model.input.time.tunit.upper()  # Should be 'S', 'M', etc.
            unit_map = {'S': 'seconds', 'M': 'minutes', 'H': 'hours', 'D': 'days'}
            tdelta_unit = unit_map.get(tunit, 'seconds')
            t_start = tref + pd.to_timedelta(self.model.input.time.tstart, unit=tdelta_unit)
            t_stop = tref + pd.to_timedelta(self.model.input.time.tstop, unit=tdelta_unit)

        # Time in seconds since tref
        t0 = (t_start - tref).total_seconds()
        t1 = (t_stop - tref).total_seconds()
        dt = t1 - t0 if shape == "constant" else timestep
        time = np.arange(t0, t1 + dt, dt)

        # Step 3: Water level generation
        nt = len(time)
        if shape == "constant":
            wl = [offset] * nt
        elif shape == "sine":
            wl = offset + amplitude * np.sin(2 * np.pi * time / period + phase * np.pi / 180)
        elif shape == "gaussian":
            wl = offset + peak * np.exp(- ((time - tpeak) / (0.25 * duration))**2)
        elif shape == "astronomical":
            # Not implemented
            return

        # Step 4: Create time series
        times = pd.date_range(start= t_start,
                              end=t_stop,
                              freq=pd.tseries.offsets.DateOffset(seconds=dt))                              

        if "timeseries" not in self.gdf_points.columns:
            self.gdf_points["timeseries"] = None
        self.gdf_points["timeseries"] = self.gdf_points["timeseries"].astype(object)
        df = pd.DataFrame({'wl': wl}, index=times)
        self.gdf_points["timeseries"] = [df.copy() for _ in range(len(self.gdf_points))]


def mline2point(gdf):
    """
    Convert (Multi)LineString to Point
    """
    from shapely.geometry import Point

    point_records = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        base_name  = row['name']  # Preserve the 'name' column

        if geom.geom_type == 'LineString':
            coords = geom.coords
        elif geom.geom_type == 'MultiLineString':
            coords = []
            for part in geom.geoms:
                coords.extend(part.coords)
        else:
            continue  # Skip unsupported geometry types

        for i, coord in enumerate(coords, start=1):
            new_name = f"{base_name}_{str(i).zfill(4)}"
            point_records.append({'geometry': Point(coord), 'name': new_name})

    # Create new GeoDataFrame
    gdf_points = gpd.GeoDataFrame(point_records, crs=gdf.crs)

    return gdf_points