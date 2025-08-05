# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:24:49 2022

@author: ormondt
"""
import time
import os
import numpy as np
from matplotlib import path
from pyproj import CRS, Transformer
import shapely

import xugrid as xu
import xarray as xr
import warnings
import geopandas as gpd
np.warnings = warnings

import pandas as pd

import dfm_tools as dfmt

import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image

class Delft3DFMGrid:
    def __init__(self, model):
        self.model = model
        self.x0 = None
        self.y0 = None
        self.nmax = None
        self.mmax = None
        self.data = None
        self.exterior = gpd.GeoDataFrame()
        self.refinement_depth = False
        self.refinement_polygon = []
        self.min_edge_size = None

    def read(self, file_name=None):
        if file_name == None:
            if not self.model.input.geometry.netfile.filepath: 
                self.model.input.geometry.netfile.filepath = "flow_net.nc"
            file_name = os.path.join(self.model.path, self.model.input.geometry.netfile.filepath)
        self.data = xu.open_dataset(file_name)
        self.data.close()
        self.type = next(iter(self.data.dims.mapping))
        self.nr_cells = self.data.dims[self.type]
        self.get_exterior()
        # crd_dict = self.data["crs"].attrs
        if self.data.grid.projected:
            print("Could not find CRS in netcdf file")   # to do  
        else:
            self.model.crs = CRS(4326)

    def write(self, file_name=None, version=0):
        if file_name == None:
            if not self.model.input.geometry.netfile.filepath: 
                self.model.input.geometry.netfile.filepath = "flow_net.nc"
            file_name = os.path.join(self.model.path, self.model.input.geometry.netfile.filepath)
        attrs = self.data.attrs
        ds = self.data.ugrid
        ds.attrs = attrs
        ds.to_netcdf(file_name)

    def build(self,
              x0,
              y0,
              nmax,
              mmax,
              dx=0.1,
              dy=0.1,
              bathymetry_list=None,
              bathymetry_database=None,
              refinement_depth = 0,
              refinement_polygon=None,
              min_edge_size = 500):

        print("Building mesh ...")

        # Refinement type
        self.type = "ugrid"

        start = time.time()

        print("Getting cells ...")

        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.nmax = nmax
        self.mmax = mmax
        self.refinement_depth = refinement_depth
        self.refinement_polygon = refinement_polygon
        self.min_edge_size = min_edge_size

        # if bathymetry_list:
        #     self.bathymetry_list= bathymetry_list
        # self.refinement_polygons = refinement_polygons

        # Make regular grid
        # self.get_regular_grid()
        self.lon_min, self.lat_min = x0, y0
        self.lon_max = x0 + mmax*dx
        self.lat_max = y0 + nmax*dy
        self.mk = dfmt.make_basegrid(self.lon_min, self.lon_max, self.lat_min, self.lat_max, dx=dx, dy=dy, crs=self.model.crs)
        self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)

        self.refine(bathymetry_list, bathymetry_database)

        # convert to xugrid
        # self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
        # self.add_attributes()

        # Interpolate bathymetry onto the grid
        if bathymetry_list:
            self.set_bathymetry(bathymetry_list, bathymetry_database)
        else:
            print("No depth values assigned. Please select bathymetry ...")

        # Initialize data arrays 
        # self.initialize_data_arrays()

        # Refine all cells 
        # if refinement_polygons is not None:
        #     self.refine_mesh()

        # Initialize data arrays
        # self.initialize_data_arrays()

        # Get all neighbor arrays (mu, mu1, mu2, nu, nu1, nu2)
        # self.get_neighbors()

        # Get uv points
        # self.get_uv_points()

        # Create xugrid dataset 
        # self.to_xugrid()
        
#        self.get_exterior()

        # self.clear_temporary_arrays()

        print("Time elapsed : " + str(time.time() - start) + " s")

    def get_bathymetry(self, bathymetry_sets, bathymetry_database=None, method='grid', dxmin=None, quiet=True):
        """
        Get bathymetry data on the grid or points.
        Parameters
        ----------  
        bathymetry_sets : list
            List of bathymetry sets to use.
        dxmin : float, optional
            Minimum distance [deg/m] between points in the grid. If not provided, it will be set to half of the grid resolution.
        """

        # from cht_bathymetry.bathymetry_database import bathymetry_database

        if not quiet:
            print("Getting bathymetry data ...")
            pass

        if not dxmin:
            dxmin=self.dx/2

        # Make new grid based on outer edges of xx and yy and dxmin
        lon_grid = np.arange(self.lon_min-1, self.lon_max+1, dxmin)
        lat_grid = np.arange(self.lat_min-1, self.lat_max+1, dxmin)
        if method == 'points':
            lon, lat = np.meshgrid(lon_grid, lat_grid)
            lon_grid = lon.ravel()
            lat_grid = lat.ravel()

        if self.data.grid.crs.is_geographic:
            dxmin = dxmin * 111111.0 # dxmin for get_bathymetry_on_grid always in meters

        # Get bathymetry on grid
        zz = bathymetry_database.get_bathymetry_on_grid(lon_grid,
                                                    lat_grid,
                                                    self.model.crs,
                                                    bathymetry_sets,
                                                    coords=method,
                                                    dxmin=dxmin)

        # Make xarray data array
        if method == 'points':
            bathy = xr.DataArray(zz,         
                                coords={'lon': ('points', lon_grid), 'lat': ('points', lat_grid)},
                                dims=['points'])
        elif method == 'grid':
            bathy = xr.DataArray(zz,         
                    coords={'lon': lon_grid, 'lat': lat_grid},
                    dims=['lat', 'lon'])
        return bathy

    def set_bathymetry(self, bathymetry_sets, bathymetry_database=None, quiet=True):
        
        # from cht_bathymetry.bathymetry_database import bathymetry_database

        if not quiet:
            print("Getting bathymetry data ...")
        
        # Check which type of grid
        xx= self.data.obj.mesh2d_node_x
        yy = self.data.obj.mesh2d_node_y

        dxmin = self.dx/2

        if self.data.grid.crs.is_geographic:
            dxmin = dxmin * 111111.0 # dxmin always in meters

        # Get bathy
        zz = bathymetry_database.get_bathymetry_on_points(xx,
                                                    yy,
                                                    dxmin,
                                                    self.model.crs,
                                                    bathymetry_sets)             
        ugrid2d = self.data.grid
        self.data["mesh2d_node_z"] = xu.UgridDataArray(xr.DataArray(data=zz, dims=[ugrid2d.node_dimension]), ugrid2d)

    def refine(self, bathymetry_list, bathymetry_database):
        if self.refinement_depth:
            # Refine based on depth
            self.refine_depth(bathymetry_list, bathymetry_database)
        if self.refinement_polygon:
            # Refine based on polygon input
            self.refine_polygon()
        # self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
            
    def refine_polygon(self, gdf=None):
        from meshkernel import GeometryList
        from meshkernel import MeshRefinementParameters

        if gdf is None:
            gdf= self.refinement_polygon

        # refine with polygon
        for i, geom in enumerate(gdf.geometry):
            x, y = np.array(geom.exterior.xy)
            geometry_list = GeometryList(x, y)
            min_edge_size = gdf.loc[i, "min_edge_size"] if "min_edge_size" in gdf.columns else self.min_edge_size
            mrp = MeshRefinementParameters(min_edge_size=min_edge_size,
                                            connect_hanging_nodes=False)
            self.mk.mesh2d_refine_based_on_polygon(polygon=geometry_list,  mesh_refinement_params=mrp)
        self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
        self.get_datashader_dataframe()

    def refine_depth(self, bathymetry_list, bathymetry_database):

        if bathymetry_list:
            self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
            if self.data.grid.crs.is_geographic:
                dxmin = self.min_edge_size / 111111.0 # For get_bathymetry, dxmin must be in local coordinates (deg/m)
            else:
                dxmin = self.min_edge_size
            bathy = self.get_bathymetry(bathymetry_list, bathymetry_database, dxmin=dxmin, method= 'grid')
            dfmt.refine_basegrid(mk=self.mk, data_bathy_sel=bathy, min_edge_size=self.min_edge_size, connect_hanging_nodes=False)
            self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
        else:
            print("Please select bathymetry first ...")
        self.get_datashader_dataframe()

    def refine_polygon_depth(self, bathymetry_list, bathymetry_database, gdf=None):
        from meshkernel import GriddedSamples, MeshRefinementParameters, RefinementType

        if gdf is None:
            gdf = self.refinement_polygon

        if bathymetry_list:
            self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)

            for _, row in gdf.iterrows():
                geom = row.geometry
                min_edge_size = row.min_edge_size

                if self.data.grid.crs.is_geographic:
                    dxmin = min_edge_size / 111111.0 # For get_bathymetry, dxmin must be in local coordinates (deg/m)
                else:
                    dxmin = min_edge_size
                bathy = self.get_bathymetry(bathymetry_list, bathymetry_database, dxmin=dxmin, method= 'grid')

                bounds = geom.bounds
                bathy_clip = bathy.sel(
                    lon=slice(bounds[0], bounds[2]),
                    lat=slice(bounds[1], bounds[3])
                )

                clipped = (
                    bathy_clip
                    .rio.write_crs(self.model.crs)
                    .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                    .rio.clip([geom], crs=self.model.crs)
                    .where(lambda x: x != 0)
                )

                # Prepare refinement input
                gridded_samples = GriddedSamples(
                    x_coordinates=clipped.lon.to_numpy(),
                    y_coordinates=clipped.lat.to_numpy(),
                    values=clipped.to_numpy().flatten()
                )

                mrp = MeshRefinementParameters(min_edge_size=min_edge_size,
                                            connect_hanging_nodes=False,
                                            refinement_type= 1)
                
                try:
                    self.mk.mesh2d_refine_based_on_gridded_samples(
                        gridded_samples=gridded_samples,
                        mesh_refinement_params=mrp,
                    )
                except:
                    print("Refinement failed. Please check the minimum edge size.")
                    continue    

                self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
        else:
            print("Please select bathymetry first ...")

        self.get_datashader_dataframe()

    def connect_nodes(self, bathymetry_list, bathymetry_database):
        # Connect hanging nodes (work around since separate function is not available in meshkernel)
        if bathymetry_list:
            self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
            dxmintmp = np.abs(self.lon_max - self.lon_min)/3 # temporary bathy to connect nodes
            bathy = self.get_bathymetry(bathymetry_list, bathymetry_database, dxmin= dxmintmp, method= 'grid')
            dfmt.refine_basegrid(mk=self.mk, data_bathy_sel=bathy, min_edge_size=1000000, connect_hanging_nodes=True)
            self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
        else:
            print("Please select bathymetry first ...")

        self.get_datashader_dataframe()

    def delete_cells(self, delete_withcoastlines=False, delete_withpolygon=None):
        # Options to delete land area
        if delete_withcoastlines:
            dfmt.meshkernel_delete_withcoastlines(mk=self.mk, res='h')
        if delete_withpolygon is not None:
            dfmt.meshkernel_delete_withgdf(mk=self.mk, coastlines_gdf=delete_withpolygon)
        dfmt.meshkernel_get_illegalcells(mk=self.mk)
        self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
        self.get_datashader_dataframe()

    # def generate_bnd(self, bnd_withcoastlines=False, bnd_withpolygon=None):
    #     from shapely import MultiPolygon, LineString, MultiLineString
    #     from shapely.ops import linemerge

    #     # Options to delete land area
    #     if bnd_withcoastlines:
    #         bnd_gdf = dfmt.generate_bndpli_cutland(mk=self.mk, res='h', buffer=0.01)
    #     if bnd_withpolygon is not None:
    #         mesh_bnds = self.mk.mesh2d_get_mesh_boundaries_as_polygons()
    #         if mesh_bnds.geometry_separator in mesh_bnds.x_coordinates:
    #             raise Exception('use dfmt.generate_bndpli_cutland() on an uncut grid')
    #         mesh_bnds_xy = np.c_[mesh_bnds.x_coordinates,mesh_bnds.y_coordinates]
    #         pol_gdf = bnd_withpolygon
    
    #         meshbnd_ls = LineString(mesh_bnds_xy)
    #         pol_mp = MultiPolygon(pol_gdf.geometry.tolist())
    #         bnd_ls = meshbnd_ls.intersection(pol_mp)
    
    #         #attempt to merge MultiLineString to single LineString
    #         if isinstance(bnd_ls,MultiLineString):
    #             print('attemting to merge lines in MultiLineString to single LineString (if connected)')
    #             bnd_ls = linemerge(bnd_ls)
    
    #         #convert MultiLineString/LineString to GeoDataFrame
    #         if isinstance(bnd_ls,MultiLineString):
    #             bnd_gdf = gpd.GeoDataFrame(geometry=list(bnd_ls.geoms))
    #         elif isinstance(bnd_ls,LineString):
    #             bnd_gdf = gpd.GeoDataFrame(geometry=[bnd_ls])
    #         bnd_gdf.crs = pol_gdf.crs
    
    #     bnd_gdf_interp = dfmt.interpolate_bndpli(bnd_gdf,res=0.06)
    #     pli_polyfile = dfmt.geodataframe_to_PolyFile(bnd_gdf_interp, name='flow_bnd')
    #     poly_file = os.path.join(self.model.path, 'bnd.pli')
    #     pli_polyfile.save(poly_file)
    #     return bnd_gdf_interp

    def snap_to_grid(self, polyline, max_snap_distance=1.0):
        if len(polyline) == 0:
            return gpd.GeoDataFrame()
        geom_list = []
        for iline, line in polyline.iterrows():
            geom = line["geometry"]
            if geom.geom_type == 'LineString':
                geom_list.append(geom)
        gdf = gpd.GeoDataFrame({'geometry': geom_list})    
        print("Snapping to grid ...")
        snapped_uds, snapped_gdf = xu.snap_to_grid(gdf, self.data.grid, max_snap_distance=max_snap_distance)
        print("Snapping to grid done.")
        snapped_gdf = snapped_gdf.set_crs(self.model.crs)
        return snapped_gdf

    def face_coordinates(self):
        # if self.data is None:
        #     return None, None
        xy = self.data.grid.face_coordinates
        return xy[:, 0], xy[:,1]

    def get_exterior(self):
        try:
            indx = self.data.grid.edge_node_connectivity[self.data.grid.exterior_edges, :]
            x = self.data.grid.node_x[indx]
            y = self.data.grid.node_y[indx]
            # Make linestrings from numpy arrays x and y
            linestrings = [shapely.LineString(np.column_stack((x[i], y[i]))) for i in range(len(x))]
            # Merge linestrings
            merged = shapely.ops.linemerge(linestrings)
            # Merge polygons
            polygons = shapely.ops.polygonize(merged)
    #        polygons = shapely.simplify(polygons, self.dx)
            self.exterior = gpd.GeoDataFrame(geometry=list(polygons), crs=self.model.crs)
        except:
            self.exterior = gpd.GeoDataFrame()    
    
    def bounds(self, crs=None, buffer=0.0):
        """Returns list with bounds (lon1, lat1, lon2, lat2), with buffer (default 0.0) and in any CRS (default : same CRS as model)"""
        if crs is None:
            crs = self.crs
        # Convert exterior gdf to WGS 84
        lst = self.exterior.to_crs(crs=crs).total_bounds.tolist()
        dx = lst[2] - lst[0]
        dy = lst[3] - lst[1]
        lst[0] = lst[0] - buffer * dx
        lst[1] = lst[1] - buffer * dy
        lst[2] = lst[2] + buffer * dx
        lst[3] = lst[3] + buffer * dy
        return lst

    def get_datashader_dataframe(self):
        # Create a dataframe with line elements
        x1 = self.data.grid.edge_node_coordinates[:,0,0]
        x2 = self.data.grid.edge_node_coordinates[:,1,0]
        y1 = self.data.grid.edge_node_coordinates[:,0,1]
        y2 = self.data.grid.edge_node_coordinates[:,1,1]
        transformer = Transformer.from_crs(self.model.crs,
                                            3857,
                                            always_xy=True)
        x1, y1 = transformer.transform(x1, y1)
        x2, y2 = transformer.transform(x2, y2)
        self.df = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))

    def map_overlay(self, file_name, xlim=None, ylim=None, color="black", width=800):
        if self.data is None:
            # No grid (yet)
            return False
        try:
            if not hasattr(self, "df"):
                self.df = None
            if self.df is None: 
                self.get_datashader_dataframe()

            transformer = Transformer.from_crs(4326,
                                        3857,
                                        always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)
            cvs = ds.Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)
            agg = cvs.line(self.df, x=['x1', 'x2'], y=['y1', 'y2'], axis=1)
            img = tf.shade(agg)
            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True
        except Exception as e:
            return False

def inpolygon(xq, yq, p):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
    return p.contains_points(q).reshape(shape)

def binary_search(val_array, vals):    
    indx = np.searchsorted(val_array, vals) # ind is size of vals 
    not_ok = np.where(indx==len(val_array))[0] # size of vals, points that are out of bounds
    indx[np.where(indx==len(val_array))[0]] = 0 # Set to zero to avoid out of bounds error
    is_ok = np.where(val_array[indx] == vals)[0] # size of vals
    indices = np.zeros(len(vals), dtype=int) - 1
    indices[is_ok] = indx[is_ok]
    indices[not_ok] = -1
    return indices


def gdf2list(gdf_in):
   gdf_out = []
   for feature in gdf_in.iterfeatures():
      gdf_out.append(gpd.GeoDataFrame.from_features([feature]))
   return gdf_out
