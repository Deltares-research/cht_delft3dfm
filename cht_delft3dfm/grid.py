"""Delft3D-FM unstructured grid: build, refine, read, write, and overlay utilities."""

import os
import time
import warnings
from typing import Optional

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
import xugrid as xu
from matplotlib import path
from pyproj import CRS, Transformer
from shapely.geometry import box

np.warnings = warnings

import datashader as ds
import datashader.transfer_functions as tf
import dfm_tools as dfmt
import pandas as pd
from datashader.utils import export_image


class Delft3DFMGrid:
    """Unstructured grid handler for Delft3D-FM models.

    Parameters
    ----------
    model : Delft3DFM
        Parent model instance.
    """

    def __init__(self, model) -> None:
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

    def read(self, file_name: Optional[str] = None) -> None:
        """Read an unstructured grid from a NetCDF file.

        Parameters
        ----------
        file_name : str, optional
            Path to the NetCDF grid file.  If ``None``, the path is taken from
            the model input configuration.
        """
        if file_name is None:
            if not self.model.input.geometry.netfile.filepath:
                self.model.input.geometry.netfile.filepath = "flow_net.nc"
            file_name = os.path.join(
                self.model.path, self.model.input.geometry.netfile.filepath
            )
        self.data = xu.open_dataset(file_name)
        self.data.close()
        self.type = next(iter(self.data.dims.mapping))
        self.nr_cells = self.data.dims[self.type]
        self.get_exterior()
        if self.data.grid.projected:
            print("Could not find CRS in netcdf file")
        else:
            self.model.crs = CRS(4326)

    def write(self, file_name: Optional[str] = None, version: int = 0) -> None:
        """Write the grid to a NetCDF file.

        Parameters
        ----------
        file_name : str, optional
            Output file path.  If ``None``, the path is taken from the model
            input configuration.
        version : int, optional
            File format version flag (currently unused, default ``0``).
        """
        if file_name is None:
            if not self.model.input.geometry.netfile.filepath:
                self.model.input.geometry.netfile.filepath = "flow_net.nc"
            file_name = os.path.join(
                self.model.path, self.model.input.geometry.netfile.filepath
            )
        attrs = self.data.attrs
        ds = self.data.ugrid
        ds.attrs = attrs
        ds.to_netcdf(file_name)

    def build(
        self,
        x0: float,
        y0: float,
        nmax: int,
        mmax: int,
        dx: float = 0.1,
        dy: float = 0.1,
        bathymetry_list=None,
        bathymetry_database=None,
        data_catalog=None,
        refinement_depth: int = 0,
        refinement_polygon=None,
        min_edge_size: float = 500,
    ) -> None:
        """Build a new base grid and optionally apply refinement and bathymetry.

        Parameters
        ----------
        x0 : float
            Longitude of the south-west corner.
        y0 : float
            Latitude of the south-west corner.
        nmax : int
            Number of rows in the base grid.
        mmax : int
            Number of columns in the base grid.
        dx : float, optional
            Base cell size in the x-direction (default ``0.1``).
        dy : float, optional
            Base cell size in the y-direction (default ``0.1``).
        bathymetry_list : list, optional
            Ordered list of bathymetry dataset descriptors.
        bathymetry_database : object, optional
            Legacy cht_bathymetry database.
        data_catalog : DataCatalog, optional
            HydroMT data catalog.
        refinement_depth : int, optional
            Depth-based refinement level (default ``0``).
        refinement_polygon : gpd.GeoDataFrame, optional
            Polygon GeoDataFrame defining refinement areas.
        min_edge_size : float, optional
            Minimum allowed edge length after refinement (default ``500``).
        """
        print("Building mesh ...")

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

        self.lon_min, self.lat_min = x0, y0
        self.lon_max = x0 + mmax * dx
        self.lat_max = y0 + nmax * dy
        self.mk = dfmt.make_basegrid(
            self.lon_min,
            self.lon_max,
            self.lat_min,
            self.lat_max,
            dx=dx,
            dy=dy,
            crs=self.model.crs,
        )
        self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)

        self.refine(bathymetry_list, bathymetry_database, data_catalog=data_catalog)

        if bathymetry_list:
            self.set_bathymetry(
                bathymetry_list, bathymetry_database, data_catalog=data_catalog
            )
        else:
            print("No depth values assigned. Please select bathymetry ...")

        print(f"Time elapsed : {time.time() - start} s")

    def get_bathymetry(
        self,
        bathymetry_sets: list,
        bathymetry_database=None,
        data_catalog=None,
        method: str = "grid",
        dxmin: Optional[float] = None,
        quiet: bool = True,
    ) -> xr.DataArray:
        """Get bathymetry data on a regular grid or at scattered points.

        Parameters
        ----------
        bathymetry_sets : list
            List of bathymetry dataset descriptors.
        bathymetry_database : object, optional
            Legacy cht_bathymetry database.
        data_catalog : DataCatalog, optional
            HydroMT data catalog.
        method : str, optional
            ``"grid"`` (default) or ``"points"``.
        dxmin : float, optional
            Minimum sampling resolution.  Defaults to half the grid cell size.
        quiet : bool, optional
            Suppress progress messages when ``True`` (default).

        Returns
        -------
        xr.DataArray
            Bathymetry values on the requested grid or point set.
        """
        if not quiet:
            print("Getting bathymetry data ...")

        if not dxmin:
            dxmin = self.dx / 2

        lon_grid = np.arange(self.lon_min - 1, self.lon_max + 1, dxmin)
        lat_grid = np.arange(self.lat_min - 1, self.lat_max + 1, dxmin)

        if data_catalog is not None:
            geom = gpd.GeoDataFrame(
                geometry=[
                    box(
                        self.lon_min - 1,
                        self.lat_min - 1,
                        self.lon_max + 1,
                        self.lat_max + 1,
                    )
                ],
                crs=self.model.crs,
            )
            dxmin_m = dxmin * 111111.0 if self.data.grid.crs.is_geographic else dxmin
            zz = None
            for ds in reversed(bathymetry_sets):
                name = ds.get("elevation", ds.get("name"))
                zmin = ds.get("zmin", -1.0e9)
                zmax = ds.get("zmax", 1.0e9)
                try:
                    da = data_catalog.get_rasterdataset(
                        name, geom=geom, zoom=(dxmin_m, "metre")
                    )
                    vals = da.values.astype(np.float64)
                    vals[(vals < zmin) | (vals > zmax)] = np.nan
                    if zz is None:
                        zz = vals
                    else:
                        mask = np.isnan(zz)
                        zz[mask] = vals[mask]
                except Exception:
                    continue
            if zz is None:
                zz = np.full((len(lat_grid), len(lon_grid)), np.nan)
            bathy = xr.DataArray(
                zz, coords={"lat": da.y.values, "lon": da.x.values}, dims=["lat", "lon"]
            )
            return bathy

        if method == "points":
            lon, lat = np.meshgrid(lon_grid, lat_grid)
            lon_grid = lon.ravel()
            lat_grid = lat.ravel()

        if self.data.grid.crs.is_geographic:
            dxmin = dxmin * 111111.0

        zz = bathymetry_database.get_bathymetry_on_grid(
            lon_grid,
            lat_grid,
            self.model.crs,
            bathymetry_sets,
            coords=method,
            dxmin=dxmin,
        )

        if method == "points":
            bathy = xr.DataArray(
                zz,
                coords={"lon": ("points", lon_grid), "lat": ("points", lat_grid)},
                dims=["points"],
            )
        elif method == "grid":
            bathy = xr.DataArray(
                zz,
                coords={"lon": lon_grid, "lat": lat_grid},
                dims=["lat", "lon"],
            )
        return bathy

    def set_bathymetry(
        self,
        bathymetry_sets: list,
        bathymetry_database=None,
        data_catalog=None,
        quiet: bool = True,
    ) -> None:
        """Interpolate bathymetry onto mesh nodes and store as a node variable.

        Parameters
        ----------
        bathymetry_sets : list
            List of bathymetry dataset descriptors.
        bathymetry_database : object, optional
            Legacy cht_bathymetry database.
        data_catalog : DataCatalog, optional
            HydroMT data catalog.
        quiet : bool, optional
            Suppress progress messages when ``True`` (default).
        """
        if not quiet:
            print("Getting bathymetry data ...")

        xx = self.data.obj.mesh2d_node_x
        yy = self.data.obj.mesh2d_node_y
        dxmin = self.dx / 2

        if data_catalog is not None:
            geom = gpd.GeoDataFrame(
                geometry=[
                    box(
                        float(xx.min()) - dxmin,
                        float(yy.min()) - dxmin,
                        float(xx.max()) + dxmin,
                        float(yy.max()) + dxmin,
                    )
                ],
                crs=self.model.crs,
            )
            dxmin_m = dxmin * 111111.0 if self.data.grid.crs.is_geographic else dxmin
            zz = np.full(len(xx), np.nan)
            for ds in reversed(bathymetry_sets):
                name = ds.get("elevation", ds.get("name"))
                zmin = ds.get("zmin", -1.0e9)
                zmax = ds.get("zmax", 1.0e9)
                try:
                    da = data_catalog.get_rasterdataset(
                        name, geom=geom, zoom=(dxmin_m, "metre")
                    )
                    from scipy.interpolate import RegularGridInterpolator

                    interp = RegularGridInterpolator(
                        (da.y.values, da.x.values),
                        da.values,
                        method="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    vals = interp(np.column_stack([yy.values, xx.values]))
                    vals[(vals < zmin) | (vals > zmax)] = np.nan
                    mask = np.isnan(zz)
                    zz[mask] = vals[mask]
                except Exception:
                    continue
        else:
            if self.data.grid.crs.is_geographic:
                dxmin = dxmin * 111111.0
            zz = bathymetry_database.get_bathymetry_on_points(
                xx,
                yy,
                dxmin,
                self.model.crs,
                bathymetry_sets,
            )

        ugrid2d = self.data.grid
        self.data["mesh2d_node_z"] = xu.UgridDataArray(
            xr.DataArray(data=zz, dims=[ugrid2d.node_dimension]), ugrid2d
        )

    def refine(self, bathymetry_list, bathymetry_database, data_catalog=None) -> None:
        """Apply configured refinement strategies to the mesh.

        Parameters
        ----------
        bathymetry_list : list or None
            Bathymetry dataset descriptors for depth-based refinement.
        bathymetry_database : object or None
            Legacy cht_bathymetry database.
        data_catalog : DataCatalog, optional
            HydroMT data catalog.
        """
        if self.refinement_depth:
            self.refine_depth(
                bathymetry_list, bathymetry_database, data_catalog=data_catalog
            )
        if self.refinement_polygon:
            self.refine_polygon()

    def refine_polygon(self, gdf: Optional[gpd.GeoDataFrame] = None) -> None:
        """Refine the mesh inside one or more polygons.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame, optional
            Polygon GeoDataFrame with an optional ``"min_edge_size"`` column.
            Defaults to ``self.refinement_polygon``.
        """
        from meshkernel import GeometryList, MeshRefinementParameters

        if gdf is None:
            gdf = self.refinement_polygon

        for i, geom in enumerate(gdf.geometry):
            x, y = np.array(geom.exterior.xy)
            geometry_list = GeometryList(x, y)
            min_edge_size = (
                gdf.loc[i, "min_edge_size"]
                if "min_edge_size" in gdf.columns
                else self.min_edge_size
            )
            mrp = MeshRefinementParameters(
                min_edge_size=min_edge_size, connect_hanging_nodes=False
            )
            self.mk.mesh2d_refine_based_on_polygon(
                polygon=geometry_list, mesh_refinement_params=mrp
            )
        self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
        self.get_datashader_dataframe()

    def refine_depth(
        self, bathymetry_list, bathymetry_database, data_catalog=None
    ) -> None:
        """Refine the mesh based on bathymetric depth gradients.

        Parameters
        ----------
        bathymetry_list : list
            Bathymetry dataset descriptors.
        bathymetry_database : object
            Legacy cht_bathymetry database.
        data_catalog : DataCatalog, optional
            HydroMT data catalog.
        """
        if bathymetry_list:
            self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
            if self.data.grid.crs.is_geographic:
                dxmin = self.min_edge_size / 111111.0
            else:
                dxmin = self.min_edge_size
            bathy = self.get_bathymetry(
                bathymetry_list,
                bathymetry_database,
                data_catalog=data_catalog,
                dxmin=dxmin,
                method="grid",
            )
            dfmt.refine_basegrid(
                mk=self.mk,
                data_bathy_sel=bathy,
                min_edge_size=self.min_edge_size,
                connect_hanging_nodes=False,
            )
            self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
        else:
            print("Please select bathymetry first ...")
        self.get_datashader_dataframe()

    def refine_polygon_depth(
        self,
        bathymetry_list,
        bathymetry_database,
        data_catalog=None,
        gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        """Refine the mesh inside polygons using bathymetric samples.

        Parameters
        ----------
        bathymetry_list : list
            Bathymetry dataset descriptors.
        bathymetry_database : object
            Legacy cht_bathymetry database.
        data_catalog : DataCatalog, optional
            HydroMT data catalog.
        gdf : gpd.GeoDataFrame, optional
            Polygon GeoDataFrame with a ``"min_edge_size"`` column.  Defaults
            to ``self.refinement_polygon``.
        """
        from meshkernel import GriddedSamples, MeshRefinementParameters

        if gdf is None:
            gdf = self.refinement_polygon

        if bathymetry_list:
            self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)

            for _, row in gdf.iterrows():
                geom = row.geometry
                min_edge_size = row.min_edge_size

                if self.data.grid.crs.is_geographic:
                    dxmin = min_edge_size / 111111.0
                else:
                    dxmin = min_edge_size
                bathy = self.get_bathymetry(
                    bathymetry_list,
                    bathymetry_database,
                    data_catalog=data_catalog,
                    dxmin=dxmin,
                    method="grid",
                )

                bounds = geom.bounds
                bathy_clip = bathy.sel(
                    lon=slice(bounds[0], bounds[2]), lat=slice(bounds[1], bounds[3])
                )

                clipped = (
                    bathy_clip.rio.write_crs(self.model.crs)
                    .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                    .rio.clip([geom], crs=self.model.crs)
                    .where(lambda x: x != 0)
                )

                gridded_samples = GriddedSamples(
                    x_coordinates=clipped.lon.to_numpy(),
                    y_coordinates=clipped.lat.to_numpy(),
                    values=clipped.to_numpy().flatten(),
                )

                mrp = MeshRefinementParameters(
                    min_edge_size=min_edge_size,
                    connect_hanging_nodes=False,
                    refinement_type=1,
                )

                try:
                    self.mk.mesh2d_refine_based_on_gridded_samples(
                        gridded_samples=gridded_samples,
                        mesh_refinement_params=mrp,
                    )
                except Exception:
                    print("Refinement failed. Please check the minimum edge size.")
                    continue

                self.data = dfmt.meshkernel_to_UgridDataset(
                    mk=self.mk, crs=self.model.crs
                )
        else:
            print("Please select bathymetry first ...")

        self.get_datashader_dataframe()

    def connect_nodes(
        self, bathymetry_list, bathymetry_database, data_catalog=None
    ) -> None:
        """Connect hanging nodes in the mesh as a refinement work-around.

        Parameters
        ----------
        bathymetry_list : list
            Bathymetry dataset descriptors.
        bathymetry_database : object
            Legacy cht_bathymetry database.
        data_catalog : DataCatalog, optional
            HydroMT data catalog.
        """
        if bathymetry_list:
            self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
            dxmintmp = np.abs(self.lon_max - self.lon_min) / 3
            bathy = self.get_bathymetry(
                bathymetry_list,
                bathymetry_database,
                data_catalog=data_catalog,
                dxmin=dxmintmp,
                method="grid",
            )
            dfmt.refine_basegrid(
                mk=self.mk,
                data_bathy_sel=bathy,
                min_edge_size=1000000,
                connect_hanging_nodes=True,
            )
            dfmt.meshkernel_get_illegalcells(mk=self.mk)
            self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
            ortho = self.mk.mesh2d_get_orthogonality()
            if ortho.values.max() > 0.02:
                print(
                    f"Warning: mesh has non-orthogonal cells, max orthogonality = {ortho.values.max()}"
                )
        else:
            print("Please select bathymetry first ...")

        self.get_datashader_dataframe()

    def delete_cells(
        self,
        delete_withcoastlines: bool = False,
        delete_withpolygon=None,
    ) -> None:
        """Delete mesh cells overlapping land or a given polygon.

        Parameters
        ----------
        delete_withcoastlines : bool, optional
            Remove cells on land using GSHHG coastlines (default ``False``).
        delete_withpolygon : gpd.GeoDataFrame, optional
            GeoDataFrame of polygons whose interior cells should be deleted.
        """
        if delete_withcoastlines:
            dfmt.meshkernel_delete_withcoastlines(mk=self.mk, res="h")
        if delete_withpolygon is not None:
            dfmt.meshkernel_delete_withgdf(
                mk=self.mk, coastlines_gdf=delete_withpolygon
            )
        dfmt.meshkernel_get_illegalcells(mk=self.mk)
        self.data = dfmt.meshkernel_to_UgridDataset(mk=self.mk, crs=self.model.crs)
        self.get_datashader_dataframe()

    def snap_to_grid(
        self, polyline: gpd.GeoDataFrame, max_snap_distance: float = 1.0
    ) -> gpd.GeoDataFrame:
        """Snap a polyline GeoDataFrame to the nearest grid edges.

        Parameters
        ----------
        polyline : gpd.GeoDataFrame
            Input LineString GeoDataFrame to snap.
        max_snap_distance : float, optional
            Maximum snapping distance (default ``1.0``).

        Returns
        -------
        gpd.GeoDataFrame
            Snapped polyline GeoDataFrame, or an empty GeoDataFrame if the
            input is empty.
        """
        if len(polyline) == 0:
            return gpd.GeoDataFrame()
        geom_list = []
        for iline, line in polyline.iterrows():
            geom = line["geometry"]
            if geom.geom_type == "LineString":
                geom_list.append(geom)
        gdf = gpd.GeoDataFrame({"geometry": geom_list})
        print("Snapping to grid ...")
        snapped_uds, snapped_gdf = xu.snap_to_grid(
            gdf, self.data.grid, max_snap_distance=max_snap_distance
        )
        print("Snapping to grid done.")
        snapped_gdf = snapped_gdf.set_crs(self.model.crs)
        return snapped_gdf

    def face_coordinates(self) -> tuple:
        """Return x and y coordinates of mesh face centres.

        Returns
        -------
        x : np.ndarray
            X-coordinates of face centres.
        y : np.ndarray
            Y-coordinates of face centres.
        """
        xy = self.data.grid.face_coordinates
        return xy[:, 0], xy[:, 1]

    def get_exterior(self) -> None:
        """Compute the exterior polygon(s) of the mesh and store in ``self.exterior``."""
        try:
            indx = self.data.grid.edge_node_connectivity[
                self.data.grid.exterior_edges, :
            ]
            x = self.data.grid.node_x[indx]
            y = self.data.grid.node_y[indx]
            linestrings = [
                shapely.LineString(np.column_stack((x[i], y[i]))) for i in range(len(x))
            ]
            merged = shapely.ops.linemerge(linestrings)
            polygons = shapely.ops.polygonize(merged)
            self.exterior = gpd.GeoDataFrame(
                geometry=list(polygons), crs=self.model.crs
            )
        except Exception:
            self.exterior = gpd.GeoDataFrame()

    def bounds(self, crs=None, buffer: float = 0.0) -> list:
        """Return the bounding box of the grid exterior.

        Parameters
        ----------
        crs : object, optional
            Target CRS for the returned bounds.  Defaults to the model CRS.
        buffer : float, optional
            Fractional buffer applied to each side (default ``0.0``).

        Returns
        -------
        list of float
            ``[lon_min, lat_min, lon_max, lat_max]`` in the requested CRS.
        """
        if crs is None:
            crs = self.crs
        lst = self.exterior.to_crs(crs=crs).total_bounds.tolist()
        dx = lst[2] - lst[0]
        dy = lst[3] - lst[1]
        lst[0] = lst[0] - buffer * dx
        lst[1] = lst[1] - buffer * dy
        lst[2] = lst[2] + buffer * dx
        lst[3] = lst[3] + buffer * dy
        return lst

    def get_datashader_dataframe(self) -> None:
        """Build the edge-segment DataFrame used for datashader rendering."""
        x1 = self.data.grid.edge_node_coordinates[:, 0, 0]
        x2 = self.data.grid.edge_node_coordinates[:, 1, 0]
        y1 = self.data.grid.edge_node_coordinates[:, 0, 1]
        y2 = self.data.grid.edge_node_coordinates[:, 1, 1]
        transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)
        x1, y1 = transformer.transform(x1, y1)
        x2, y2 = transformer.transform(x2, y2)
        self.df = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))

    def map_overlay(
        self,
        file_name: str,
        xlim=None,
        ylim=None,
        color: str = "black",
        width: int = 800,
    ) -> bool:
        """Render the grid as a PNG overlay image using datashader.

        Parameters
        ----------
        file_name : str
            Output PNG file path.
        xlim : list of float, optional
            ``[lon_min, lon_max]`` extent for the image.
        ylim : list of float, optional
            ``[lat_min, lat_max]`` extent for the image.
        color : str, optional
            Line colour (default ``"black"``).
        width : int, optional
            Output image width in pixels (default ``800``).

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure.
        """
        if self.data is None:
            return False
        try:
            if not hasattr(self, "df"):
                self.df = None
            if self.df is None:
                self.get_datashader_dataframe()

            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)
            cvs = ds.Canvas(
                x_range=xlim, y_range=ylim, plot_height=height, plot_width=width
            )
            agg = cvs.line(self.df, x=["x1", "x2"], y=["y1", "y2"], axis=1)
            img = tf.shade(agg)
            img_path = os.path.dirname(file_name)
            if not img_path:
                img_path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=img_path)
            return True
        except Exception:
            return False


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


def binary_search(val_array: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Find indices of *vals* inside a sorted array using binary search.

    Parameters
    ----------
    val_array : np.ndarray
        Sorted 1-D reference array.
    vals : np.ndarray
        Values to look up.

    Returns
    -------
    np.ndarray of int
        Array of indices into *val_array* where each value of *vals* was found,
        or ``-1`` if not found.
    """
    indx = np.searchsorted(val_array, vals)
    not_ok = np.where(indx == len(val_array))[0]
    indx[np.where(indx == len(val_array))[0]] = 0
    is_ok = np.where(val_array[indx] == vals)[0]
    indices = np.zeros(len(vals), dtype=int) - 1
    indices[is_ok] = indx[is_ok]
    indices[not_ok] = -1
    return indices


def gdf2list(gdf_in: gpd.GeoDataFrame) -> list:
    """Split a GeoDataFrame into a list of single-row GeoDataFrames.

    Parameters
    ----------
    gdf_in : gpd.GeoDataFrame
        Input GeoDataFrame with one or more rows.

    Returns
    -------
    list of gpd.GeoDataFrame
        One GeoDataFrame per feature.
    """
    gdf_out = []
    for feature in gdf_in.iterfeatures():
        gdf_out.append(gpd.GeoDataFrame.from_features([feature]))
    return gdf_out
