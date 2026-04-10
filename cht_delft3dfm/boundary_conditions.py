"""Delft3D-FM boundary conditions: generate, load, write, and set forcing time series."""

import os
from pathlib import Path
from typing import Optional

# from pandas.tseries.offsets import DateOffset
import dfm_tools as dfmt
import geopandas as gpd
import hydrolib.core.dflowfm as hcdfm
import numpy as np
import pandas as pd
from cht_utils.fileio.pli_file import gdf2pli, pli2gdf


class Delft3DFMBoundaryConditions:
    """Boundary condition manager for Delft3D-FM models.

    Parameters
    ----------
    model : Delft3DFM
        Parent model instance.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.forcing = "timeseries"
        self.gdf = gpd.GeoDataFrame()
        self.times = []
        self.bcafile = "waterlevel_astro.bc"
        self.gdf_points = gpd.GeoDataFrame()  # Contains the tide points
        self.bzsfile = "waterlevel_timeseries.bc"

    def generate_bnd(
        self,
        bnd_withcoastlines: bool = False,
        bnd_withpolygon=None,
        resolution: float = 0.06,
    ) -> None:
        """Generate the open boundary polyline from the mesh.

        Parameters
        ----------
        bnd_withcoastlines : bool, optional
            Remove land sections using GSHHG coastlines (default ``False``).
        bnd_withpolygon : gpd.GeoDataFrame, optional
            Polygon GeoDataFrame used to clip the boundary to a specific area.
        resolution : float, optional
            Interpolation resolution along the boundary in degrees (default
            ``0.06``).
        """
        from shapely import LineString, MultiLineString, MultiPolygon
        from shapely.ops import linemerge

        if not self.model.grid.mk:
            print('"First generate the grid')
            return

        if bnd_withcoastlines:
            bnd_gdf = dfmt.generate_bndpli_cutland(
                mk=self.model.grid.mk, res="h", buffer=0.01
            )
        if bnd_withpolygon is not None:
            mesh_bnds = self.model.grid.mk.mesh2d_get_mesh_boundaries_as_polygons()
            if mesh_bnds.geometry_separator in mesh_bnds.x_coordinates:
                raise Exception("use dfmt.generate_bndpli_cutland() on an uncut grid")
            mesh_bnds_xy = np.c_[mesh_bnds.x_coordinates, mesh_bnds.y_coordinates]
            pol_gdf = bnd_withpolygon

            meshbnd_ls = LineString(mesh_bnds_xy)
            pol_mp = MultiPolygon(pol_gdf.geometry.tolist())
            bnd_ls = meshbnd_ls.intersection(pol_mp)

            if isinstance(bnd_ls, MultiLineString):
                print(
                    "attemting to merge lines in MultiLineString to single LineString (if connected)"
                )
                bnd_ls = linemerge(bnd_ls)

            names = pol_gdf.get("name", [None])
            if isinstance(bnd_ls, MultiLineString):
                geoms = list(bnd_ls.geoms)
                names = (list(names) * len(geoms))[: len(geoms)]
                bnd_gdf = gpd.GeoDataFrame({"geometry": geoms, "name": names})
            else:
                bnd_gdf = gpd.GeoDataFrame({"geometry": [bnd_ls], "name": [names[0]]})
            bnd_gdf.crs = pol_gdf.crs

        self.gdf = dfmt.interpolate_bndpli(bnd_gdf, res=resolution)
        self.gdf_points = mline2point(self.gdf)

    def write_bnd(self, file_name: Optional[str] = None) -> None:
        """Write the boundary polyline to a PLI file.

        Parameters
        ----------
        file_name : str, optional
            Output file path.  Defaults to ``<model.path>/bnd.pli``.
        """
        if not file_name:
            file_name = os.path.join(self.model.path, "bnd.pli")
        gdf2pli(self.gdf, file_name, add_point_name=True)

    def load_bnd(self, file_name: Optional[str] = None) -> None:
        """Load the boundary polyline from a PLI file.

        Parameters
        ----------
        file_name : str, optional
            Input file path.  Defaults to ``<model.path>/bnd.pli``.
        """
        if not file_name:
            file_name = os.path.join(self.model.path, "bnd.pli")
        self.gdf = pli2gdf(file_name)
        self.gdf.crs = self.model.crs
        self.gdf_points = mline2point(self.gdf)

    def write_boundary_conditions_astro(self) -> None:
        """Write astronomical tidal boundary conditions to the BC file.

        Does nothing if no BCA file name is set or there are no boundary
        points.
        """
        if not self.bcafile:
            return

        if len(self.gdf_points.index) == 0:
            return

        filename = os.path.join(self.model.path, self.bcafile)

        with open(filename, "w") as fid:
            for ip, point in self.gdf_points.iterrows():
                astro = point["astro"]
                name = point["name"]
                fid.write("[forcing]\n")
                fid.write(f"Name                            = {name}\n")
                fid.write("Function                        = astronomic\n")
                fid.write("Quantity                        = astronomic component\n")
                fid.write("Unit                            = -\n")
                fid.write("Quantity                        = waterlevelbnd amplitude\n")
                fid.write("Unit                            = m\n")
                fid.write("Quantity                        = waterlevelbnd phase\n")
                fid.write("Unit                            = deg\n")
                for constituent, row in astro.iterrows():
                    fid.write(
                        f"{constituent:6s}{row['amplitude']:10.5f}{row['phase']:10.2f}\n"
                    )
                fid.write("\n")

    def write_boundary_conditions_timeseries(self) -> None:
        """Write time-series water-level boundary conditions to the BC file.

        Does nothing if no BZS file name is set or there are no boundary
        points.
        """
        if not self.bzsfile:
            return

        if len(self.gdf_points.index) == 0:
            return

        filename = os.path.join(self.model.path, self.bzsfile)

        tref = pd.to_datetime(str(self.model.input.time.refdate), format="%Y%m%d")
        tref_str = tref.strftime("%Y-%m-%d")

        with open(filename, "w") as fid:
            for ip, point in self.gdf_points.iterrows():
                timeseries = point["timeseries"]
                name = point["name"]

                if timeseries is None or timeseries.empty:
                    continue

                fid.write("[forcing]\n")
                fid.write(f"Name                            = {name}\n")
                fid.write("Function                        = timeseries\n")
                fid.write("Time-interpolation              = linear\n")
                fid.write("Quantity                        = time\n")
                fid.write(f"Unit                            = minutes since {tref_str}\n")
                fid.write("Quantity                        = waterlevelbnd\n")
                fid.write("Unit                            = m\n")

                for timestamp, row in timeseries.iterrows():
                    minutes_since_ref = (timestamp - tref).total_seconds() / 60.0
                    fid.write(f"{minutes_since_ref:10.2f}{row['wl']:10.2f}\n")

                fid.write("\n")

    def write_ext_wl(
        self,
        poly_file: Optional[str] = None,
        ext_file_new: Optional[str] = None,
        forcingtype: str = "bca",
    ) -> None:
        """Write a water-level boundary entry to the new-style external forcing file.

        Parameters
        ----------
        poly_file : str, optional
            Path to the PLI boundary file.  Defaults to
            ``<model.path>/bnd.pli``.
        ext_file_new : str, optional
            Path to the external forcing file.  Defaults to
            ``<model.path>/bnd_new.ext``.
        forcingtype : str, optional
            ``"bca"`` for astronomical forcing or ``"bzs"`` for time-series
            forcing (default ``"bca"``).
        """
        if not poly_file:
            poly_file = os.path.join(self.model.path, "bnd.pli")

        if not ext_file_new:
            ext_file_new = os.path.join(self.model.path, "bnd_new.ext")

        if forcingtype == "bca":
            forcingfile = self.bcafile
        elif forcingtype == "bzs":
            forcingfile = self.bzsfile

        if not self.model.input.external_forcing.extforcefilenew:
            self.model.input.external_forcing.extforcefilenew = hcdfm.ExtModel()
            self.model.input.external_forcing.extforcefilenew.filepath = Path(
                ext_file_new
            )

        boundary_object = hcdfm.Boundary(
            quantity="waterlevelbnd",
            locationfile=poly_file,
            forcingfile=forcingfile,
        )
        self.model.input.external_forcing.extforcefilenew.boundary.append(
            boundary_object
        )

        self.model.input.external_forcing.extforcefilenew.save(
            filepath=ext_file_new, path_style="windows"
        )

    def set_timeseries(
        self,
        shape: str = "constant",
        timestep: float = 600.0,
        offset: float = 0.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
        period: float = 43200.0,
        peak: float = 1.0,
        tpeak: float = 86400.0,
        duration: float = 43200.0,
    ) -> None:
        """Generate synthetic water-level time series for all boundary points.

        Parameters
        ----------
        shape : str, optional
            Waveform shape: ``"constant"``, ``"sine"``, or ``"gaussian"``
            (default ``"constant"``).
        timestep : float, optional
            Time step in seconds (default ``600.0``).
        offset : float, optional
            Constant water-level offset in metres (default ``0.0``).
        amplitude : float, optional
            Wave amplitude in metres for the sine shape (default ``1.0``).
        phase : float, optional
            Phase offset in degrees for the sine shape (default ``0.0``).
        period : float, optional
            Wave period in seconds for the sine shape (default ``43200.0``).
        peak : float, optional
            Peak height in metres for the Gaussian shape (default ``1.0``).
        tpeak : float, optional
            Time of peak in seconds since the start for the Gaussian shape
            (default ``86400.0``).
        duration : float, optional
            Characteristic duration in seconds for the Gaussian shape
            (default ``43200.0``).
        """
        tref = pd.to_datetime(str(self.model.input.time.refdate), format="%Y%m%d")

        if self.model.input.time.startdatetime:
            t_start = pd.to_datetime(
                str(self.model.input.time.startdatetime), format="%Y%m%d%H%M%S"
            )
            t_stop = pd.to_datetime(
                str(self.model.input.time.stopdatetime), format="%Y%m%d%H%M%S"
            )
        else:
            tunit = self.model.input.time.tunit.upper()
            unit_map = {"S": "seconds", "M": "minutes", "H": "hours", "D": "days"}
            tdelta_unit = unit_map.get(tunit, "seconds")
            t_start = tref + pd.to_timedelta(
                self.model.input.time.tstart, unit=tdelta_unit
            )
            t_stop = tref + pd.to_timedelta(
                self.model.input.time.tstop, unit=tdelta_unit
            )

        t0 = (t_start - tref).total_seconds()
        t1 = (t_stop - tref).total_seconds()
        dt = t1 - t0 if shape == "constant" else timestep
        time = np.arange(t0, t1 + dt, dt)

        nt = len(time)
        if shape == "constant":
            wl = [offset] * nt
        elif shape == "sine":
            wl = offset + amplitude * np.sin(
                2 * np.pi * time / period + phase * np.pi / 180
            )
        elif shape == "gaussian":
            wl = offset + peak * np.exp(-(((time - tpeak) / (0.25 * duration)) ** 2))
        elif shape == "astronomical":
            return

        times = pd.date_range(
            start=t_start, end=t_stop, freq=pd.tseries.offsets.DateOffset(seconds=dt)
        )

        if "timeseries" not in self.gdf_points.columns:
            self.gdf_points["timeseries"] = None
        self.gdf_points["timeseries"] = self.gdf_points["timeseries"].astype(object)
        df = pd.DataFrame({"wl": wl}, index=times)
        self.gdf_points["timeseries"] = [df.copy() for _ in range(len(self.gdf_points))]


def mline2point(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode (Multi)LineString geometries to individual Point features.

    Each vertex of every LineString is assigned a name derived from the parent
    row's ``"name"`` column and a zero-padded sequence number.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing LineString or MultiLineString geometries.

    Returns
    -------
    gpd.GeoDataFrame
        New GeoDataFrame where every row is a single Point.
    """
    from shapely.geometry import Point

    point_records = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        base_name = row["name"]

        if geom.geom_type == "LineString":
            coords = geom.coords
        elif geom.geom_type == "MultiLineString":
            coords = []
            for part in geom.geoms:
                coords.extend(part.coords)
        else:
            continue

        for i, coord in enumerate(coords, start=1):
            new_name = f"{base_name}_{str(i).zfill(4)}"
            point_records.append({"geometry": Point(coord), "name": new_name})

    gdf_points = gpd.GeoDataFrame(point_records, crs=gdf.crs)

    return gdf_points
