"""Main Delft3D-FM model class and supporting boundary/meteo data structures."""

import datetime
import math
import os
from typing import Optional, Union

import dfm_tools as dfmt
import geopandas as gpd
import hydrolib.core.dflowfm as hcdfm
import numpy as np
import pandas as pd
import shapely
from cht_utils.fileio.deltares_ini import IniStruct
from hydrolib.core.dflowfm.mdu.models import FMModel
from pyproj import CRS, Transformer

from .boundary_conditions import Delft3DFMBoundaryConditions
from .cross_sections import Delft3DFMCrossSections
from .grid import Delft3DFMGrid
from .observation_points import Delft3DFMObservationPoints
from .thin_dams import Delft3DFMThinDams
from .utils.geometry import Point


class Delft3DFM:
    """Top-level Delft3D-FM model object.

    Aggregates the grid, boundary conditions, observation points, cross
    sections, thin dams, and meteorological forcing for a single model
    instance.

    Parameters
    ----------
    input_file : str, optional
        Path to an MDU file.  When provided the model is loaded immediately.
    crs : object, optional
        Coordinate reference system (e.g. a ``pyproj.CRS`` instance).
    """

    def __init__(
        self,
        input_file: Optional[str] = None,
        crs=None,
    ) -> None:
        self.path = os.getcwd()
        self.crs = crs
        self.grid = Delft3DFMGrid(self)
        self.boundary_conditions = Delft3DFMBoundaryConditions(self)
        self.observation_points = Delft3DFMObservationPoints(self)
        self.cross_sections = Delft3DFMCrossSections(self)
        self.thin_dams = Delft3DFMThinDams(self)
        self.boundary = []
        self.obstacle = []
        self.meteo = Delft3DFMMeteo()

        self.input = FMModel()

        if input_file:
            self.path = os.path.dirname(input_file)
            self.load(input_file)

    def load(self, inputfile: str) -> None:
        """Read the MDU file and all referenced attribute files.

        Parameters
        ----------
        inputfile : str
            Path to the MDU file.
        """
        self.read_input_file(inputfile)
        self.read_attribute_files()

    def read_input_file(self, input_file: str) -> None:
        """Parse the MDU file and populate ``self.input``.

        Parameters
        ----------
        input_file : str
            Path to the MDU file.
        """
        self.path = os.path.dirname(input_file)
        self.input = FMModel(input_file)

    def write_input_file(self, input_file: Optional[str] = None) -> None:
        """Save the MDU file.

        Parameters
        ----------
        input_file : str, optional
            Output path.  Defaults to ``<model.path>/test.mdu``.
        """
        if not input_file:
            input_file = os.path.join(self.path, "test.mdu")

        self.input.save(input_file, path_style="windows")
        try:
            dfmt.make_paths_relative(input_file)
        except Exception:
            pass

    def read_attribute_files(self) -> None:
        """Read all attribute files referenced by the current model input."""
        # Grid
        self.grid = Delft3DFMGrid(self)
        if self.input.geometry.netfile.filepath:
            self.grid.read(self.input.geometry.netfile.filepath)

        # External forcing (boundary conditions)
        if self.input.external_forcing.extforcefilenew:
            self.read_ext_file_new()

        # Observation points
        self.observation_points.read()

        pass

    def read_ext_file_new(self) -> None:
        """Parse the new-style external forcing file and load boundary data."""
        ext_file = os.path.join(
            self.path, self.input.external_forcing.extforcefilenew.filepath
        )

        d = IniStruct(filename=ext_file)

        for section in d.section:
            if section.name.lower() == "boundary":
                bnd = Delft3DFMBoundary(
                    quantity=section.get_value("quantity"),
                    locationfile=section.get_value("locationfile"),
                    forcingfile=section.get_value("forcingfile"),
                )
                bnd.read_location_file(path=self.path)
                bnd.read_forcing_file(path=self.path)
                self.boundary.append(bnd)

    def write_flow_boundary_conditions(
        self, path: Optional[str] = None, file_name: Optional[str] = None
    ) -> None:
        """Write flow boundary conditions to a Delft3D-FM BC file.

        Parameters
        ----------
        path : str, optional
            Output directory.  Defaults to ``self.path``.
        file_name : str, optional
            Output file name.  Defaults to the forcing file of the first
            boundary object.
        """
        from deltares_ini import Keyword, Section

        if not path:
            path = self.path

        frc_file = self.boundary[0].forcingfile
        file_name = os.path.join(path, frc_file)

        d = IniStruct()

        refdate = datetime.datetime.strptime(str(self.input.time.refdate), "%Y%m%d")
        trefstr = refdate.strftime("%Y-%m-%d %H%M%S")
        tunitstr = f"seconds since {trefstr}"
        vunitstr = "m"

        for ind, bnd in enumerate(self.boundary):
            bnd.forcingfile = frc_file

            for ip, point in enumerate(bnd.point):
                s = Section()
                s.name = "forcing"
                s.keyword.append(Keyword(name="Name", value=point.name))
                s.keyword.append(Keyword(name="Function", value="timeseries"))
                s.keyword.append(Keyword(name="Time-interpolation", value="linear"))
                s.keyword.append(Keyword(name="Quantity", value="time"))
                s.keyword.append(Keyword(name="Unit", value=tunitstr))
                s.keyword.append(Keyword(name="Quantity", value=bnd.quantity))
                s.keyword.append(Keyword(name="Unit", value=vunitstr))

                data = point.data.copy()
                tmsec = pd.to_timedelta(point.data.index - refdate, unit="s")
                data.index = tmsec.total_seconds()
                s.data = data

                d.section.append(s)

        d.write(file_name)

    def write_ext_meteo(self, file_name: Optional[str] = None) -> None:
        """Write meteorological forcing to an old-style EXT file.

        Used in CoSMoS.

        Parameters
        ----------
        file_name : str, optional
            Output file path.  Defaults to the path stored in the model input.
        """
        if not file_name:
            if not self.input.external_forcing.extforcefile:
                return
            file_name = os.path.join(
                self.path, self.input.external_forcing.extforcefile.filepath
            )

        if not file_name:
            return

        ext_old = hcdfm.ExtOldModel(file_name)
        if self.meteo.amu_file:
            forcing = hcdfm.ExtOldForcing(
                quantity="windx",
                filename=self.meteo.amu_file,
                filetype=hcdfm.ExtOldFileType.ArcInfo,
                method=hcdfm.ExtOldMethod.InterpolateTimeAndSpace,
                operand=hcdfm.Operand.override,
            )
            ext_old.forcing.append(forcing)

        if self.meteo.amv_file:
            forcing = hcdfm.ExtOldForcing(
                quantity="windy",
                filename=self.meteo.amv_file,
                filetype=hcdfm.ExtOldFileType.ArcInfo,
                method=hcdfm.ExtOldMethod.InterpolateTimeAndSpace,
                operand=hcdfm.Operand.override,
            )
            ext_old.forcing.append(forcing)

        if self.meteo.amp_file:
            forcing = hcdfm.ExtOldForcing(
                quantity="atmosphericpressure",
                filename=self.meteo.amp_file,
                filetype=hcdfm.ExtOldFileType.ArcInfo,
                method=hcdfm.ExtOldMethod.InterpolateTimeAndSpace,
                operand=hcdfm.Operand.override,
            )
            ext_old.forcing.append(forcing)

        if self.meteo.spw_file:
            forcing = hcdfm.ExtOldForcing(
                quantity="airpressure_windx_windy",
                filename=self.meteo.spw_file,
                filetype=hcdfm.ExtOldFileType.SpiderWebData,
                method=hcdfm.ExtOldMethod.PassThrough,
                operand=hcdfm.Operand.override,
            )
            ext_old.forcing.append(forcing)

        ext_old.save(filepath=file_name, path_style="windows")

    def delete_observation_point(self, name_or_index: Union[str, int]) -> None:
        """Remove an observation point by name or index (legacy method).

        .. deprecated::
            Use :meth:`observation_points.delete_point` instead.

        Parameters
        ----------
        name_or_index : str or int
            Name string or integer row index of the point to remove.
        """
        if isinstance(name_or_index, str):
            name = name_or_index
            for index, row in self.observation_point_gdf.iterrows():
                if row["name"] == name:
                    self.observation_point_gdf = self.observation_point_gdf.drop(
                        index
                    ).reset_index(drop=True)
                    return
            print(f"Point {name} not found!")
        else:
            index = name_or_index
            if len(self.observation_point_gdf.index) < index + 1:
                print("Index exceeds length!")
            self.observation_point_gdf = self.observation_point_gdf.drop(
                index
            ).reset_index(drop=True)
            return

    def add_observation_point_gdf(self, x: float, y: float, name: str) -> None:
        """Add an observation point to the legacy GeoDataFrame (CoSMoS).

        .. deprecated::
            Use :meth:`observation_points.add_point` instead.

        Parameters
        ----------
        x : float
            X-coordinate of the point.
        y : float
            Y-coordinate of the point.
        name : str
            Point identifier.
        """
        point = shapely.geometry.Point(x, y)
        gdf_list = []
        d = {"name": name, "long_name": None, "geometry": point}
        gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.crs)
        self.observation_point_gdf = pd.concat(
            [self.observation_point_gdf, gdf_new], ignore_index=True
        )

    def read_observation_points(
        self, path: Optional[str] = None, file_list: Optional[list] = None
    ) -> None:
        """Read observation points into the legacy GeoDataFrame (CoSMoS).

        .. deprecated::
            Use :meth:`observation_points.read` instead.

        Parameters
        ----------
        path : str, optional
            Directory containing the observation-point files.
        file_list : list of str, optional
            Explicit list of file paths to read.
        """
        if not path:
            path = self.path

        if not file_list:
            if not self.input.output.obsfile:
                return
            file_list = []
            for i, v in enumerate(self.input.output.obsfile):
                file_list.append(os.path.join(path, v.filepath))

        gdf_list = []

        for file_name in file_list:
            if not os.path.exists(file_name):
                print(f"Warning : file {file_name} does not exist !")
                return
            df = pd.read_csv(
                file_name,
                index_col=False,
                header=None,
                delim_whitespace=True,
                names=["x", "y", "name"],
            )

            for ind in range(len(df.x.values)):
                name = str(df.name.values[ind])
                x = df.x.values[ind]
                y = df.y.values[ind]

                point2 = shapely.geometry.Point(x, y)
                d = {"name": name, "long_name": None, "geometry": point2}
                gdf_list.append(d)
        self.observation_point_gdf = gpd.GeoDataFrame(gdf_list, crs=self.crs)

    def write_observation_points(
        self, file_name: Optional[str] = None, path: Optional[str] = None
    ) -> None:
        """Write observation points from the legacy GeoDataFrame (CoSMoS).

        .. deprecated::
            Use :meth:`observation_points.write` instead.

        Parameters
        ----------
        file_name : str, optional
            Output file name.  Defaults to the first observation file in the
            model input.
        path : str, optional
            Output directory.  Defaults to ``self.path``.
        """
        if not path:
            path = self.path

        if not file_name:
            file_name = self.input.output.obsfile[0].filepath

        file_name = os.path.join(path, file_name)

        with open(file_name, "w") as fid:
            for index, row in self.observation_point_gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.1f}{y:12.1f}  "{name}"\n'
                fid.write(string)

    def list_observation_names(self) -> list:
        """Return observation point names from the legacy GeoDataFrame.

        .. deprecated::
            Use :meth:`observation_points.list_names` instead.

        Returns
        -------
        list of str
            Names of all observation points.
        """
        names = []
        for index, row in self.observation_point_gdf.iterrows():
            names.append(row["name"])
        return names

    def read_observation_lines(
        self, path: Optional[str] = None, file_name: Optional[str] = None
    ) -> None:
        """Read observation cross sections into the legacy GeoDataFrame (CoSMoS).

        .. deprecated::
            Use :meth:`cross_sections.read` instead.

        Parameters
        ----------
        path : str, optional
            Directory containing the cross-section file.
        file_name : str, optional
            Cross-section file name.
        """
        if not path:
            path = self.path

        if not file_name:
            file_name = self.input.output.crsfile[0].filepath

        data = hcdfm.PolyFile(os.path.join(path, file_name))
        self.observation_line_gdf = dfmt.PolyFile_to_geodataframe_linestrings(
            data, crs=None
        )

    def write_observation_lines(
        self, path: Optional[str] = None, file_name: Optional[str] = None
    ) -> None:
        """Write observation cross sections from the legacy GeoDataFrame (CoSMoS).

        .. deprecated::
            Use :meth:`cross_sections.write` instead.

        Parameters
        ----------
        path : str, optional
            Output directory.  Defaults to ``self.path``.
        file_name : str, optional
            Output file name.
        """
        if not path:
            path = self.path

        if not file_name:
            file_name = self.input.output.crsfile[0].filepath

        file_name = os.path.join(path, file_name)

        pli_polyfile = dfmt.geodataframe_to_PolyFile(self.observation_line_gdf)
        pli_polyfile.save(file_name)

    def read_timeseries_output(
        self,
        name_list: Optional[list] = None,
        path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_name_wave=None,
    ):
        """Read model output time series from a NetCDF history file.

        Parameters
        ----------
        name_list : list of str, optional
            Station names to extract.  Defaults to all stations.
        path : str, optional
            Directory containing the output file.
        file_name : str, optional
            Name of the history NetCDF file (default ``"flow_his.nc"``).
        file_name_wave : list of str, optional
            Wave output NetCDF file names.  When provided a dict of DataFrames
            with ``"hs"`` and ``"tp"`` columns is returned.

        Returns
        -------
        pd.DataFrame or dict of pd.DataFrame
            Water-level (or wave parameter) time series for each requested
            station.
        """
        import numpy as np
        import pandas as pd
        import xarray as xr

        if not path:
            path = self.path

        if not file_name:
            file_name = "flow_his.nc"

        file_name = os.path.join(path, file_name)

        ddd = xr.open_dataset(file_name)
        stations = ddd.waterlevel.coords["station_name"].values
        all_stations = []
        for ist, st in enumerate(stations):
            all_stations.append(st.decode().strip())

        if not name_list:
            name_list = []
            for st in all_stations:
                name_list.append(st)

        if not file_name_wave:
            times = ddd.waterlevel.coords["time"].values
            df = pd.DataFrame(index=times, columns=name_list)

            for station in name_list:
                for ist, st in enumerate(all_stations):
                    if station == st:
                        wl = ddd.waterlevel.values[:, ist]
                        wl[np.isnan(wl)] = -999.0
                        df[st] = wl
                        break

            ddd.close()
        else:
            ddd.close()
            ddd = {}
            for i, v in enumerate(file_name_wave):
                fn = os.path.join(path, v)
                ddd[i] = xr.open_dataset(fn)

            times = ddd[0].Hsig.coords["time"].values

            df = {}
            for station in name_list:
                for ist, st in enumerate(all_stations):
                    if station == st:
                        df[st] = pd.DataFrame(index=times, columns=["hs", "tp"])
                        for i, v in enumerate(file_name_wave):
                            Hs = ddd[i].Hsig.values[:, ist]
                            Tp = ddd[i].RTpeak.values[:, ist]
                            if max(Hs) > -999:
                                break
                        Tp[np.isnan(Hs)] = -999.0
                        Hs[np.isnan(Hs)] = -999.0
                        df[st]["hs"] = Hs
                        df[st]["tp"] = Tp
                        break
            for i, v in enumerate(file_name_wave):
                ddd[i].close()

        return df

    def grid_coordinates(self, loc: str = "cor") -> tuple:
        """Compute rotated grid coordinates.

        Parameters
        ----------
        loc : str, optional
            ``"cor"`` for cell corners (default) or ``"cen"`` for centres.

        Returns
        -------
        xg : np.ndarray
            2-D x-coordinate array.
        yg : np.ndarray
            2-D y-coordinate array.
        """
        cosrot = math.cos(self.input.rotation * math.pi / 180)
        sinrot = math.sin(self.input.rotation * math.pi / 180)
        if loc == "cor":
            xx = np.linspace(
                0.0, self.input.mmax * self.input.dx, num=self.input.mmax + 1
            )
            yy = np.linspace(
                0.0, self.input.nmax * self.input.dy, num=self.input.nmax + 1
            )
        else:
            xx = np.linspace(
                0.5 * self.input.dx,
                self.input.mmax * self.input.dx - 0.5 * self.input.dx,
                num=self.input.mmax,
            )
            yy = np.linspace(
                0.5 * self.input.dy,
                self.input.nmax * self.input.dy - 0.5 * self.input.dy,
                num=self.input.nmax,
            )

        xg0, yg0 = np.meshgrid(xx, yy)
        xg = self.input.x0 + xg0 * cosrot - yg0 * sinrot
        yg = self.input.y0 + xg0 * sinrot + yg0 * cosrot

        return xg, yg

    def bounding_box(self, crs=None) -> tuple:
        """Return the axis-aligned bounding box of the grid.

        Parameters
        ----------
        crs : object, optional
            Target CRS.  Defaults to the model CRS.

        Returns
        -------
        x_range : list of float
            ``[x_min, x_max]``.
        y_range : list of float
            ``[y_min, y_max]``.
        """
        xg, yg = self.grid_coordinates(loc="cor")

        if crs:
            transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
            xg, yg = transformer.transform(xg, yg)

        x_range = [np.min(np.min(xg)), np.max(np.max(xg))]
        y_range = [np.min(np.min(yg)), np.max(np.max(yg))]

        return x_range, y_range

    def outline(self, crs=None) -> tuple:
        """Return the four-corner outline of the grid.

        Parameters
        ----------
        crs : object, optional
            Target CRS.  Defaults to the model CRS.

        Returns
        -------
        xp : list of float
            X-coordinates of the five outline vertices (closed).
        yp : list of float
            Y-coordinates of the five outline vertices (closed).
        """
        xg, yg = self.grid_coordinates(loc="cor")

        if crs:
            transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
            xg, yg = transformer.transform(xg, yg)

        xp = [xg[0, 0], xg[0, -1], xg[-1, -1], xg[-1, 0], xg[0, 0]]
        yp = [yg[0, 0], yg[0, -1], yg[-1, -1], yg[-1, 0], yg[0, 0]]

        return xp, yp

    def clear_spatial_attributes(self) -> None:
        """Reset grid and boundary-condition objects to empty defaults."""
        self.grid = Delft3DFMGrid(self)
        self.boundary_conditions = Delft3DFMBoundaryConditions(self)

    def make_index_tiles(
        self, path: str, zoom_range: Optional[list] = None
    ) -> None:
        """Generate index tile files mapping tile pixels to model grid cells.

        Parameters
        ----------
        path : str
            Root directory where tile files are written.
        zoom_range : list of int, optional
            ``[min_zoom, max_zoom]`` range (default ``[0, 13]``).
        """
        import fileops as fo
        from tiling import deg2num, num2deg

        if not zoom_range:
            zoom_range = [0, 13]

        npix = 256

        lon_range, lat_range = self.bounding_box(crs=CRS.from_epsg(4326))

        cosrot = math.cos(-self.input.rotation * math.pi / 180)
        sinrot = math.sin(-self.input.rotation * math.pi / 180)

        transformer_a = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True
        )
        transformer_b = Transformer.from_crs(
            CRS.from_epsg(3857), self.crs, always_xy=True
        )

        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            print(f"Processing zoom level {izoom}")

            zoom_path = os.path.join(path, str(izoom))

            dxy = (40075016.686 / npix) / 2**izoom
            xx = np.linspace(0.0, (npix - 1) * dxy, num=npix)
            yy = xx[:]
            xv, yv = np.meshgrid(xx, yy)

            ix0, iy0 = deg2num(lat_range[0], lon_range[0], izoom)
            ix1, iy1 = deg2num(lat_range[1], lon_range[1], izoom)

            for i in range(ix0, ix1 + 1):
                path_okay = False
                zoom_path_i = os.path.join(zoom_path, str(i))

                for j in range(iy0, iy1 + 1):
                    file_name = os.path.join(zoom_path_i, f"{j}.dat")

                    lat, lon = num2deg(i, j, izoom)

                    xo, yo = transformer_a.transform(lon, lat)

                    x = xv[:] + xo + 0.5 * dxy
                    y = yv[:] + yo + 0.5 * dxy

                    x, y = transformer_b.transform(x, y)

                    x00 = x - self.input.x0
                    y00 = y - self.input.y0
                    xg = x00 * cosrot - y00 * sinrot
                    yg = x00 * sinrot + y00 * cosrot

                    iind = np.floor(xg / self.input.dx).astype(int)
                    jind = np.floor(yg / self.input.dy).astype(int)
                    ind = iind * self.input.nmax + jind
                    ind[iind < 0] = -999
                    ind[jind < 0] = -999
                    ind[iind > 255] = -999
                    ind[jind > 255] = -999

                    if np.any(ind >= 0):
                        if not path_okay:
                            if not os.path.exists(zoom_path_i):
                                fo.mkdir(zoom_path_i)
                                path_okay = True

                        with open(file_name, "wb") as fid:
                            fid.write(ind)


class Delft3DFMBoundary:
    """A single open boundary condition with location and forcing data.

    Parameters
    ----------
    quantity : str, optional
        Physical quantity (e.g. ``"waterlevelbnd"``).
    locationfile : str, optional
        Path to the PLI location file.
    forcingfile : str, optional
        Path to the BC forcing file.
    """

    def __init__(
        self,
        quantity: Optional[str] = None,
        locationfile: Optional[str] = None,
        forcingfile: Optional[str] = None,
    ) -> None:
        self.quantity = quantity
        self.locationfile = locationfile
        self.forcingfile = forcingfile
        self.geometry = []
        self.point = []

    def read_location_file(self, path: str = "") -> None:
        """Read boundary point locations from a PLI file.

        Parameters
        ----------
        path : str, optional
            Directory prefix for the location file.
        """
        loc_file = os.path.join(path, self.locationfile)

        from cht_utils.pli_file import read_pli_file

        d = read_pli_file(loc_file)
        name0 = os.path.split(loc_file)[-1][0:-4]
        for polyline in d:
            for ip, x in enumerate(polyline.x):
                name = f"{name0}_{str(ip + 1).zfill(4)}"
                point = Delft3DFMBoundaryPoint(
                    x=polyline.x[ip], y=polyline.y[ip], name=name
                )
                self.point.append(point)

    def read_forcing_file(self, path: str = "") -> None:
        """Read time-series forcing data for each boundary point.

        Parameters
        ----------
        path : str, optional
            Directory prefix for the forcing file.
        """
        frc_file = os.path.join(path, self.forcingfile)

        if os.path.exists(frc_file):
            d = IniStruct(filename=frc_file)
            for ind, point in enumerate(self.point):
                point.data = d.section[ind].data

    def plot(self, ax) -> None:
        """Placeholder for boundary plotting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot onto.
        """
        pass


class Delft3DFMBoundaryPoint:
    """A single point on an open boundary.

    Parameters
    ----------
    name : str, optional
        Point identifier.
    x : float, optional
        X-coordinate.
    y : float, optional
        Y-coordinate.
    crs : object, optional
        Coordinate reference system.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        crs=None,
    ) -> None:
        self.name = name
        self.geometry = Point(x, y, crs=crs)
        self.data = None


class SfincsFlowBoundaryConditions:
    """Stub class for SFINCS flow boundary conditions (unused placeholder)."""

    def __init__(self) -> None:
        self.geometry = []

    def read(self, bndfile: str, bzsfile: str) -> None:
        """Read boundary points and time series.

        Parameters
        ----------
        bndfile : str
            Path to the BND file.
        bzsfile : str
            Path to the BZS file.
        """
        self.read_points(bndfile)
        self.read_time_series(bzsfile)

    def read_points(self, file_name: str) -> None:
        """Read boundary point locations (stub).

        Parameters
        ----------
        file_name : str
            Path to the boundary point file.
        """
        pass

    def read_time_series(self, file_name: str) -> None:
        """Read boundary time-series data (stub).

        Parameters
        ----------
        file_name : str
            Path to the time-series file.
        """
        pass

    def set_xy(self, x, y) -> None:
        """Set the boundary geometry coordinates.

        Parameters
        ----------
        x : array-like
            X-coordinates.
        y : array-like
            Y-coordinates.
        """
        self.geometry.x = x
        self.geometry.y = y
        pass

    def plot(self, ax) -> None:
        """Placeholder for plotting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot onto.
        """
        pass


class SfincsWaveBoundaryConditions:
    """Stub class for SFINCS wave boundary conditions (unused placeholder)."""

    def __init__(self) -> None:
        self.geometry = []

    def read(self, bndfile: str, bzsfile: str) -> None:
        """Read boundary points and time series.

        Parameters
        ----------
        bndfile : str
            Path to the BND file.
        bzsfile : str
            Path to the BZS file.
        """
        self.read_points(bndfile)
        self.read_time_series(bzsfile)

    def read_points(self, file_name: str) -> None:
        """Read boundary point locations (stub).

        Parameters
        ----------
        file_name : str
            Path to the boundary point file.
        """
        pass

    def read_time_series(self, file_name: str) -> None:
        """Read boundary time-series data (stub).

        Parameters
        ----------
        file_name : str
            Path to the time-series file.
        """
        pass

    def set_xy(self, x, y) -> None:
        """Set the boundary geometry coordinates.

        Parameters
        ----------
        x : array-like
            X-coordinates.
        y : array-like
            Y-coordinates.
        """
        self.geometry.x = x
        self.geometry.y = y
        pass

    def plot(self, ax) -> None:
        """Placeholder for plotting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot onto.
        """
        pass


class FlowBoundaryPoint:
    """A single flow boundary point with optional time-series and tidal data.

    Parameters
    ----------
    x : float
        X-coordinate.
    y : float
        Y-coordinate.
    name : str, optional
        Point identifier.
    crs : object, optional
        Coordinate reference system.
    data : object, optional
        Time-series data.
    astro : object, optional
        Tidal constituent data.
    """

    def __init__(
        self,
        x: float,
        y: float,
        name: Optional[str] = None,
        crs=None,
        data=None,
        astro=None,
    ) -> None:
        self.name = name
        self.geometry = Point(x, y, crs=crs)
        self.data = data
        self.astro = astro


class Delft3DFMMeteo:
    """Container for meteorological forcing file references.

    Attributes
    ----------
    amu_file : str or None
        Path to the wind-u (AMU) file.
    amv_file : str or None
        Path to the wind-v (AMV) file.
    amp_file : str or None
        Path to the pressure (AMP) file.
    ampr_file : str or None
        Path to the precipitation (AMPR) file.
    spw_file : str or None
        Path to the spiderweb (SPW) file.
    """

    def __init__(self) -> None:
        self.amu_file = None
        self.amv_file = None
        self.amp_file = None
        self.ampr_file = None
        self.spw_file = None


def read_timeseries_file(file_name: str, ref_date) -> pd.DataFrame:
    """Read a whitespace-delimited time-series file with a seconds-offset index.

    Parameters
    ----------
    file_name : str
        Path to the time-series file.
    ref_date : datetime-like
        Reference date used to convert the seconds offset to absolute times.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by absolute timestamps with one column per data
        channel.
    """
    df = pd.read_csv(file_name, index_col=0, header=None, delim_whitespace=True)
    ts = ref_date + pd.to_timedelta(df.index, unit="s")
    df.index = ts

    return df
