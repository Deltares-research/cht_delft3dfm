# -*- coding: utf-8 -*-
"""
Created on Sat May 15 08:08:40 2021

@author: ormondt
"""
import os
import pandas as pd
import datetime
import numpy as np
from pandas.tseries.offsets import DateOffset

import math
from pyproj import CRS
from pyproj import Transformer


from pathlib import Path
from hydrolib.core.dflowfm.mdu.models import FMModel
import hydrolib.core.dflowfm as hcdfm

from cht_utils.geometry import RegularGrid
from cht_utils.geometry import Point
from cht_utils.deltares_ini import IniStruct

class Delft3DFM:
    
    def __init__(self, input_file=None, crs=None):
 
#        self.epsg                     = epsg
        # self.mdu                      = MDU()
        # self.input                    = Delft3DFMInput(self.mdu)
        self.crs                      = crs
        self.grid                     = None
        self.mask                     = None
        self.boundary                 = []
        self.observation_point        = []
        self.obstacle                 = []
        self.meteo                    = Delft3DFMMeteo()
        
        if input_file:
            self.path = os.path.dirname(input_file)
            self.load(input_file)

    def load(self, inputfile):
        # Reads sfincs.inp and attribute files
        self.read_input_file(inputfile)
        self.read_attribute_files()
    
    def read_input_file(self, input_file):
        
        # Reads mdu file
        
        # Get the path of sfincs.inp
        self.path = os.path.dirname(input_file)
        self.input = FMModel(input_file)
        
        # # Read file (ini format)
        # d=IniStruct(filename=input_file)
        
        # # Defaults have already been set, so only loop over entries in mdu file
        # for section in d.section:
        #     for keyword in section.keyword:
        #         # Look up keyword in mdu table
        #         for mdu in self.mdu.dict:
        #             if mdu["keyword"].lower() == keyword.name.lower():
        #                 val = keyword.value
        #                 if val:
        #                     if mdu["type"] == "float":
        #                         val = float(val)
        #                     elif mdu["type"] == "int":
        #                         val = int(val)
        #                     elif mdu["type"] == "date":
        #                         val = datetime.datetime.strptime(val.rstrip(), '%Y%m%d')
        #                 else:
        #                     val = mdu["default"]
                            
        #                 setattr(self.input, mdu["name"], val)                       
        #                 break
        
        # Adjust start and stop times to Python datetime
        # self.input.tstart = datetime.datetime.strptime(self.input.time.startdatetime, '%Y%m%d%H%M%S')
        # self.input.tstop  = datetime.datetime.strptime(self.input.time.stopdatetime,  '%Y%m%d%H%M%S')

    def write_input_file(self, input_file=None):

        if not input_file:
            input_file = os.path.join(self.path, "test.mdu")
        
        # # Store input data in ini structure        
        # d = IniStruct()
        # # Loop through all the keywords in the mdu table
        # for mdu in self.mdu.dict:
        #     val = getattr(self.input, mdu["name"])
        #     d.set_value(mdu["group"], mdu["keyword"], val, mdu["comment"])
        
        # Adjust start and stop time to seconds    
        # t0 = (self.input.tstart - self.input.refdate).total_seconds()
        # t1 = (self.input.tstop  - self.input.refdate).total_seconds()
        # d.set_value("time", "TStart", t0, "Start time w.r.t. RefDate (in TUnit)")
        # d.set_value("time", "TStop",  t1, "Stop time w.r.t. RefDate (in TUnit)")
                
        # d.write(input_file) 

        self.input.save(input_file, path_style='windows')      
                       
    def read_attribute_files(self):

        # Grid
#        self.grid = SfincsGrid()

        # External forcing (boundary conditions)
        if self.input.external_forcing.extforcefilenew:
            self.read_ext_file_new()
        
#        # External forcing (meteo)
#        if self.input.extforcefile:
#            self.read_ext_file()


        # Flow boundary conditions
#        self.read_flow_boundary_points()
#        self.read_flow_boundary_conditions()

        # Observation points
        self.read_observation_points()

#        self.grid.compute_coordinates(x0,y0,dx,dy,nx,ny,rotation)
        pass
    

##### Boundary points #####

    def read_ext_file_new(self):

        ext_file = os.path.join(self.path,
                                self.input.external_forcing.extforcefilenew.filepath)

        d=IniStruct(filename=ext_file)
        
        for section in d.section:
            if section.name.lower()=="boundary":
                # New boundary found
                bnd = Delft3DFMBoundary(quantity=section.get_value("quantity"),
                                        locationfile=section.get_value("locationfile"),
                                        forcingfile=section.get_value("forcingfile"))                
                bnd.read_location_file(path=self.path)
                bnd.read_forcing_file(path=self.path)                
                self.boundary.append(bnd)


    # def read_flow_boundary_points(self):
        
    #     # Read SFINCS bnd file
        
    #     self.flow_boundary_point = []
        
    #     if not self.input.bndfile:
    #         return
                    
    #     bnd_file = os.path.join(self.path,
    #                             self.input.bndfile)

    #     if not os.path.exists(bnd_file):
    #         return
        
    #     # Read the bnd file
    #     df = pd.read_csv(bnd_file, index_col=False, header=None,
    #          delim_whitespace=True, names=['x', 'y'])
        
    #     # Loop through points
    #     for ind in range(len(df.x.values)):
    #         name = str(ind + 1).zfill(4)
    #         point = FlowBoundaryPoint(df.x.values[ind],
    #                                   df.y.values[ind],
    #                                   name=name)
    #         self.flow_boundary_point.append(point)

    # def write_flow_boundary_points(self, file_name=None):

    #     # Write SFINCS bnd file
    #     if not file_name:
    #         if not self.input.bndfile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                  self.input.bndfile)
            
    #     if not file_name:
    #         return
            
    #     fid = open(file_name, "w")
    #     for point in self.flow_boundary_point:
    #         string = f'{point.geometry.x:12.1f}{point.geometry.y:12.1f}"\n'
    #         fid.write(string)
    #     fid.close()    

    # ### Flow Boundary points ###
    
    # def read_flow_boundary_conditions(self, file_name=None):

    #     # Read SFINCS bzs file
        
    #     if not file_name:
    #         if not self.input.bzsfile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                  self.input.bzsfile)
            
    #     if not file_name:
    #         return
        
    #     if not os.path.exists(file_name):
    #         return
        
    #     if not self.input.tref:
    #         # tref has not yet been defined
    #         return

    #     df = read_timeseries_file(file_name, self.input.tref)

    #     ts  = df.index
    #     for icol, point in enumerate(self.flow_boundary_point):
    #         point.data = pd.Series(df.iloc[:,icol].values, index=ts)
        
    def write_flow_boundary_conditions(self, path=None, file_name=None):
        
        # Write Delft3D-FM bc file (there can be only ONE bc file !!!)

        from deltares_ini import Section
        from deltares_ini import Keyword

        if not path:
            path = self.path
        
        frc_file = self.boundary[0].forcingfile
        file_name = os.path.join(path, frc_file)
            
        d=IniStruct()
        
        refdate = datetime.datetime.strptime(str(self.input.time.refdate), '%Y%m%d')
        trefstr  = refdate.strftime('%Y-%m-%d %H%M%S')
        tunitstr = "seconds since " + trefstr
        vunitstr = "m"
    
        for ind, bnd in enumerate(self.boundary):        

            bnd.forcingfile = frc_file

            for ip, point in enumerate(bnd.point):
                
                s=Section()
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
#            point_names.append(point.name)                        
#        # Return DataFrame bzs
#        bzs = overall.read_timeseries_output(name_list=point_names,
#                                             path=output_path,
#                                             file_name=output_file)
        # ts  = bzs.index
        # for ip, point in enumerate(bnd.point):
        #     point.data = pd.Series(bzs.iloc[:,ip].values, index=ts) + boundary_water_level_correction    

        
        # # Build a new DataFrame
        # df = pd.DataFrame()
        # for point in self.flow_boundary_point:
        #     df = pd.concat([df, point.data], axis=1)
        # tmsec = pd.to_timedelta(df.index - self.input.tref, unit="s")
        # df.index = tmsec.total_seconds()
        # df.to_csv(file_name,
        #           index=True,
        #           sep=" ",
        #           header=False,
        #           float_format="%0.3f")

    # def read_astro_boundary_conditions(self, file_name=None):

    #     # Read SFINCS bca file
    #     if not file_name:
    #         if not self.input.bcafile:
    #             return
    #         file_name = os.path.join(self.path,
    #                                  self.input.bcafile)
            
    #     if not file_name:
    #         return
        
    #     if not os.path.exists(file_name):
    #         return

    #     d = IniStruct(filename=file_name)
    #     for ind, point in enumerate(self.flow_boundary_point):
    #         point.astro = d.section[ind].data

    def write_ext_meteo(self, file_name=None):

        # Write Delft3D-FM ext file (meteo)
        if not file_name:
            if not self.input.external_forcing.extforcefile:
                return
            file_name = os.path.join(self.path,
                                      self.input.external_forcing.extforcefile.filepath)
            
        if not file_name:
            return

        ext_old = hcdfm.ExtOldModel(file_name)
        if self.meteo.amu_file:
            forcing = hcdfm.ExtOldForcing(quantity='windx',
                                            filename= self.meteo.amu_file,
                                            filetype=hcdfm.ExtOldFileType.ArcInfo, #4
                                            method=hcdfm.ExtOldMethod.InterpolateTimeAndSpace, #2
                                            operand=hcdfm.Operand.override, #O
                                            )
            ext_old.forcing.append(forcing)

            # fid.write("QUANTITY=windx\n")
            # fid.write("FILENAME=" + self.meteo.amu_file + "\n")
            # fid.write("FILETYPE=4\n")
            # fid.write("METHOD=2\n")
            # fid.write("OPERAND=O\n")
            # fid.write("\n")
        
        if self.meteo.amv_file:
            forcing = hcdfm.ExtOldForcing(quantity='windy',
                                filename= self.meteo.amv_file,
                                filetype=hcdfm.ExtOldFileType.ArcInfo, #4
                                method=hcdfm.ExtOldMethod.InterpolateTimeAndSpace, #2
                                operand=hcdfm.Operand.override, #O
                                )
            ext_old.forcing.append(forcing)

            # fid.write("QUANTITY=windy\n")
            # fid.write("FILENAME=" + self.meteo.amv_file + "\n")
            # fid.write("FILETYPE=4\n")
            # fid.write("METHOD=2\n")
            # fid.write("OPERAND=O\n")
            # fid.write("\n")
        
        if self.meteo.amp_file:
            forcing = hcdfm.ExtOldForcing(quantity='atmosphericpressure',
                                filename= self.meteo.amp_file,
                                filetype=hcdfm.ExtOldFileType.ArcInfo, #4
                                method=hcdfm.ExtOldMethod.InterpolateTimeAndSpace, #2
                                operand=hcdfm.Operand.override, #O
                                )
            ext_old.forcing.append(forcing)
        
            # fid.write("QUANTITY=atmosphericpressure\n")
            # fid.write("FILENAME=" + self.meteo.amp_file + "\n")
            # fid.write("FILETYPE=4\n")
            # fid.write("METHOD=2\n")
            # fid.write("OPERAND=O\n")
            # fid.write("\n")

        if self.meteo.spw_file:
            forcing = hcdfm.ExtOldForcing(quantity='airpressure_windx_windy',
                                filename= self.meteo.spw_file,
                                filetype=hcdfm.ExtOldFileType.SpiderWebData, #5
                                method=hcdfm.ExtOldMethod.PassThrough, #1
                                operand=hcdfm.Operand.override, #O
                                )
            ext_old.forcing.append(forcing)

            # fid.write("QUANTITY=airpressure_windx_windy\n")
            # fid.write("FILENAME=" + self.meteo.spw_file + "\n")
            # fid.write("FILETYPE=5\n")
            # fid.write("METHOD=1\n")
            # fid.write("OPERAND=O\n")
            # fid.write("\n")    

        ext_old.save(filepath=file_name, path_style='windows') 


    ### Observation points ###

    def add_observation_point(self, x, y, name):
                
        self.observation_point.append(ObservationPoint(x, y, name, crs=None))

    def read_observation_points(self, file_name=None):
        
        self.observation_point = []

        if not file_name:
            if not self.input.output.obsfile:
                return
            file_name = os.path.join(self.path,
                                     self.input.output.obsfile[0].filepath)
                            
        if not os.path.exists(file_name):
            print("Warning : file " + file_name + " does not exist !")
            return
        
        # Loop through points
        df = pd.read_csv(file_name, index_col=False, header=None,
             delim_whitespace=True, names=['x', 'y', 'name'])
        
        for ind in range(len(df.x.values)):
            point = ObservationPoint(df.x.values[ind],
                                     df.y.values[ind],
                                     name=str(df.name.values[ind]))
            self.observation_point.append(point)

    def write_observation_points(self, file_name=None, path=None):

        if not path:
            path = self.path
            
        if not file_name:
            file_name = self.input.output.obsfile[0].filepath
            
        file_name = os.path.join(path, file_name)    

        if self.crs.is_geographic == 0:    
            fid = open(file_name, "w")
            for point in self.observation_point:
                string = f'{point.geometry.x:12.1f}{point.geometry.y:12.1f}  "{point.name}"\n'
                fid.write(string)
            fid.close()
        else:
            fid = open(file_name, "w")
            for point in self.observation_point:
                string = f'{point.geometry.x:12.6f}{point.geometry.y:12.6f}  "{point.name}"\n'
                fid.write(string)
            fid.close()
        
            
    ### Output ###

    def read_timeseries_output(self,
                               name_list = None,
                               path=None,
                               file_name = None,
                               file_name_wave = None):

        import xarray as xr
        import pandas as pd
        import numpy as np

        # Returns a dataframe with timeseries    
        
        if not path:
            path = self.path

        if not file_name:
            file_name = "flow_his.nc"

        file_name = os.path.join(path, file_name)
    
#        if not self.observation_point:
#            # First read observation points
#            self.read_observation_points()
                    
        # Open netcdf file
        ddd = xr.open_dataset(file_name)
        stations=ddd.waterlevel.coords["station_name"].values
        all_stations = []
        for ist, st in enumerate(stations):
            # st=str(st)[2:-1]
            # all_stations.append(st)
            all_stations.append(st.decode().strip())
        
        # If name_list is empty, add all points    
        if not name_list:
            name_list = []
            for st in all_stations:
                name_list.append(st)      
        
        if not file_name_wave:
            times   = ddd.waterlevel.coords["time"].values
            df = pd.DataFrame(index=times, columns=name_list)

            for station in name_list:
                for ist, st in enumerate(all_stations):
                    if station == st:
                        wl = ddd.waterlevel.values[:,ist]
                        wl[np.isnan(wl)] = -999.0
                        df[st]=wl
                        break            

            ddd.close()
        else:
            ddd.close()
            ddd={}
            for i,v in enumerate(file_name_wave):
                file_name = os.path.join(path, v)
                ddd[i] = xr.open_dataset(file_name)
            
            times   = ddd[0].Hsig.coords["time"].values
            
            df={}
            for station in name_list:
                for ist, st in enumerate(all_stations):
                    if station == st:

                        df[st] = pd.DataFrame(index=times, columns=["hs", "tp"])
                        for i,v in enumerate(file_name_wave): # Loop to get most detailed model with output at station location
                            Hs = ddd[i].Hsig.values[:,ist]
                            Tp = ddd[i].RTpeak.values[:,ist]
                            if max(Hs)>-999:
                                break
                        Tp[np.isnan(Hs)] = -999.0
                        Hs[np.isnan(Hs)] = -999.0
                        df[st]["hs"]=Hs
                        df[st]["tp"]=Tp                       
                        break    
            for i,v in enumerate(file_name_wave):        
                ddd[i].close()

        return df     

    def grid_coordinates(self, loc='cor'):

        cosrot = math.cos(self.input.rotation*math.pi/180)
        sinrot = math.sin(self.input.rotation*math.pi/180)
        if loc=="cor":
            xx     = np.linspace(0.0,
                                 self.input.mmax*self.input.dx,
                                 num=self.input.mmax + 1)
            yy     = np.linspace(0.0,
                                 self.input.nmax*self.input.dy,
                                 num=self.input.nmax + 1)
        else:
            xx     = np.linspace(0.5*self.input.dx,
                                 self.input.mmax*self.input.dx - 0.5*self.input.dx,
                                 num=self.input.mmax)
            yy     = np.linspace(0.5*self.input.dy,
                                 self.input.nmax*self.input.dy - 0.5*self.input.dy,
                                 num=self.input.nmax)
            
        xg0, yg0 = np.meshgrid(xx, yy)
        xg = self.input.x0 + xg0*cosrot - yg0*sinrot
        yg = self.input.y0 + xg0*sinrot + yg0*cosrot

        return xg, yg
    
    def bounding_box(self, crs=None):

        xg, yg = self.grid_coordinates(loc='cor')
        
        if crs:
            transformer = Transformer.from_crs(self.crs,
                                               crs,
                                               always_xy=True)
            xg, yg = transformer.transform(xg, yg)
        
        x_range = [np.min(np.min(xg)), np.max(np.max(xg))]
        y_range = [np.min(np.min(yg)), np.max(np.max(yg))]
        
        return x_range, y_range

    def outline(self, crs=None):

        xg, yg = self.grid_coordinates(loc='cor')
        
        if crs:
            transformer = Transformer.from_crs(self.crs,
                                               crs,
                                               always_xy=True)
            xg, yg = transformer.transform(xg, yg)
        
        xp = [ xg[0,0], xg[0,-1], xg[-1,-1], xg[-1,0], xg[0,0] ]
        yp = [ yg[0,0], yg[0,-1], yg[-1,-1], yg[-1,0], yg[0,0] ]
        
        return xp, yp
        
    def make_index_tiles(self, path, zoom_range=None):
        
        from tiling import deg2num
        from tiling import num2deg
        import fileops as fo
        
        if not zoom_range:
            zoom_range = [0, 13]

        npix = 256
        
        # Compute lon/lat range
        lon_range, lat_range = self.bounding_box(crs=CRS.from_epsg(4326))
        
        cosrot = math.cos(-self.input.rotation*math.pi/180)
        sinrot = math.sin(-self.input.rotation*math.pi/180)       
        
        transformer_a = Transformer.from_crs(CRS.from_epsg(4326),
                                             CRS.from_epsg(3857),
                                             always_xy=True)
        transformer_b = Transformer.from_crs(CRS.from_epsg(3857),
                                             self.crs,
                                             always_xy=True)
        
        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            
            print("Processing zoom level " + str(izoom))
        
            zoom_path = os.path.join(path, str(izoom))
        
            dxy = (40075016.686/npix) / 2 ** izoom
            xx = np.linspace(0.0, (npix - 1)*dxy, num=npix)
            yy = xx[:]
            xv, yv = np.meshgrid(xx, yy)
        
            ix0, iy0 = deg2num(lat_range[0], lon_range[0], izoom)
            ix1, iy1 = deg2num(lat_range[1], lon_range[1], izoom)
        
            for i in range(ix0, ix1 + 1):
            
                path_okay = False
                zoom_path_i = os.path.join(zoom_path, str(i))
            
                for j in range(iy0, iy1 + 1):
            
                    file_name = os.path.join(zoom_path_i, str(j) + ".dat")
            
                    # Compute lat/lon at ll corner of tile
                    lat, lon = num2deg(i, j, izoom)
            
                    # Convert to Global Mercator
                    xo, yo   = transformer_a.transform(lon,lat)
            
                    # Tile grid on local mercator
                    x        = xv[:] + xo + 0.5*dxy
                    y        = yv[:] + yo + 0.5*dxy
            
                    # Convert tile grid to crs of SFINCS model
                    x,y      = transformer_b.transform(x,y)
                    
                    # Now rotate around origin of SFINCS model
                    x00 = x - self.input.x0
                    y00 = y - self.input.y0
                    xg  = x00*cosrot - y00*sinrot
                    yg  = x00*sinrot + y00*cosrot
                    
                    iind = np.floor(xg/self.input.dx).astype(int)
                    jind = np.floor(yg/self.input.dy).astype(int)
                    ind  = iind*self.input.nmax + jind
                    ind[iind<0]   = -999
                    ind[jind<0]   = -999
                    ind[iind>255] = -999
                    ind[jind>255] = -999
#                    ind           = np.ascontiguousarray(np.transpose(ind))

                    # if i==142 and j==305:
                    
                    #     from matplotlib import pyplot as plt
                    #     fig, ax = plt.subplots(1,1)                                            
                    #     ax.plot(x,y)
                    #     ax.plot(x.transpose(),y.transpose())
                    #     ax.axis('equal')
                    #     x_range, y_range = self.bounding_box()
                    #     xp = [x_range[0], x_range[1],x_range[1],x_range[0],x_range[0]]
                    #     yp = [y_range[0], y_range[0],y_range[1],y_range[1],y_range[0]]
                    #     ax.plot(xp,yp)
                    #     xout, yout = self.outline()
                    #     ax.plot(xout,yout)
                    #     fig, ax = plt.subplots(1,1)                                            
                    #     ax.pcolor(ind.reshape([256, 256]))
                    #     xxx=1
                    
                    if np.any(ind>=0):
                        
                        if not path_okay:
                            if not os.path.exists(zoom_path_i):
                                fo.mkdir(zoom_path_i)
                                path_okay = True
                             
                        # And write indices to file
                        fid = open(file_name, "wb")
                        fid.write(ind)
                        fid.close()


    # def read(self, file_name):
    #     inp = Delft3DFMInput() # default values
    #     return inp
    
    # def write(self, file_name, inp):
    #     return

# class Delft3DFMInput():
#     def __init__(self, mdu):
        
        # for entry in mdu.dict:
        #     setattr(self, entry["name"], entry["default"])
        


class Delft3DFMBoundary():

    def __init__(self, quantity=None, locationfile=None, forcingfile=None):
        self.quantity     = quantity
        self.locationfile = locationfile
        self.forcingfile  = forcingfile
        self.geometry     = []
        self.point        = []
        
    def read_location_file(self, path=""):
        loc_file = os.path.join(path,
                                self.locationfile)

        from cht_utils.pli_file import read_pli_file
        d = read_pli_file(loc_file)
#        d = plifile(loc_file)
        name0 = os.path.split(loc_file)[-1][0:-4]
        for polyline in d:
            for ip, x in enumerate(polyline.x):
                name = name0 + "_" + str(ip + 1).zfill(4)
                point = Delft3DFMBoundaryPoint(x=polyline.x[ip],
                                               y=polyline.y[ip],
                                               name=name)            
                self.point.append(point)
        
    def read_forcing_file(self, path=""):

        frc_file = os.path.join(path,
                                self.forcingfile)
        
        if os.path.exists(frc_file):
            d = IniStruct(filename=frc_file)
            for ind, point in enumerate(self.point):
                point.data = d.section[ind].data        
        
    def plot(self,ax):
        pass

class Delft3DFMBoundaryPoint():

    def __init__(self, name=None, x=None, y=None, crs=None):
        self.name       = name
        self.geometry   = Point(x, y, crs=crs)
        self.data       = None # Use a pandas dataframe
    
    
class SfincsFlowBoundaryConditions():
    
    def __init__(self):
        self.geometry = []

    def read(self, bndfile, bzsfile):
        self.read_points(bndfile)
        self.read_time_series(bzsfile)

    def read_points(self, file_name):
        pass

    def read_time_series(self, file_name):
        pass
    
    def set_xy(self, x, y):
        self.geometry.x = x
        self.geometry.y = y
        pass
    
    def plot(self,ax):
        pass

class SfincsWaveBoundaryConditions():
    
    def __init__(self):
        self.geometry = []

    def read(self, bndfile, bzsfile):
        self.read_points(bndfile)
        self.read_time_series(bzsfile)

    def read_points(self, file_name):
        pass

    def read_time_series(self, file_name):
        pass
    
    def set_xy(self, x, y):
        self.geometry.x = x
        self.geometry.y = y
        pass
    
    def plot(self,ax):
        pass

class FlowBoundaryPoint():

    def __init__(self, x, y, name=None, crs=None, data=None, astro=None):
        
        self.name                   = name
        self.geometry               = Point(x, y, crs=crs)
        self.data                   = data
        self.astro                  = astro

class ObservationPoint():

    def __init__(self, x, y, name, crs=None):
        
        self.name     = name
        self.geometry = Point(x, y, crs=crs)

class Delft3DFMMeteo():
    
    def __init__(self):

        self.amu_file = None
        self.amv_file = None
        self.amp_file = None
        self.ampr_file = None
        self.spw_file = None

def read_timeseries_file(file_name, ref_date):
    
    # Returns a dataframe with time series for each of the columns

    df = pd.read_csv(file_name, index_col=0, header=None,
                      delim_whitespace=True)
    ts = ref_date + pd.to_timedelta(df.index, unit="s")
    df.index = ts
    
    return df
    
# class MDU():
    
#     def __init__(self):

#         self.dict = []

#         # model
#         self.dict.append({"name":"program",              "keyword":"Program",          "type":"str",   "default":"D-Flow FM",     "group":"model",    "comment":None})
#         self.dict.append({"name":"version",              "keyword":"Version",          "type":"str",   "default":"1.1.154.42806", "group":"model",    "comment":None})
#         self.dict.append({"name":"mduformatversion",     "keyword":"MDUFormatVersion", "type":"str",   "default":"1.02",          "group":"model",    "comment":None})
#         self.dict.append({"name":"autostart",            "keyword":"AutoStart",        "type":"int",   "default":1,               "group":"model",    "comment":"Autostart simulation after loading MDU or not (0=no, 1=autostart, 2=autostartstop)."})

#         # geometry
#         self.dict.append({"name":"netfile",              "keyword":"NetFile",          "type":"str",   "default":None,            "group":"geometry", "comment":None})
#         self.dict.append({"name":"bathymetryfile",       "keyword":"BathymetryFile",   "type":"str",   "default":None,            "group":"geometry", "comment":"*.xyb"})
#         self.dict.append({"name":"waterlevinifile",      "keyword":"WaterLevIniFile",  "type":"str",   "default":None,            "group":"geometry", "comment":"Initial water levels sample file *.xyz"})
#         self.dict.append({"name":"landboundaryfile",     "keyword":"LandBoundaryFile", "type":"str",   "default":None,            "group":"geometry", "comment":"Only for plotting"})
#         self.dict.append({"name":"thindamfile",          "keyword":"ThinDamFile",      "type":"str",   "default":None,            "group":"geometry", "comment":"*_thd.pli, Polyline(s) for tracing thin dams."})
#         self.dict.append({"name":"thindykefile",         "keyword":"ThindykeFile",     "type":"str",   "default":None,            "group":"geometry", "comment":"*._tdk.pli, Polyline(s) x,y,z, z = thin dyke top levels"})
#         self.dict.append({"name":"proflocfile",          "keyword":"ProflocFile",      "type":"str",   "default":None,            "group":"geometry", "comment":"*_proflocation.xyz)    x,y,z, z = profile refnumber"})
#         self.dict.append({"name":"profdeffile",          "keyword":"ProfdefFile",      "type":"str",   "default":None,            "group":"geometry", "comment":"*_profdefinition.def) definition for all profile nrs"})
#         self.dict.append({"name":"manholefile",          "keyword":"ManholeFile",      "type":"str",   "default":None,            "group":"geometry", "comment":None})
#         self.dict.append({"name":"waterlevini",          "keyword":"WaterLevIni",      "type":"float", "default":0.0,             "group":"geometry", "comment":"Initial water level"})
#         self.dict.append({"name":"botlevuni",            "keyword":"BotLevUni",        "type":"float", "default":-5.0,            "group":"geometry", "comment":"Uniform bottom level, (only if Botlevtype>=3, used at missing z values in netfile"})
#         self.dict.append({"name":"botlevtype",           "keyword":"BotLevType",       "type":"int",   "default":3,               "group":"geometry", "comment":None})
#         self.dict.append({"name":"anglat",               "keyword":"AngLat",           "type":"float", "default":0.0,             "group":"geometry", "comment":"Angle of latitude (deg), 0=no Coriolis"})
#         self.dict.append({"name":"conveyance2d",         "keyword":"Conveyance2D",     "type":"int",   "default":3,               "group":"geometry", "comment":"-1:R=HU,0:R=H, 1:R=A/P, 2:K=analytic-1D conv, 3:K=analytic-2D conv"})
#         self.dict.append({"name":"conveyance2d",         "keyword":"Conveyance2D",     "type":"int",   "default":3,               "group":"geometry", "comment":"-1:R=HU,0:R=H, 1:R=A/P, 2:K=analytic-1D conv, 3:K=analytic-2D conv"})

#         # numerics
#         self.dict.append({"name":"cflmax",               "keyword":"CFLMax",           "type":"float", "default":0.7,             "group":"numerics", "comment":"Max. Courant nr."})
#         self.dict.append({"name":"cflwavefrac",          "keyword":"CFLWaveFrac",      "type":"float", "default":0.1,             "group":"numerics", "comment":"Wave velocity fraction, total courant vel = u + cflw*wavevelocity"})
#         self.dict.append({"name":"advectype",            "keyword":"AdvecType",        "type":"int",   "default":3,               "group":"numerics", "comment":"Adv type, 0=no, 1= Wenneker, qu-udzt, 2=1, q(uio-u), 3=Perot q(uio-u), 4=Perot q(ui-u), 5=Perot q(ui-u) without itself"})
#         self.dict.append({"name":"limtypsa",             "keyword":"Limtypsa",         "type":"int",   "default":0,               "group":"numerics", "comment":"Limiter type for salinity transport,           0=no, 1=minmod,2=vanLeer,3=Kooren,4=Monotone Central"})
#         self.dict.append({"name":"Hdam",                 "keyword":"Hdam",             "type":"float", "default":0.0,             "group":"numerics", "comment":"Threshold for minimum bottomlevel step at which to apply energy conservation factor i.c. flow contraction"})
#         self.dict.append({"name":"Icgsolver",            "keyword":"Icgsolver",        "type":"float", "default":4.0,             "group":"numerics", "comment":"Solver type (1: sobekGS_OMP, 2: sobekGS_OMPthreadsafe, 3: sobekGS, 4: sobekGS + Saadilud, 5: parallel/global Saad, 6: parallel/Petsc, 7: parallel/GS)"})
#         self.dict.append({"name":"Tlfsmo",               "keyword":"Tlfsmo",           "type":"float", "default":0.0,             "group":"numerics", "comment":"Fourier smoothing time on water level boundaries"})
#         self.dict.append({"name":"Qhrelax",              "keyword":"Qhrelax",          "type":"float", "default":0.0,             "group":"numerics", "comment":"Relaxation on Q-h open boundaries"})

#         # physics
#         self.dict.append({"name":"uniffrictcoef",        "keyword":"UnifFrictCoef",    "type":"float", "default":0.02,            "group":"physics",  "comment":"Uniform friction coefficient, 0=no friction"})
#         self.dict.append({"name":"uniffrictcoef1D",        "keyword":"UnifFrictCoef1D",    "type":"float", "default":0.02,            "group":"physics",  "comment":"Uniform friction coefficient in 1D links, 0=no friction"})

#         self.dict.append({"name":"uniffricttype",        "keyword":"UnifFrictType",    "type":"int",   "default":1,               "group":"physics",  "comment":"0=Chezy, 1=Manning, 2=White Colebrook, 3=z0 etc"})
#         self.dict.append({"name":"vicouv",               "keyword":"Vicouv",           "type":"float", "default":1.0,             "group":"physics",  "comment":"Uniform horizontal eddy viscosity"})
#         self.dict.append({"name":"smagorinsky",          "keyword":"Smagorinsky",      "type":"float", "default":0.0,             "group":"physics",  "comment":"Add Smagorinsky horizontal turbulence : vicu = vicu + ( (Smagorinsky*dx)**2)*S, e.g. 0.1"})
#         self.dict.append({"name":"elder",                "keyword":"Elder",            "type":"float", "default":0.0,             "group":"physics",  "comment":"Add Elder contribution                : vicu = vicu + Elder*kappa*ustar*H/6),   e.g. 1.0"})
#         self.dict.append({"name":"wall_ks",              "keyword":"wall_ks",          "type":"float", "default":0.01,            "group":"physics",  "comment":"Nikuradse roughness for side walls, wall_z0=wall_ks/30"})
#         self.dict.append({"name":"vicoww",               "keyword":"Vicoww",           "type":"float", "default":1.0e-06,             "group":"physics",  "comment":"Uniform vertical eddy viscosity"})
#         self.dict.append({"name":"dicoww",               "keyword":"Dicoww",           "type":"float", "default":1.0e-06,             "group":"physics",  "comment":"Uniform vertical eddy diffusivity"})

#         self.dict.append({"name":"tidalforcing",         "keyword":"TidalForcing",     "type":"float", "default":1,               "group":"physics",  "comment":"Tidal forcing (0=no, 1=yes) (only for jsferic == 1)"})
#         self.dict.append({"name":"salinity",             "keyword":"Salinity",         "type":"float", "default":0,               "group":"physics",  "comment":"Include salinity, (0=no, 1=yes)"})
#         self.dict.append({"name":"temperature",          "keyword":"Temperature",      "type":"float", "default":0,               "group":"physics",  "comment":"Include temperature (0: no, 1: only transport, 3: excess model of D3D, 5: composite (ocean) model)"})
#         self.dict.append({"name":"rhomean",              "keyword":"Rhomean",          "type":"float", "default":1024.0,          "group":"physics",  "comment":"Average water density (kg/m3)"})
#         self.dict.append({"name":"ag",                   "keyword":"Ag",               "type":"float", "default":9.81,            "group":"physics",  "comment":"Gravitational acceleration"})
#         self.dict.append({"name":"stanton",              "keyword":"Stanton",          "type":"float", "default":0.0013,          "group":"physics",  "comment":"Coefficient for convective heat flux"})
#         self.dict.append({"name":"dalton",               "keyword":"Dalton",           "type":"float", "default":0.0013,          "group":"physics",  "comment":"Coefficient for evaporative heat flux"})
#         self.dict.append({"name":"backgroundwatertemperature", "keyword":"Backgroundwatertemperature",  "type":"float", "default":20,          "group":"physics",  "comment":"Background water temperature for eqn. of state"})


#         # wind
#         self.dict.append({"name":"icdtyp",               "keyword":"ICdtyp",           "type":"int",   "default":3,               "group":"wind",     "comment":"( ), Cd = const, 2=S&B 2 breakpoints, 3= S&B 3 breakpoints"})
#         self.dict.append({"name":"cdbreakpoints",        "keyword":"Cdbreakpoints",    "type":"list",     "default":[1.0e-03,3.1e-03,1.5e-03], "group":"wind",     "comment":"( ),   e.g. 0.00063  0.00723"})
#         self.dict.append({"name":"windspeedbreakpoints", "keyword":"Windspeedbreakpoints", "type":"list",  "default":[0.0e+00,2.8e+01,5.0e+01], "group":"wind",     "comment":"(m/s), e.g. 0.0      100.0"})
#         self.dict.append({"name":"pavbnd",               "keyword":"PavBnd",           "type":"float", "default":1.013e+05,       "group":"wind",     "comment":"Background pressure (Pa)"})
#         self.dict.append({"name":"rhoair",               "keyword":"Rhoair",           "type":"float", "default":1.2,       "group":"wind",     "comment":"Air density"})

#         # waves
#         self.dict.append({"name":"wavemodelnr",          "keyword":"Wavemodelnr",           "type":"int",   "default":3,               "group":"waves",     "comment":"Wave model nr. (0: none, 1: fetch/depth limited Hurdle-Stive, 2: Young-Verhagen, 3: SWAN)"})
#         self.dict.append({"name":"wavenikuradse",        "keyword":"WaveNikuradse",         "type":"float", "default":0.01,            "group":"waves",     "comment":"Wave friction Nikuradse ks c , used in Krone-Swart"})
#         self.dict.append({"name":"rouwav",               "keyword":"Rouwav",                "type":"str",   "default":"FR84",          "group":"waves",     "comment":"Friction model for wave induced shear stress"})
#         self.dict.append({"name":"gammax",               "keyword":"Gammax",                "type":"float", "default":1.0,             "group":"waves",     "comment":"Maximum wave height/water depth ratio"})

#         # time
#         self.dict.append({"name":"refdate",              "keyword":"RefDate",         "type":"date",  "default":20000101,         "group":"time",     "comment":"Reference date (yyyymmdd)"})
#         self.dict.append({"name":"tunit",                "keyword":"Tunit",           "type":"str",   "default":"s",              "group":"time",     "comment":"Time units in MDU (H, M or S)"})
#         self.dict.append({"name":"dtuser",               "keyword":"DtUser",          "type":"float", "default":300.0,            "group":"time",     "comment":"User timestep in seconds (interval for external forcing update)"})
#         self.dict.append({"name":"dtmax",                "keyword":"DtMax",           "type":"float", "default":300.0,            "group":"time",     "comment":"Max timestep in seconds"})
#         self.dict.append({"name":"dtinit",               "keyword":"DtInit",          "type":"float", "default":1.0,              "group":"time",     "comment":"Initial timestep in seconds"})
#         self.dict.append({"name":"autotimestep",         "keyword":"AutoTimestep",    "type":"int",   "default":1,                "group":"time",     "comment":"Use CFL timestep limit or not (1/0)"})
#         self.dict.append({"name":"tstart",               "keyword":"TStart",          "type":"float", "default":0.0,              "group":"time",     "comment":"Start time w.r.t. RefDate (in TUnit)"})
#         self.dict.append({"name":"tstop",                "keyword":"TStop",           "type":"float", "default":86400.0,          "group":"time",     "comment":"Stop time w.r.t. RefDate (in TUnit)"})
        
#         # restart
#         self.dict.append({"name":"restartfile",          "keyword":"RestartFile",     "type":"str",   "default":None,             "group":"restart",     "comment":"Restart netcdf-file, either *_rst.nc or *_map.nc"})
#         self.dict.append({"name":"restartdatetime",      "keyword":"RestartDateTime", "type":"float", "default":0.0,              "group":"restart",     "comment":"Restart date and time when restarting from *_map.nc [YYYY-MM-DD HH:MM:SS]"})

#         # external forcing
#         self.dict.append({"name":"extforcefile",         "keyword":"ExtForceFile",    "type":"str",   "default":None,             "group":"external forcing",     "comment":"*.ext"})
#         self.dict.append({"name":"extforcefilenew",      "keyword":"ExtForceFileNew", "type":"str",   "default":None,             "group":"external forcing",     "comment":"*.ext (new format)"})
#         self.dict.append({"name":"extforcefileold",      "keyword":"ExtForceFileOld", "type":"str",   "default":None,             "group":"external forcing",     "comment":"*.ext (old format)"})

#         # output
#         self.dict.append({"name":"obsfile",              "keyword":"ObsFile",         "type":"str",   "default":None,             "group":"output",   "comment":"*.xyn Coords+name of observation stations"})
#         self.dict.append({"name":"crsfile",              "keyword":"CrsFile",         "type":"str",   "default":None,             "group":"output",   "comment":"*_crs.pli Polyline(s) definining cross section(s)"})
#         self.dict.append({"name":"hisfile",              "keyword":"HisFile",         "type":"str",   "default":None,             "group":"output",   "comment":"*_his.nc History file in NetCDF format"})
#         self.dict.append({"name":"hisinterval",          "keyword":"HisInterval",     "type":"float", "default":600.0,            "group":"output",   "comment":"Interval (in s) between history outputs"})
#         self.dict.append({"name":"xlsinterval",          "keyword":"XLSInterval",     "type":"float", "default":0.0,              "group":"output",   "comment":"Interval (in s) between XLS history"})
#         self.dict.append({"name":"flowgeomfile",         "keyword":"FlowGeomFile",    "type":"str",   "default":None,             "group":"output",   "comment":"*_flowgeom.nc Flow geometry file in NetCDF format"})
#         self.dict.append({"name":"mapfile",              "keyword":"MapFile",         "type":"str",   "default":None,             "group":"output",   "comment":"*_map.nc Map file in NetCDF format"})
#         self.dict.append({"name":"mapinterval",          "keyword":"MapInterval",     "type":"float", "default":3600.0,           "group":"output",   "comment":"Interval (in s) between map file outputs"})
#         self.dict.append({"name":"rstinterval",          "keyword":"RstInterval",     "type":"float", "default":0.0,              "group":"output",   "comment":"Interval (in s) between restart file outputs"})
#         self.dict.append({"name":"waqfilebase",          "keyword":"WaqFileBase",     "type":"str",   "default":None,             "group":"output",   "comment":"Basename (without extension) for all Delwaq files to be written"})
#         self.dict.append({"name":"waqinterval",          "keyword":"WaqInterval",     "type":"float", "default":0.0,              "group":"output",   "comment":"Interval (in s) between Delwaq file outputs"})
#         self.dict.append({"name":"snapshotdir",          "keyword":"SnapshotDir",     "type":"str",   "default":None,             "group":"output",   "comment":"Directory where snapshots/screendumps are saved."})
#         self.dict.append({"name":"outputdir",            "keyword":"OutputDir",       "type":"str",   "default":None,             "group":"output",   "comment":"Output directory of map-, his-, rst-, dat- and timings-files, default: DFM_OUTPUT_<modelname>. Set to . for current dir."})