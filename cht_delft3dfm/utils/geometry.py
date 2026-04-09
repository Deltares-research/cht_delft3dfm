# -*- coding: utf-8 -*-
"""
Created on Sun May 16 14:56:46 2021

@author: ormondt
"""

import numpy as np
import geopandas as gpd
import shapely
import math


class Geometry:
    def __init__(self):
        pass


class RegularGrid(Geometry):
    def __init__(
        self,
        hw,
        x0=None,
        y0=None,
        dx=None,
        dy=None,
        nmax=None,
        mmax=None,
        rotation=None,
        crs=None,
    ):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.mmax = mmax
        self.nmax = nmax
        self.rotation = rotation
        self.crs = crs
        if x0:
            self.xg, self.yg = self.grid_coordinates_corners()
            self.xz, self.yz = self.grid_coordinates_centres()

    def build(self, x0, y0, dx, dy, nx, ny, rotation, crs):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.mmax = nx
        self.nmax = ny
        self.rotation = rotation
        self.xg, self.yg = self.grid_coordinates_corners()
        self.xz, self.yz = self.grid_coordinates_centres()
        self.crs = crs

    def grid_coordinates_corners(self):

        cosrot = np.cos(self.rotation * np.pi / 180)
        sinrot = np.sin(self.rotation * np.pi / 180)
        xx = np.linspace(0.0, self.mmax * self.dx, num=self.mmax + 1)
        yy = np.linspace(0.0, self.nmax * self.dy, num=self.nmax + 1)
        xg0, yg0 = np.meshgrid(xx, yy)
        xg = self.x0 + xg0 * cosrot - yg0 * sinrot
        yg = self.y0 + xg0 * sinrot + yg0 * cosrot

        return xg, yg

    def grid_coordinates_centres(self):

        cosrot = np.cos(self.rotation * np.pi / 180)
        sinrot = np.sin(self.rotation * np.pi / 180)
        xx = np.linspace(
            0.5 * self.dx, self.mmax * self.dx - 0.5 * self.dx, num=self.mmax
        )
        yy = np.linspace(
            0.5 * self.dy, self.nmax * self.dy - 0.5 * self.dy, num=self.nmax
        )
        xg0, yg0 = np.meshgrid(xx, yy)
        xz = self.x0 + xg0 * cosrot - yg0 * sinrot
        yz = self.y0 + xg0 * sinrot + yg0 * cosrot

        return xz, yz

    def plot(self, ax):

        pass

    def to_gdf(self):

        lines = []

        cosrot = math.cos(self.rotation * math.pi / 180)
        sinrot = math.sin(self.rotation * math.pi / 180)

        for n in range(self.nmax):
            for m in range(self.mmax):
                xa = self.x0 + m * self.dx * cosrot - n * self.dy * sinrot
                ya = self.y0 + m * self.dx * sinrot + n * self.dy * cosrot
                xb = self.x0 + (m + 1) * self.dx * cosrot - n * self.dy * sinrot
                yb = self.y0 + (m + 1) * self.dx * sinrot + n * self.dy * cosrot
                line = shapely.geometry.LineString([[xa, ya], [xb, yb]])
                lines.append(line)
                xb = self.x0 + m * self.dx * cosrot - (n + 1) * self.dy * sinrot
                yb = self.y0 + m * self.dx * sinrot + (n + 1) * self.dy * cosrot
                line = shapely.geometry.LineString([[xa, ya], [xb, yb]])
                lines.append(line)
        geom = shapely.geometry.MultiLineString(lines)
        gdf = gpd.GeoDataFrame(crs=self.crs, geometry=[geom])

        return gdf


class Point:

    def __init__(self, x, y, name=None, crs=None):

        self.x = x
        self.y = y
        self.crs = crs
        self.name = name
        self.data = None


class Polyline(Geometry):

    def __init__(self, x=None, y=None, crs=None, name=None, closed=False):

        self.point = []
        self.name = name
        self.data = None
        self.closed = closed
        self.crs = crs

        if x is not None:
            for j, xp in enumerate(x):
                pnt = Point(x[j], y[j])
                self.point.append(pnt)

    def add_point(self, x, y, name=None, data=None, position=-1):

        pnt = Point(x, y, name=name, data=data)
        if position < 0:
            # Add point to the end
            self.point.append(pnt)
        else:
            #
            pass

    def plot(self, ax=None):
        pass
