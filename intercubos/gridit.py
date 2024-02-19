import os
import tempfile
import itertools as it
from pyproj import Transformer
from shapely.geometry import Point, Polygon, box
import geopandas as gpd
import numpy as np
import folium
import logging

class Grid:
    def __init__(self, sw_lon, sw_lat, ne_lon, ne_lat, stepsize=10000):
        """
        Args:
          stepsize (int): Stepsize in meters, e.g. 10000 == 10km
        """
        # Set up transformers, EPSG:3857 is metric, same as EPSG:900913
        self.to_proxy_transformer = Transformer.from_crs(
            'epsg:4326', 'epsg:3857')
        self.to_original_transformer = Transformer.from_crs(
            'epsg:3857', 'epsg:4326'
        )
        
        # Create corners of rectangle to be transformed to a grid
        sw = Point((sw_lon, sw_lat))
        ne = Point((ne_lon, ne_lat))
    
        
        # Project corners to target projection
        self.transformed_sw = self.to_proxy_transformer.transform(sw.x, sw.y)
        self.transformed_ne = self.to_proxy_transformer.transform(ne.x, ne.y)
        
        # Iterate over 2D area
        self.gridpoints = []
        x = self.transformed_sw[0]
        while x < (self.transformed_ne[0]+stepsize):
            self.gridpoints.append([])
            y = self.transformed_sw[1]
            while y < (self.transformed_ne[1]+stepsize):
                p = Point(
                    self.to_original_transformer.transform(x, y)
                ) # x == lon, y == lat  
                self.gridpoints[-1].append(p)
                y += stepsize
            x += stepsize

        self.grid_polygons = [
            box(
                gp.x, gp.y,
                self.gridpoints[gp_xi+1][gp_yi+1].x,
                self.gridpoints[gp_xi+1][gp_yi+1].y
            )
            for gp_xi,gp_row in enumerate(self.gridpoints)
            for gp_yi,gp in enumerate(gp_row)
            if gp_xi < len(self.gridpoints)-1 and gp_yi < len(gp_row)-1
        ]
        self.grid = gpd.GeoDataFrame(
            {'geometry':self.grid_polygons}, crs="EPSG:4326"
        )
        #grid.to_file("grid.shp")

    def add_to_map(self, m):
        grid_layer = folium.FeatureGroup("grid").add_to(m)
        for gp in it.chain(*self.gridpoints):
            plusIcon = folium.DivIcon(
                html='<p>+</p>'
                #'<svg height="100" width="100">'
                #'<text x="50" y="50" fill="black">+</text></svg>'
            )
            folium.Marker(
                location=[gp.x,gp.y],
                icon=plusIcon
            ).add_to(grid_layer)
        folium.LayerControl().add_to(m)

    def assign_to_grid(self, data, colname='count'):
        # https://james-brennan.github.io/posts/fast_gridding_geopandas/
        if not isinstance(data, gpd.GeoDataFrame):
            data = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(
                    data.decimalLongitude,
                    data.decimalLatitude, crs="EPSG:4326"
                )
            ).drop(['decimalLongitude','decimalLatitude'], axis=1)
        merged = gpd.sjoin(
            data, self.grid, how='left', predicate='within'
        )
        #merged.dissolve(by="index_right", aggfunc="count")
        self.grid[colname] = merged.index_right.value_counts()
        self.grid = self.grid.fillna(0)

    def plot(self, colname='count', crs=None, zoom='auto',
             filename=None, colorbar=True, vmax=None,
             logcol=False, ax=None, bax=True):
        import matplotlib.colors as mcolors
        import numpy as np
        import contextily as cx
        #https://kbkb-wx-python.blogspot.com/2015/12/python-transparent-colormap.html
        colors = [
            (1,0,0,c) for c in np.linspace(0,1,100)
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'custom_cmap', colors, N=10
        )
        ax = (self.grid.to_crs(crs) if crs else self.grid).plot(
            column=colname, figsize=(10,10), edgecolor='k',
            vmin=0,vmax=vmax,cmap=cmap,
            #https://matplotlib.org/stable/users/explain/colors/colormapnorms.html
            norm=(mcolors.SymLogNorm(vmin=0, vmax=vmax, linthresh=5) if logcol else None),
            legend=colorbar, ax=ax
        ) # cannot use general alpha or overwrites alpha cmap
        if bax: cx.add_basemap(ax, crs=crs, zoom=zoom)
        if filename:
            ax.get_figure().savefig(filename)
        return ax

    def time_lapse_plot(self, data, filename,
                        time_column='eventDate',
                        time_resolution='year',
                        crs=None, figsize=(10,10),
                        duration=500):
        import dateutil.parser
        from dateutil.parser import ParserError
        from operator import attrgetter
        import matplotlib.pyplot as plt
        import imageio.v3 as iio
        
        def trydate(timestr):
            try: return dateutil.parser.parse(timestr)
            except ParserError: return None
        time_column = data[time_column].apply(trydate)
        if nasum := (nans := time_column.isna()).sum():
            logging.warning('Filtering %s non-parseable dates', nasum)
            data = data[~nans]
            time_column = time_column.dropna()
        time_grouping = time_column.apply(attrgetter('year'))
        with tempfile.TemporaryDirectory(prefix=os.path.dirname(filename)+'/tl') as tmpdir:
            max_grid_cells = {}
            for grpname, grp in data.groupby(time_grouping):
                self.assign_to_grid(grp, colname=f"counts_{grpname}")
                max_grid_cells[grpname] = self.grid[f"counts_{grpname}"].max()
            vmax = max(max_grid_cells.values())
            min_t, max_t = min(max_grid_cells), max(max_grid_cells)
            for grpname in max_grid_cells:
                fig, axes = plt.subplots(nrows=2,ncols=1, height_ratios=[12, 1], figsize=figsize)
                self.plot(
                    crs=crs,
                    colorbar=True,colname=f"counts_{grpname}",
                    vmax=vmax, ax=axes[0]
                )
                make_timeline(axes[1], grpname, min_t, max_t)
                fig.savefig(f"{tmpdir}/{grpname}.png")
            images = list()
            for grpname in max_grid_cells:
                images.append(
                    iio.imread(f"{tmpdir}/{grpname}.png")
                )
            frames = np.stack(images, axis=0)
            iio.imwrite(filename, frames, duration=duration)
            #from pygifsicle import optimize
            #optimize(filename)

def make_timeline(ax, time, min_t, max_t, tick_interval=5, fontsize=12):
    # https://github.com/souravbhadra/maplapse/blob/main/maplapse/maplapse.py
    ax.axhline(y=0.5, color='darkgray', linestyle='-', zorder=1)
    ax.scatter(x=int(time), y=0.5, zorder=2, color='b', s=30, marker='o')
    ax.set_xlim(left=min_t, right=max_t)
    ax.set_ylim(bottom=0.45, top=0.55)
    for k in np.linspace(min_t, max_t, tick_interval):
        ax.text(
            int(k), 0.56, f"{int(k)}",
            ha='center', va='top', fontsize=fontsize
        )
    ax.set_axis_off()
    ax.set_title(f'{time}', fontsize=fontsize)

