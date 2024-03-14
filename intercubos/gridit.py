import os
import tempfile
import itertools as it
from pyproj import Transformer
from shapely.geometry import Point, Polygon, box
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import logging
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, sw_lon, sw_lat, ne_lon, ne_lat,  stepsize=10000):
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
        self.grid = self.grid.copy() #TODO step to avoid fragmentation of dataframe, but is not a good strategy
        self.grid = self.grid.fillna(0)

    def remove_empty_grid_cells(self, columns=None):
        """
        Based on sum of columns remove empty cells

        Args:
          columns (None|str|list): Can be a single column name
          or list of columns names. If not provide, all columns
          except 'geometry' are used for calculating.
        """
        if columns is None:
            columns = list(set(self.grid.columns) - {'geometry'})
        elif isinstance(columns, str):
            columns = [columns]
        non_empty_cells = self.grid[columns].sum(axis=1) > 0
        logging.info(
            'Filtered %s empty cells from %s total',
            (~non_empty_cells).sum(), len(non_empty_cells)
        )
        self.grid = self.grid[non_empty_cells].copy()
        
    def plot(self, colname='count', crs=None, zoom='auto',
             filename=None, edgecolor='k', color=(1,0,0),
             colorbar=True, vmax=None, figsize=(10,10),
             logcol=False, ax=None, bax=True):
        import matplotlib.colors as mcolors
        import numpy as np
        import contextily as cx
        #https://kbkb-wx-python.blogspot.com/2015/12/python-transparent-colormap.html
        colors = [
            color+(c,) for c in np.linspace(0,1,100)
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'custom_cmap', colors, N=10
        )
        ax = (self.grid.to_crs(crs) if crs else self.grid).plot(
            column=colname, figsize=figsize, edgecolor=edgecolor,
            vmin=0,vmax=vmax,cmap=cmap,
            #https://matplotlib.org/stable/users/explain/colors/colormapnorms.html
            norm=(mcolors.SymLogNorm(vmin=0, vmax=vmax, linthresh=5) if logcol else None),
            legend=colorbar, ax=ax
        ) # cannot use general alpha or overwrites alpha cmap
        if bax: cx.add_basemap(ax, crs=crs, zoom=zoom)
        if filename:
            ax.get_figure().savefig(filename, transparent=True)
        return ax

    def plot_interaction(self, counts1, counts2,
                         crs=None, figsize=(10,10),
                         fillcolors=((1,0,0),(0,0,1)),
                         logcol=False, gridcolor='k',
                         colorbar=True, vmax=None,
                         zoom='auto', filename=None,
                         title=True
                         ):

        #fig, ax = plt.subplots(nrows=1,ncols=1,figsize=figsize)
        ax = self.plot(
            crs=crs, edgecolor=gridcolor,
            colorbar=colorbar, color=fillcolors[0],
            logcol=logcol, vmax=vmax, colname=counts1,
            zoom=zoom, figsize=figsize
        )
        self.plot(
            crs=crs, edgecolor=gridcolor,
            colorbar=colorbar, color=fillcolors[1],
            logcol=logcol, vmax=vmax, colname=counts2,
            zoom=zoom,ax=ax
        )
        if title: ax.set_title(f"{counts1} - {counts2}")
        if filename: ax.get_figure().savefig(filename, transparent=True)
    
    def time_lapse_plot(self, data, filename,
                        time_column='eventDate',
                        time_resolution='year',
                        crs=None, figsize=(10,10),
                        fillcolors=((1,0,0),(0,0,1),(0,1,0)),
                        logcol=False, gridcolor='k',
                        duration=500):
        import dateutil.parser
        from dateutil.parser import ParserError
        from operator import attrgetter
        import imageio.v3 as iio

        if not isinstance(data, list):
            cubes = [data]
        else: cubes = data
        
        def trydate(timestr):
            try: return dateutil.parser.parse(timestr)
            except ParserError: return None

        time_column_name = time_column
        frame_figs = {}
        with tempfile.TemporaryDirectory(
                prefix=os.path.dirname(filename)+'/tl') as tmpdir:
            for i,data in enumerate(cubes):
                time_column = data[time_column_name].apply(trydate)
                if nasum := (nans := time_column.isna()).sum():
                    logging.warning('Filtering %s non-parseable dates', nasum)
                    data = data[~nans]
                    time_column = time_column.dropna()
                time_grouping = time_column.apply(attrgetter(time_resolution))
                max_grid_cells = {}
                for grpname, grp in data.groupby(time_grouping):
                    self.assign_to_grid(grp, colname=f"counts_{grpname}")
                    max_grid_cells[grpname] = self.grid[f"counts_{grpname}"].max()
                vmax = max(max_grid_cells.values())
                min_t, max_t = min(max_grid_cells), max(max_grid_cells)
                for grpname in max_grid_cells:
                    if not i: # only for first cube
                        fig, axes = plt.subplots(
                            nrows=2,ncols=1,
                            height_ratios=[12, 1],
                            figsize=figsize
                        )
                        make_timeline(axes[1], grpname, min_t, max_t)
                        frame_figs[grpname] = (fig, axes)
                    else: fig, axes = frame_figs[grpname]
                    self.plot(
                        crs=crs, edgecolor=gridcolor,
                        colorbar=True, color=fillcolors[i],
                        logcol=logcol,
                        colname=f"counts_{grpname}",
                        vmax=vmax, ax=axes[0]
                    )
                    if i == (len(cubes)-1):
                        fig.savefig(f"{tmpdir}/{grpname}.png", transparent=True)
            
            images = list()
            for grpname in max_grid_cells:
                images.append(
                    iio.imread(f"{tmpdir}/{grpname}.png")
                )
            frames = np.stack(images, axis=0)
            iio.imwrite(filename, frames, duration=duration, loop=4)
            #from pygifsicle import optimize
            #optimize(filename)

    def interactions(self, cube1, cube2=None,
                     taxColumn='species',
                     remove_empty_grid_cells=True,
                     minimumAbundance=1,
                     filename=None, crs='EPSG:3857',
                     zoom='auto'):
        """Measure interactions base on co-occurrence
        and relate to known interactions

        Args:
          cube1 (OcCube): Occurrence cube for analysis
          cube2 (None|OcCube): If provided, interactions between
            cubes are analysed. Otherwise within cube interactions
            are considered.
          taxColumn (str): Column name that provides the taxa names for both cubes.
          minimumAbundance (int): Minimum abundances for the taxa, set to 1 to include all.
        
        """
        cube1_taxa = cube1.cube[taxColumn].value_counts()
        if (poorTaxa:=
            (taxaSelection:=
             (cube1_taxa < minimumAbundance)).sum()):
            logging.info(
                'Filtering %s from %s %s in cube1',
                poorTaxa, len(taxaSelection), taxColumn
            )
            cube1_taxa = cube1_taxa[~taxaSelection]
        for t in cube1_taxa.index:
            self.assign_to_grid(
                cube1.cube[cube1.cube[taxColumn]==t],
                colname=t # could include cube1 specifier
            )
        if cube2 is None:
            infra_cube_analysis = True
            cube2_taxa = cube1_taxa
        else:
            infra_cube_analysis = False
            cube2_taxa = cube2.cube[taxColumn].value_counts()
            if (poorTaxa:=
                (taxaSelection:=
                 (cube2_taxa < minimumAbundance)).sum()):
                logging.info(
                    'Filtering %s from %s %s in cube2',
                    poorTaxa, len(taxaSelection), taxColumn
                )
                cube2_taxa = cube2_taxa[~taxaSelection]
            if cube2_taxa.index.isin(
                    cube1_taxa.index).sum():
                raise ValueError('Shared taxa in cube1 and cube2')
            for t in cube2_taxa.index:
                self.assign_to_grid(
                    cube2.cube[cube2.cube[taxColumn]==t],
                    colname=t # could include cube1 specifier
            )
        # Optionally remove empty cells
        if remove_empty_grid_cells:
            # Could make it more restricted on interaction columns
            self.remove_empty_grid_cells()
        # Calculate co-occurrence statistic
        from scipy.stats import chi2_contingency
        con_p = {}
        con_c = {}
        contingencies = {}
        for t1 in cube1_taxa.index:
            con_p[t1] = {}
            con_c[t1] = {}
            contingencies[t1] = {}
            for t2 in cube2_taxa.index:
                if t1 == t2: continue # skipping self interaction
                contingency = pd.crosstab(
                    self.grid[t1]>0,
                    self.grid[t2]>0
                )
                c, p, dof, expected = chi2_contingency(contingency)
                con_p[t1][t2] = p
                con_c[t1][t2] = c
                contingencies[t1][t2] = contingency
        con_p = pd.DataFrame(con_p)
        con_c = pd.DataFrame(con_c)
        top_ix = con_p.stack().sort_values().head()
        print(top_ix)
        if filename:
            self.plot_interaction(
                *top_ix.index[0], crs=crs, zoom=zoom,
                filename=filename
            )
        return con_p

class GridScan:
    def __init__(self, cube1, cube2=None,
                 remove_empty_grid_cells=True,bounds=None):
        self.cube1 = cube1
        self.cube2 = cube2
        # TODO total bounds from cub1 and cube2
        self.bounds = bounds or cube1.cube.total_bounds
        # bounds -> sw_lon, sw_lat, ne_lon, ne_lat
        
    def scan(self, start_size, iterations=4, linear=False,
             remove_empty_grid_cells=True,minimumAbundance=10,
             outdir=None, zoom='auto', crs='EPSG:3857',
             simulate=False, sig_level=.05, figsize=(10,10),
             plotfile='relevant_cooccurrence.png'):
        size_experiments = {}
        for iteration in range(iterations):
            current_size = (
                (start_size/(iteration+1)) if linear
                else (start_size/(2**iteration))
            )
            if simulate:
                print('Iteration', iteration, 'with', current_size, 'm grid cell')
                continue
            grid = Grid(*self.bounds, stepsize=current_size)
            outplot = os.path.join(
                outdir,f"{current_size}.png"
            ) if outdir else None
            size_experiments[current_size] = grid.interactions(
                cube1=self.cube1,cube2=self.cube2,
                minimumAbundance=minimumAbundance,
                remove_empty_grid_cells=remove_empty_grid_cells,
                filename=outplot, crs=crs, zoom=zoom
            )
            if outdir:
                for size in sorted(size_experiments):
                    se = size_experiments[size].unstack()
                    se[
                        se < .05
                    ].to_csv(os.path.join(outdir,f"sig_cooc_at_{size}km.csv"))
            if outdir and plotfile:
                sig_results = { # set to km
                    s/1000:(size_experiments[s] <= sig_level
                       ).sum().sum() for s in size_experiments
                }
                fig, ax = plt.subplots(figsize=figsize)
                ax.scatter(
                    sig_results.keys(),sig_results.values()
                )
                ax.set_xlabel('Grid cell size (km)')
                ax.set_ylabel(f"Chi2 tests <= {sig_level}(#)")
                ax.set_title('Grid cell size vs relevant co-occurrence')
                fig.savefig(
                    os.path.join(outdir,plotfile),
                    transparent=True
                )
                
        return size_experiments
            
    
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

