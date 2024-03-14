# Creating cube

## Dev environment

    docker run -v /c/Users/$USERNAME/repos/ixcubes:/code -p 5000:5000 -it python:3.11-slim-bookworm /bin/bash

## Getting species data for a country

https://datacatalog.worldbank.org/search/dataset/0063384/Global-Species-Database
https://github.com/CatalogueOfLife/coldp/blob/master/docs/schemaNU.png
Country codes: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2

## Catalogue of life

https://www.catalogueoflife.org/data/download

## Assign to grid

https://inbo.github.io/tutorials/tutorials/spatial_point_in_polygon/

## Code to make a cube for Guarden case study Madagascar

https://github.com/gbif/occurrence-cube

    from pygbif import occurrences, species
    import pandas as pd
    import folium
    limit = 300
    insectClass = species.name_suggest(
        'insects', limit=1
    )[0]['classKey']
    plantKingdom = species.name_suggest(
        'Plantae', limit=1
    )[0]['kingdomKey']
    insectdata = []
    plantdata = []
    #for year in range(1950,2024):
    for year in range(2020,2024):
        print(
            'Total count for', year, ':',
            occurrences.count(country = 'MG', year = year)
        )
        firstIBatch = occurrences.search(
            country = 'MG', classKey = insectClass, year = year
        )
        firstPBatch = occurrences.search(
            country = 'MG', kingdomKey = plantKingdom,
            year = year
        )
        for offset in range(0,firstIBatch['count'],limit):
            batch = occurrences.search(
                country = 'MG', classKey = insectClass,
                year = year, offset = offset, limit = limit
            )
            insectdata += batch['results']
        for offset in range(0,firstPBatch['count'],limit):
            batch = occurrences.search(
                country = 'MG', kingdomKey = plantKingdom,
                year = year, offset = offset, limit = limit
            )
            plantdata += batch['results']
        insectdata = pd.DataFrame(insectdata)
        insectdata = insectdata[
            (~insectdata.decimalLatitude.isna()&
             ~insectdata.decimalLongitude.isna())
        ]
        plantdata = pd.DataFrame(plantdata)
        plantdata = plantdata[
            (~plantdata.decimalLatitude.isna()&
             ~plantdata.decimalLongitude.isna())
        ]
    # If data already local
    # insectdata = pd.read_csv('results/insectdata_2020_2023.csv', index_col=0)
    # plantdata = pd.read_csv('results/plantdata_2020_2023.csv', index_col=0)
    lat = plantdata.decimalLatitude.mean()
    lon = plantdata.decimalLongitude.mean()
    m = folium.Map(
        location=(lat,lon), prefer_canvas=True, zoom_start=6
    )
    plant_layer = folium.FeatureGroup("plants").add_to(m)
    plantdata.T.apply(lambda x: folium.CircleMarker(
        location=[x['decimalLatitude'],x['decimalLongitude']],
        tooltip=x.species,
        popup=x.basisOfRecord,
        color="green"
        #icon=folium.Icon(color="green")
    ).add_to(plant_layer))
    insect_layer = folium.FeatureGroup("insects").add_to(m)
    insectdata.T.apply(lambda x: folium.CircleMarker(
        location=[x['decimalLatitude'],x['decimalLongitude']],
        tooltip=x.species,
        popup=x.basisOfRecord,
        color="red"
        #icon=folium.Icon(color="red")
    ).add_to(insect_layer))
    sw_lat = min(
        insectdata.decimalLatitude.min(),
        plantdata.decimalLatitude.min()
    )
    sw_lon = min(
        insectdata.decimalLongitude.min(),
        plantdata.decimalLongitude.min()
    )
    ne_lat = max(
        insectdata.decimalLatitude.max(),
        plantdata.decimalLatitude.max()
    )
    ne_lon = max(
        insectdata.decimalLongitude.max(),
        plantdata.decimalLongitude.max()
    )

## Making a grid

https://stackoverflow.com/questions/40342355/how-can-i-generate-a-regular-geographic-grid-using-python

    import itertools as it
    import shapely.geometry
    import pyproj
    
    # Set up transformers, EPSG:3857 is metric, same as EPSG:900913
    to_proxy_transformer = pyproj.Transformer.from_crs(
        'epsg:4326', 'epsg:3857')
    to_original_transformer = pyproj.Transformer.from_crs(
        'epsg:3857', 'epsg:4326'
    )
    
    # Create corners of rectangle to be transformed to a grid
    sw = shapely.geometry.Point((sw_lon, sw_lat))
    ne = shapely.geometry.Point((ne_lon, ne_lat))
    
    stepsize = 10000 # 10 km grid step size
    
    # Project corners to target projection
    transformed_sw = to_proxy_transformer.transform(sw.x, sw.y) # Transform NW point to 3857
    transformed_ne = to_proxy_transformer.transform(ne.x, ne.y) # .. same for SE
    
    # Iterate over 2D area
    gridpoints = []
    x = transformed_sw[0]
    while x < (transformed_ne[0]+stepsize):
        gridpoints.append([])
        y = transformed_sw[1]
        while y < (transformed_ne[1]+stepsize):
            p = shapely.geometry.Point(
                to_original_transformer.transform(x, y)
            ) # x == lon, y == lat  
            gridpoints[-1].append(p)
            y += stepsize
        x += stepsize
    
    grid_layer = folium.FeatureGroup("grid").add_to(m)
    for gp in it.chain(*gridpoints):
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

## Assign occurrences to grid

    import geopandas as gpd
    from shapely.geometry import Polygon
    grid_polygons = [
        shapely.geometry.box(
            gp.x, gp.y,
            gridpoints[gp_xi+1][gp_yi+1].x,
            gridpoints[gp_xi+1][gp_yi+1].y
        )
        #Logical error in Polygon use
        #Polygon([
            #(XleftOrigin, Ytop), (XrightOrigin, Ytop)
            #(gridpoints[gp_xi+1][gp_yi].x,gridpoints[gp_xi+1][gp_yi].y),
            #(gridpoints[gp_xi+1][gp_yi+1].x,gridpoints[gp_xi+1][gp_yi+1].y),
            #(XrightOrigin, Ybottom), (XleftOrigin, Ybottom)
            #(gridpoints[gp_xi+1][gp_yi].x,gridpoints[gp_xi+1][gp_yi].y),
            #(gp.x,gp.y)
        #])
        for gp_xi,gp_row in enumerate(gridpoints)
        for gp_yi,gp in enumerate(gp_row)
        if gp_xi < len(gridpoints)-1 and gp_yi < len(gp_row)-1
    ]
    grid = gpd.GeoDataFrame({'geometry':grid_polygons}, crs="EPSG:4326")
    #grid.to_file("grid.shp")

    # https://james-brennan.github.io/posts/fast_gridding_geopandas/
    insectdata = gpd.GeoDataFrame(
        insectdata,
        geometry=gpd.points_from_xy(
            insectdata.decimalLongitude,
            insectdata.decimalLatitude, crs="EPSG:4326"
        )
    ).drop(['decimalLongitude','decimalLatitude'], axis=1)
    plantdata = gpd.GeoDataFrame(
        plantdata,
        geometry=gpd.points_from_xy(
            plantdata.decimalLongitude,
            plantdata.decimalLatitude, crs="EPSG:4326"
        )
    ).drop(['decimalLongitude','decimalLatitude'], axis=1)
    insects_merged = gpd.sjoin(
        insectdata, grid, how='left', predicate='within'
    )
    #insects_merged.dissolve(by="index_right", aggfunc="count")
    grid['insect_count'] = insects_merged.index_right.value_counts()
    grid['plant_count'] = gpd.sjoin(
        plantdata, grid, how='left', predicate='within'
    ).index_right.value_counts()
    grid = grid.fillna(0)

## Grid stats

    from scipy.stats import spearmanr, pearsonr
    occgrid = grid[
        ~((grid.insect_count == 0) & (grid.plant_count == 0))
    ]
    print(spearmanr(occgrid.insect_count, occgrid.plant_count))
    
## Worldclim data

https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_bio.zip

    import zipfile
    import rasterio
    import rasterio.fill
    import rasterio.plot
    import rasterstats
    import matplotlib.pyplot as plt
    zf = zipfile.ZipFile('/code/data/wc2.1_30s_bio.zip')
    for ztfn in zf.infolist():
        ztf = f"zip+file://data/wc2.1_30s_bio.zip!{ztfn.filename}"
        dataset = rasterio.open(ztf)
        band1 = dataset.read(1)
        break
    #zonal_stats = rasterstats.zonal_stats(grid.head(), band1,
    #    affine=dataset.transform)
    # Rescale raster for plotting
    scale = 0.1
    #t = dataset.transform
    #transform = rasterio.Affine(
    #    t.a / scale, t.b, t.c, t.d, t.e / scale, t.f
    #)
    height = dataset.height * scale
    width = dataset.width * scale
    band1 = dataset.read(
        1, masked=True,
        out_shape=(int(dataset.count), int(height), int(width)),
        resampling=rasterio.enums.Resampling.cubic
    )
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / band1.shape[-1]),
        (dataset.height / band1.shape[-2]))
    # To get pixel array locations after transform scaling
    # https://rasterio.readthedocs.io/en/stable/topics/transforms.html
    transformer = rasterio.transform.AffineTransformer(
        transform
    ) # use with method ".rowcol" to get row and col for coordinate
    # Fill missing data
    band1_filled = rasterio.fill.fillnodata(
        band1, mask=dataset.read_masks(1),
        max_search_distance=10, smoothing_iterations=0
    )

    fig, ax = plt.subplots()
    rasterio.plot.show(band1, ax=ax)
    fig.savefig('/code/results/bioclim1.png')
    
    zonal_stats = gpd.GeoDataFrame(rasterstats.zonal_stats(
        grid, band1, affine=transform
    ), geometry=grid.geometry)
    rasterstats.point_query(
        gridpoints[140][60], band1, affine=transform
    )
    grid['bioclim_1'] = zonal_stats['mean']
    df=grid[['insect_count','plant_count','bioclim_1']].dropna()
    df=df[df.bioclim_1>0] # filtering masked, not ideal though
    stats.pearsonr(df.bioclim_1, df.insect_count)

## Implemented in the package `intercubos`

    from intercubos.occurrences import OcCube
    from intercubos.gridit import Grid
    mg_insects = OcCube(
        'insects', 'MG', rank='class',
        years=range(2000,2024), clip2region=True,
	verbose=True
    )
    mg_plants = OcCube(
        'Plantae', 'MG', rank='kingdom',
        years=range(2000,2024), clip2region=True,
	verbose=True
    )
    #(sw_lon, sw_lat, ne_lon, ne_lat) = mg_plants.bounds
    #grid = Grid(sw_lon, sw_lat, ne_lon, ne_lat, stepsize=5000)
    grid = Grid(*mg_plants.bounds, stepsize=5000)
    grid.assign_to_grid(mg_plants.cube, colname='total_plants')
    grid.assign_to_grid(mg_insects.cube, colname='total_insects')
    #ax = grid.plot(
    #    crs='EPSG:3857', edgecolor=None, vmax=10000, logcol=True,
    #    colorbar=True, colname='total_plants'
    #)
    #ax = grid.plot(
    #    crs='EPSG:3857', edgecolor=None, vmax=10000,
    #    color=(0,0,1), logcol=True,
    #    filename='/code/results/mg_plants_insects.png',
    #    colorbar=True, colname='total_insects', ax=ax
    #)
    ax = grid.plot_interaction('total_plants','total_insects',
        crs='EPSG:3857', edgecolor=None, vmax=10000,
        logcol=True, colorbar=True,
        filename='/code/results/mg_plants_insects.png'
    )
    
## Finding optimal grid cell size for interactions

    from intercubos.occurrences import OcCube
    from intercubos.gridit import GridScan
    mg_plants = OcCube(
        'Plantae', 'MG', rank='kingdom',
	filename='/code/data/mg_plants.pck'
        years=range(2000,2024), verbose=True
    )
    mg_insects = OcCube(
        'insects', 'MG', rank='class',
	filename='/code/data/mg_insects.pck'
        years=range(2000,2024), verbose=True
    )
    gs = GridScan(mg_plants,mg_insects)
    size_experiments = gs.scan(100000, minimumAbundance=500)
    for size in sorted(size_experiments):
        print(size,(size_experiments[size] < .05).sum(axis=1).sum())  
        print((size_experiments[size] < .05).sum(axis=0).sum())

## Peru tree data retrieval

Inspiration https://data-blog.gbif.org/post/downloading-long-species-lists-on-gbif/
https://tools.bgci.org/global_tree_search.php to download global tree
check list and Peru native tree checklist.

    from intercubos.occurrences import OcCube, make_map
    from intercubos.gridit import Grid
    import pandas as pd
    peru = OcCube(
        'Plantae', 'PE', rank='kingdom',
        years=range(2000,2024), verbose=True
    )
    peru.set_region_polygon('PE')
    peru.clip_region()
    trees = pd.read_csv(
        'data/global_tree_search_trees_1_7.csv',sep=',',
        encoding="ISO-8859-1"
    )
    perutrees = pd.read_csv(
        'data/globaltreesearch_results_Peru.csv',sep=',',
        encoding="ISO-8859-1" # or latin-1
    )
    treecounts = pd.DataFrame(peru.cube.species[
        peru.cube.species.isin(trees.TaxonName)
    ].value_counts())
    perutreecounts = peru.cube.species[
        peru.cube.species.isin(perutrees.taxon)
    ].value_counts()
    # Following should be empty but it is not!
    perutreecounts.index[~perutreecounts.index.isin(treecounts.index)]
    treecounts['native'] = treecounts.index.isin(perutreecounts.index)
    treecounts.groupby('native').describe()
    # Assign to grid
    (sw_lon, sw_lat, ne_lon, ne_lat) = peru.cube.total_bounds
    #calc_bounding_box([perus.cube]) # lat/lon reversal
    grid = Grid(sw_lon, sw_lat, ne_lon, ne_lat, stepsize=100000)
    grid.assign_to_grid(peru.cube)
    grid.assign_to_grid(
        peru.cube[peru.cube.species.isin(trees.TaxonName)],
        colname='trees'
    )
    ax = grid.plot(
        crs='EPSG:3857',filename='/code/results/peruset.png'
    )
    grid.plot(
        crs='EPSG:3857',
        filename='/code/results/perutrees.png',
        colorbar=True, colname='trees'
    ) 
    grid.time_lapse_plot(
        peru.cube, '/code/results/timelapse.gif', crs='EPSG:3857'
    )
