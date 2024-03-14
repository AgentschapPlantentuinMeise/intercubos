import os
import time
import pickle
import datetime
import logging
import zipfile
from pygbif import occurrences, species
import pandas as pd
import geopandas as gpd
import folium
from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.overpass import Overpass
from shapely.geometry import MultiPolygon
from intercubos.config import appdirs, config

gbif_config = config['occurrences.gbif']

class OcCube:
    def __init__(
            self, taxonName, regionName, years, rank='species',
            clip2region=False, filename=None, verbose=False):
        if filename and os.path.exists(filename):
            with open(filename,'rb') as f:
                self.cube = pickle.load(f)
        else:
            self.cube = get_occurrences(
                taxonName, regionName, rank=rank, years=years, verbose=verbose
            )
            if filename:
                with open(filename,'wb') as f:
                    pickle.dump(self.cube,f)
        # Clipping after optionally saved file
        if clip2region:
            self.set_region_polygon(regionName)
            self.clip_region()

    def set_region_polygon(self, regionCode, admin_level='lowest'):
        nominatim = Nominatim()
        overpass = Overpass()
        nr = nominatim.query(regionCode)
        ovr = overpass.query(
            f'rel["name"="{nr.toJSON()[0]["name"]}"]; out body geom;'
        ) # to filter also on admin_level rel["admin_level"=]
        if admin_level == 'lowest':
            rel = sorted(
                ovr.relations(),
                key=lambda x: int(x.tags().get(
                    'admin_level', 100
                )))[0]
        elif admin_level == 'highest':
            rel = sorted(
                ovr.relations(),
                key=lambda x: int(x.tags().get(
                    'admin_level', -100
                )), reverse=True)[0]
        elif isinstance(admin_level, int):
            rel = next(filter(
                   lambda x: int(x.tags().get('admin_level',-100)) == admin_level,
                   ovr.relations()))
        else: rel = ovr.relations()[0]
        self.regionPolygon = MultiPolygon([rel.geometry()['coordinates']])

    def clip_region(self):
        logging.info('Cube size before clipping: %s', len(self.cube))
        self.cube = self.cube.clip(self.regionPolygon)
        logging.info('Cube size after clipping: %s', len(self.cube))

    @property
    def bounds(self):
        return self.cube.total_bounds
        
def get_occurrences(
    taxonName, regionName, regionPolygon=None,
    admin_level = 'lowest', # lowest, highest, int or None
    rank='species', # species, class, kingdom
    # Filtering parameters
    years = range(1950,datetime.datetime.now().year),
    limit = gbif_config.getint('limit'), verbose=False
):
    # Taxon
    nameKey = species.name_suggest(
       taxonName, limit=1
    )[0][f"{rank}Key"]
    nameConfig = {f"{rank}Key": nameKey}
    data = []
    total_taxon_counts = {}
    total_region_counts = {}

    # Two strategies to get data: download if configured, or year by year API calls
    if config['occurrences.gbif']['user']:
        if not config['occurrences.gbif']['pwd']:
            from getpass import getpass
            config['occurrences.gbif']['pwd'] = getpass('GBIF password: ')
        download = True
        data_keys = []
    else: download = False
    for year in years:
       total_region_counts[year] = occurrences.count(country = regionName, year = year)
       total_taxon_counts[year] = occurrences.search(
           country = regionName, year = year, limit=0, **nameConfig
       )['count'] # limit == 0 as records not required
       if verbose:
           logging.info(
               'Total count for %s: %s, %s count: %s',
               year, total_region_counts[year],
               taxonName, total_taxon_counts[year]
           )
       if download:
           query = {"type":"and",
               'predicates':[
                   {'type': 'equals', 'key':'YEAR', 'value': year},
                   {'type': 'equals', 'key':'TAXON_KEY', 'value': nameConfig[f"{rank}Key"]},
                   {'type': 'equals', 'key':'COUNTRY', 'value': regionName}
           ]}
           # Check if not too many concurrent download preparations (max 3)
           while sum([
                   r['status']=='RUNNING'
                   for r in                    occurrences.download_list()['results']
           ]) >= 3: time.sleep(30)
           dld = occurrences.download(query)
           data_keys.append(dld[0])
       else:
           for offset in range(0,total_taxon_counts[year],limit):
               batch = occurrences.search(
                   country = regionName, **nameConfig,
                   year = year, offset = offset, limit = limit
               )
               data += batch['results']
    if download:
        # Save download keys
        with open(os.path.join(appdirs.user_data_dir, f"{taxonName}_{regionName}.keys"), 'wt') as fout:
            fout.write('\n'.join(data_keys))
        for download_key in data_keys:
            if os.path.exists(
                    os.path.join(appdirs.user_data_dir, download_key+'.zip')
            ): continue
            metadld = occurrences.download_meta(download_key)
            if metadld['status'] != 'SUCCEEDED':
                logging.info('Waiting for GBIF job completion')
                while (metadld := occurrences.download_meta(download_key))['status'] == 'RUNNING':
                    time.sleep(60)
            if metadld['status'] != 'SUCCEEDED':
                raise RuntimeError('GBIF download issue')
            occurrences.download_get(
                download_key,
                path=appdirs.user_data_dir
            )
        # Load data:
        data = pd.DataFrame()
        for download_key in data_keys:
            zf = zipfile.ZipFile(
                os.path.join(appdirs.user_data_dir, download_key+'.zip')
            )
            zcsvf = zf.open(download_key+'.csv')
            data = pd.concat(
                [
                    data,
                    pd.read_csv(zcsvf, sep='\t', on_bad_lines='warn')
                ], ignore_index=True
            )
        
    else: data = pd.DataFrame(data)
    data = data[
        (~data.decimalLatitude.isna()&
         ~data.decimalLongitude.isna())
    ]
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(
            data.decimalLongitude, data.decimalLatitude, crs="EPSG:4326"
        )
    ).drop(['decimalLongitude','decimalLatitude'], axis=1)

def calc_bounding_box(data_layers):
    sw_lat = min([
        data.decimalLatitude.min()
        for data in data_layers
    ])
    sw_lon = min([
        data.decimalLongitude.min()
        for data in data_layers
    ])
    ne_lat = max([
        data.decimalLatitude.max()
        for data in data_layers
    ])
    ne_lon = max([
        data.decimalLongitude.max()
        for data in data_layers
    ])
    return (sw_lat, sw_lon, ne_lat, ne_lon)

def make_map(data_layers):
    lat = data_layers[list(data_layers)[0]][0].decimalLatitude.mean()
    lon = data_layers[list(data_layers)[0]][0].decimalLongitude.mean()
    m = folium.Map(
        location=(lat,lon), prefer_canvas=True, zoom_start=6
    )
    for name, (data, color) in data_layers.items():
        layer = folium.FeatureGroup(name).add_to(m)
        data.T.apply(lambda x: folium.CircleMarker(
            location=[x['decimalLatitude'],x['decimalLongitude']],
            tooltip=x.species,
            popup=x.basisOfRecord,
            color=color
        ).add_to(layer))
    return m

