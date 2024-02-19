import os
import configparser
from appdirs import AppDirs

appname = 'intercubos'
appauthor = 'MBG'
appdirs = AppDirs(appname, appauthor)
if not os.path.exists(appdirs.user_data_dir):
    os.makedirs(appdirs.user_data_dir)
config_file = os.path.join(
    appdirs.user_config_dir,
    'icubos.ini'
)

config = configparser.ConfigParser()
config['occurrences.gbif'] = {}
config['occurrences.gbif']['limit'] = '300'
config['occurrences.gbif']['user'] = os.environ.get('GBIF_USER')
config['occurrences.gbif']['pwd'] = os.environ.get('GBIF_PWD')
config['occurrences.gbif']['email'] = os.environ.get('GBIF_EMAIL')

if os.path.exists(config_file):
    config.read(config_file)
