"""Downloads candidate and confirmed planet tables from NExSci"""
from datetime import datetime

import pandas as pd

from scripts.utils import load_data

NEXSCI_API = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph' \
             '-nstedAPI'
NEW_API = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query='
# The "exoplanets" table includes all confirmed planets and hosts in the
# archive with parameters derived from a single, published reference

# All confirmed planets
'''tt = datetime.now().strftime('%H:%M:%S')
print(f"{tt} Downloading all confirmed planets from NExSci...")
df = pd.read_csv(NEW_API + 'select+*+from+pscomppars&format=csv',
                 low_memory=False)
df.to_csv('data/confirmed-planets.csv', index=False)

# full KOI table
tt = datetime.now().strftime('%H:%M:%S')
print(f'{tt} Downloading full KOI table from NExSci...')
df = pd.read_csv(NEXSCI_API + '?table=cumulative&select=*')
df.to_csv('data/kepler-kois-full.csv', index=False)

# grab all the K2 candidates (or at least the ones they have put into this
# not-quite-complete table)
tt = datetime.now().strftime('%H:%M:%S')
print(f'{tt} Downloading full K2 candidates table from NExSci...')
df = pd.read_csv(NEW_API + 'select+*+from+k2pandc&format=csv', low_memory=False)
df.to_csv('data/k2-candidates-table.csv', index=False)

# get the TOI list from ExoFOP-TESS.
tt = datetime.now().strftime('%H:%M:%S')
print(f'{tt} Downloading full TESS candidates table from ExoFOP...')
df = pd.read_csv('https://exofop.ipac.caltech.edu/tess/download_toi.php?sort'
                 '=toi&output=csv')
df.to_csv('data/tess-candidates.csv', index=False)

with open('data/last_update_time.txt', 'w') as ff:
    ff.write(str(datetime.now()))'''

# create the master data frame used in all the plots
tt = datetime.now().strftime('%H:%M:%S')
print(f'{tt} Creating master data frame and testing all planet properties.')
load_data()
tt = datetime.now().strftime('%H:%M:%S')
print(f'{tt} Done.')
"""
# all old KOI releases. Should only have to download these once
print('Downloading full KOI table from NExSci...')
df = pd.read_csv(NEXSCI_API + '?table=q1_q17_dr25_sup_koi&select=*')
df.to_csv('data/kepler-kois-q1_q17_dr25_sup.csv')


print('Downloading full KOI table from NExSci...')
df = pd.read_csv(NEXSCI_API + '?table=q1_q17_dr25_koi&select=*')
df.to_csv('data/kepler-kois-q1_q17_dr25.csv')


print('Downloading full KOI table from NExSci...')
df = pd.read_csv(NEXSCI_API + '?table=q1_q17_dr24_koi&select=*')
df.to_csv('data/kepler-kois-q1_q17_dr24.csv')


print('Downloading full KOI table from NExSci...')
df = pd.read_csv(NEXSCI_API + '?table=q1_q16_koi&select=*')
df.to_csv('data/kepler-kois-q1_q16_koi.csv')


print('Downloading full KOI table from NExSci...')
df = pd.read_csv(NEXSCI_API + '?table=q1_q12_koi&select=*')
df.to_csv('data/kepler-kois-q1_q12_koi.csv')

print('Downloading full KOI table from NExSci...')
df = pd.read_csv(NEXSCI_API + '?table=q1_q8_koi&select=*')
df.to_csv('data/kepler-kois-q1_q8_koi.csv')


print('Downloading full KOI table from NExSci...')
df = pd.read_csv(NEXSCI_API + '?table=q1_q6_koi&select=*')
df.to_csv('data/kepler-kois-q1_q6_koi.csv')
"""
