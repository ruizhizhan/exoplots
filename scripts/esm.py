import numpy as np
import pandas as pd

from utils import get_equilibrium_temperature, get_esm

# if we want to redo the order column to sort by ESM
reset_order = False

# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')


# calculate the values we want
albedo = 0.
# fraction of planet radiating (1 for fast spin, 0.5 tidally locked)
redist = 1.

dfpl['t_eq'] = get_equilibrium_temperature(dfpl, albedo=albedo,
                                           radiative_fraction=redist)

esmscale = 4.29
refwav = 7.5

esm = get_esm(dfpl, wavelength_micron=refwav, scale=esmscale, albedo=albedo,
              radiative_fraction=redist)
dfpl['esm'] = esm

toprint = (dfpl['esm'] > 1) & (dfpl['rade'] < 2)
print('here 1')
subset = dfpl[toprint]
print('here 2')

toprintcol = ['name', 'order', 'period', 'rade', 'tran_depth_ppm',
              'tran_dur_hr', 'masse', 'masse_est', 'semi_au', 'st_mass',
              'st_rad', 'st_teff', 'st_log_lum', 'Kmag', 'insol', 't_eq',
              'esm', 'disposition']

old = pd.read_csv('data/rpe.txt')
nextind = old['order'].max() + 1

dfpl['order'] = -1

for ind, row in subset.iterrows():
    srch = old['name'] == row['name']
    assert srch.sum() <= 1
    if srch.sum() == 1:
        dfpl.at[ind, 'order'] = old.loc[srch, 'order'].values[0]
    else:
        dfpl.at[ind, 'order'] = nextind
        nextind += 1

subset = dfpl[toprint]

assert subset['order'].min() == 0
assert np.unique(subset['order']).size == subset['order'].size

# reset the order flag
if reset_order:
    subset.sort_values('esm', ascending=False, inplace=True)
    subset['order'] = np.arange(len(subset))

subset.sort_values('order', ascending=True, inplace=True)
subset.to_csv('data/rpe.txt', index=False, columns=toprintcol,
              float_format='%.5f')
