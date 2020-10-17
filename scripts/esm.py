import numpy as np
import pandas as pd

# if we want to redo the order column to sort by ESM
reset_order = False

# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')

# teq = (incident flux)**0.25 * 255
# earth eq temp at 0 albedo is 278.3 K
dfpl['t_eq'] = dfpl['insol']**0.25 * 278.3

planck = 6.6261e-34
cc = 2.9979e8
boltz = 1.38065e-23

exp1 = np.exp((planck * cc)/(0.0000075 * boltz * dfpl['st_teff'])) - 1
exp2 = np.exp((planck * cc)/(0.0000075 * boltz * 1.1 * dfpl['t_eq'])) - 1

dfpl['7.5dep'] = dfpl['tran_depth_ppm'] * exp1 / exp2

dfpl['esm'] = 4.29 * dfpl['7.5dep'] * (10.**(-0.2 * dfpl['Kmag']))

toprint = (dfpl['esm'] > 1) & (dfpl['rade'] < 2)

subset = dfpl[toprint]

toprintcol = ['name', 'order', 'period', 'rade', 'tran_depth_ppm',
              'tran_dur_hr', 'masse', 'masse_est', 'semi_au', 'st_mass',
              'st_rad', 'st_teff', 'st_log_lum', 'Kmag', 'insol', 't_eq',
              '7.5dep', 'esm', 'disposition']

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

