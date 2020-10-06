import numpy as np
import pandas as pd


# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')

# teq = (incident flux)**0.25 * 255
dfpl['t_eq'] = dfpl['insol']**0.25 * 255

planck = 6.6261e-34
cc = 2.9979e8
boltz = 1.38065e-23

exp1 = np.exp((planck * cc)/(0.0000075 * boltz * dfpl['st_teff'])) - 1
exp2 = np.exp((planck * cc)/(0.0000075 * boltz * 1.1 * dfpl['t_eq'])) - 1

dfpl['7.5dep'] = dfpl['tran_depth_ppm'] * exp1 / exp2

dfpl['esm'] = 4.29 * dfpl['7.5dep'] * (10.**(-0.2 * dfpl['Kmag']))

toprint = (dfpl['esm'] > 1) & (dfpl['rade'] < 2)

subset = dfpl[toprint]

toprint = ['name', 'period', 'rade', 
            'tran_depth_ppm', 'tran_dur_hr', 'semi_au',
            'insol', 't_eq', 'Kmag', '7.5dep', 'esm',
            'st_mass', 'st_rad', 'st_teff',
            'st_log_lum', 'masse', 'masse_est', 'disposition']

subset.sort_values('esm', ascending=False, inplace=True)
subset.to_csv('data/rpe.txt', index=False, columns=toprint, 
              float_format='%.5f')

