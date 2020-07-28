import numpy as np
from astropy.io import fits

from utils import load_data

# this is saved as a record of where the distances came from, but not intended
# to be run nightly. It requires the Kepler/K2 and Gaia cross-match files from
# https://gaia-kepler.fun, which are hundreds of MB and unreasonable to store
# as part of the repo. If those files are downloaded and put into the data
# directory however, you can set run to True and recreate these.

run = False

if run:
    dfcon, dfkoi, dfk2, dftoi = load_data()

    gkep = fits.open('data/kepler_dr2_1arcsec.fits')

    dists = []
    ukics = np.unique(dfkoi['kepid'])

    for ikoi in ukics:
        srch = np.where(gkep[1].data['kepid'] == ikoi)[0]
        if len(srch) == 0:
            dists.append(np.nan)
        elif len(srch) == 1:
            dists.append(gkep[1].data[srch[0]]['r_est'])
        elif len(srch) > 1:
            ms = gkep[1].data[srch]['phot_g_mean_mag']
            km = gkep[1].data[srch[0]]['kepmag']
            ind = np.argmin(np.abs(km - ms))
            dists.append(gkep[1].data[srch[ind]]['r_est'])

    dists = np.array(dists)

    np.savetxt('data/koi_distances.txt', np.vstack((ukics, dists)).T,
               fmt='%d  %f')
    gkep.close()

    gk2 = fits.open('data/k2_dr2_1arcsec.fits')
    k2dists = []
    uepics = np.unique(dfk2['epic'])

    for ik2 in uepics:
        srch = np.where(gk2[1].data['epic_number'] == ik2)[0]
        if len(srch) == 0:
            k2dists.append(np.nan)
        elif len(srch) == 1:
            k2dists.append(gk2[1].data[srch[0]]['r_est'])
        elif len(srch) > 1:
            ms = gk2[1].data[srch]['phot_g_mean_mag']
            km = gk2[1].data[srch[0]]['k2_kepmag']
            ind = np.argmin(np.abs(km - ms))
            k2dists.append(gk2[1].data[srch[ind]]['r_est'])

    k2dists = np.array(k2dists)

    np.savetxt('data/k2oi_distances.txt', np.vstack((uepics, k2dists)).T,
               fmt='%d  %f')

    gk2.close()
