import numpy as np
import pandas as pd
from astropy.io import fits

from utils import load_data

# this is saved as a record of where the distances came from, but not intended
# to be run nightly. It requires the Kepler/K2 and Gaia cross-match files from
# https://gaia-kepler.fun, which are hundreds of MB and unreasonable to store
# as part of the repo. If those files are downloaded and put into the data
# directory however, you can set run to True and recreate these.

run = False

if run:
    _, dfkoi, dfk2, _, _ = load_data(updated_koi_params=False,
                                     updated_k2_params=False)

    gkep = fits.open('data/kepler_dr2_1arcsec.fits')

    dists = []
    ukics = np.unique(dfkoi['IC'])

    # these are KICs of confirmed planets in the Kepler field but not KOIs
    fillkics = [5446285, 8435766, 12644769, 8572936, 9837578, 6762829, 10020423,
                10020423, 4862625, 5807616, 5807616, 5473556, 12351927, 9472174,
                9632895, 8410697, 6504534, 5812701, 5446285, 10020423, 10748390,
                11442793, 7906827, 3526061]
    # these are confirmed only planets with planet names KIC ##### b
    morekics = [5951458, 10001893, 7917485, 5479689, 8121913]

    ukics = np.unique(np.concatenate((ukics, fillkics, morekics)))

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

    newkep = pd.read_csv('data/kepler_berger2020_full.txt', delimiter='&')

    kepmass = np.zeros(ukics.size) + np.nan
    kepteff = np.zeros(ukics.size) + np.nan
    keprad = np.zeros(ukics.size) + np.nan
    keplum = np.zeros(ukics.size) + np.nan
    kepdist = np.zeros(ukics.size) + np.nan

    for ii, ikoi in enumerate(ukics):
        srch = np.where(newkep['KIC'] == ikoi)[0]
        if len(srch) == 0:
            continue
        elif len(srch) > 1:
            raise Exception('Multiple KICs?')

        ind = srch[0]
        kepmass[ii] = newkep['iso_mass'][ind]
        kepteff[ii] = newkep['iso_teff'][ind]
        keprad[ii] = newkep['iso_rad'][ind]
        keplum[ii] = newkep['iso_lum'][ind]
        kepdist[ii] = newkep['iso_dis'][ind]

    out = np.vstack((ukics, kepmass, keprad, kepteff, keplum, kepdist)).T
    head = 'KIC, Mass, Rad, Teff, Log(Lum), Dist (pc)'
    np.savetxt('data/koi_params_berger2020.txt', out, header=head,
               fmt='%d  %f  %f  %f  %f  %f')

    gk2 = fits.open('data/k2_dr2_1arcsec.fits')
    k2dists = []
    uepics = np.unique(dfk2['IC'])

    # these are EPICs of confirmed planets in the K2 fields but not K2 cands
    fillepics = [246389858, 246389858, 246389858, 211529129, 248777106,
                 60021410, 211311380, 211311380, 211311380, 211311380,
                 211311380, 247887989, 247887989, 247887989, 247887989,
                 247589423, 247589423, 247589423, 228813918, 245950175,
                 245950175, 245950175, 245950175, 245950175, 246393474,
                 246393474, 220522664, 210897587, 210897587, 210897587,
                 247098361, 249622103, 249622103, 249622103, 229426032,
                 246067459, 248545986, 248545986, 248545986, 249801827,
                 249801827, 246911830, 201498078, 211964830, 211964830,
                 248435473, 248435473, 248435473, 248435473, 247267267,
                 246471491, 246471491, 246471491, 246471491, 249889081,
                 249451861, 249624646, 249624646, 247418783, 212628254,
                 246151543, 246078672, 246865365, 201518346, 246199087,
                 246199087, 246199087, 246199087, 246199087, 246199087,
                 246199087, 210818897, 210818897, 210818897, 210818897,
                 246441449, 60017806, 212779563, 249631677,
                 249384674, 249384674, 249557502, 249826231, 201663879,
                 201796690, 248480671, 248558190, 248616368, 248639308,
                 246074965, 246472939, 251554286, 211914998, 211730267,
                 211537087, 211525753, 228724232, 228724232, 201729655,
                 220299658,
                 249391469, 250001426, 250099723, 248775938, 248782482,
                 246909566, 245944983, 246163416, 246313886, 246331418,
                 246331418]

    # these are confirmed only planets with planet names EPIC ##### b
    moreepics = [249893012, 248847494, 246851721, 201170410, 201757695,
                 246193072, 201427007, 220492298]

    uepics = np.unique(np.concatenate((uepics, fillepics, moreepics)))

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

    newk2 = pd.read_csv('data/k2_hardegree-ullman2020_full.txt', skiprows=93,
                        delimiter='\t')

    k2mass = np.zeros(uepics.size) + np.nan
    k2teff = np.zeros(uepics.size) + np.nan
    k2rad = np.zeros(uepics.size) + np.nan
    k2lum = np.zeros(uepics.size) + np.nan
    k2dist = np.zeros(uepics.size) + np.nan

    for ii, ik2 in enumerate(uepics):
        srch = np.where(newk2['EPIC'] == ik2)[0]
        if len(srch) == 0:
            continue

        ind = srch[0]
        k2mass[ii] = newk2['Mstar'][ind]
        k2teff[ii] = newk2['Teff'][ind]
        k2rad[ii] = newk2['Rstar'][ind]
        lum = (k2rad[ii]**2) * ((k2teff[ii]/5772)**4)
        k2lum[ii] = np.log10(lum)
        k2dist[ii] = newk2['Dist'][ind]

    out = np.vstack((uepics, k2mass, k2rad, k2teff, k2lum, k2dist)).T
    head = 'EPIC, Mass, Rad, Teff, Log(Lum), Dist (pc)'
    np.savetxt('data/k2_params_hardegree-ullman2020.txt', out, header=head,
               fmt='%d  %f  %f  %f  %f  %f')
