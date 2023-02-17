import warnings

import numpy as np
import pandas as pd
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad

# this is saved as a record of where the distances came from, but not intended
# to be run nightly. The TIC queries take a significant amount of time.
run = False

if run:
    koifile = 'data/kepler-kois-full.csv'
    k2file = 'data/k2-candidates-table.csv'

    dfkoi = pd.read_csv(koifile)
    ukics = np.unique(dfkoi['kepid'])

    # these are KICs of confirmed planets in the Kepler field but not KOIs
    fillkics = [3526061, 4862625, 5446285, 5473556, 5807616, 5812701, 6504534,
                6762829, 7906827, 8410697, 8435766, 8572936, 9472174, 9632895,
                9837578, 10020423, 10748390, 11442793, 12351927, 12644769]
    # these are confirmed only planets with planet names KIC ##### b
    morekics = [5479689, 5951458, 7917485, 8121913, 10001893]

    ukics = np.unique(np.concatenate((ukics, fillkics, morekics)))
    # these FPs don't match anything for some reason.
    knownbad = [2436378, 2558370, 3247404, 3542574, 3556229, 5021174, 5193384,
                5481426, 7296094, 7960295, 8299947, 8432034, 8621353, 9531850,
                9692336, 9837586, 9896438, 10661778, 10743600, 10879213,
                11774383, 11825057, 11913013, 12062660, 12106934]

    dists = []
    for ii, ikoi in enumerate(ukics):
        cat = Catalogs.query_criteria(catalog='tic', KIC=ikoi)
        assert ((len(cat) == 1 and int(cat['KIC'][0]) == ikoi) or
                (ikoi in knownbad))
        if len(cat) == 0:
            dists.append(np.nan)
        else:
            dists.append(cat['d'][0])
        if (ii % 100) == 0:
            print(ii, ukics.size)

    dists = np.array(dists)

    np.savetxt('data/koi_distances.txt', np.vstack((ukics, dists)).T,
               fmt='%d  %f')

    dfk2 = pd.read_csv(k2file, low_memory=False)
    epics = []
    for iep in dfk2['epic_hostname']:
        epics.append(int(iep[4:]))
    epics = np.array(epics)
    tics = []
    for itic in dfk2['tic_id']:
        if type(itic) == str:
            assert itic[:3] == 'TIC'
            tics.append(int(itic[3:]))
        else:
            assert np.isnan(itic)
            tics.append(0)
    tics = np.array(tics)

    uepics = np.unique(epics)

    # these are EPICs of confirmed planets in the K2 fields but not K2 cands
    fillepics = [60017806, 60021410, 201498078, 201518346, 201663879, 201729655,
                 201796690, 210818897, 210897587, 211311380, 211525753,
                 211529129, 211537087, 211730267, 211914998, 211964830,
                 212628254, 212779563, 220299658, 220522664, 228724232,
                 228813918, 229426032, 245944983, 245950175, 246067459,
                 246074965, 246078672, 246151543, 246163416, 246199087,
                 246313886, 246331418, 246389858, 246393474, 246441449,
                 246471491, 246472939, 246865365, 246909566, 246911830,
                 247098361, 247267267, 247418783, 247589423, 247887989,
                 248435473, 248480671, 248545986, 248558190, 248616368,
                 248639308, 248775938, 248777106, 248782482, 249384674,
                 249391469, 249451861, 249557502, 249622103, 249624646,
                 249631677, 249801827, 249826231, 249889081, 250001426,
                 250099723, 251554286]

    # these are confirmed only planets with planet names EPIC ##### b
    moreepics = [201170410, 201427007, 201757695, 220492298, 246193072,
                 246851721, 248847494, 249893012]

    uepics = np.unique(np.concatenate((uepics, fillepics, moreepics)))

    xmatch = pd.read_csv('data/k2ticxmatch_20210831.csv')

    # these FPs don't match anything for some reason.
    knownbadepic = [251809286, 251809628]
    knownbadtic = [-251809286, -251809628]
    k2dists = []
    k2tics = []
    for ii, ik2 in enumerate(uepics):
        itics = tics[epics == ik2]
        if len(itics) == 0 or itics[0] == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                names = Simbad.query_objectids(f"EPIC {ik2}")
            itic = 0
            if names is not None:
                for iname in names:
                    if iname[0][:3] == 'TIC':
                        itic = int(iname[0][3:])
            if itic == 0:
                assert ((xmatch['epic'] == ik2).sum() == 1) or \
                       (ik2 in knownbadepic)
                if ik2 in knownbadepic:
                    itic = -1 * ik2
                else:
                    itic = int(xmatch.loc[xmatch['epic'] == ik2, 'tid'])
        else:
            itic = itics[0]
            assert np.unique(itics).size == 1
        cat = Catalogs.query_criteria(catalog='tic', ID=itic)
        assert ((len(cat) == 1 and int(cat['ID'][0]) == itic) or
                (itic in knownbadtic))
        if len(cat) == 0:
            k2dists.append(np.nan)
            k2tics.append(itic)
        else:
            k2dists.append(cat['d'][0])
            k2tics.append(itic)
        if (ii % 100) == 0:
            print(ii, uepics.size)

    k2dists = np.array(k2dists)
    k2tics = np.array(k2tics)

    outarr = np.vstack((uepics, k2tics, k2dists)).T
    np.savetxt('data/k2oi_distances.txt', outarr, fmt='%d  %d  %f')
