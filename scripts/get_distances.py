import warnings

import numpy as np
import pandas as pd
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad

from utils import load_data

# this is saved as a record of where the distances came from, but not intended
# to be run nightly. The TIC queries take a significant amount of time.
run = False

if run:
    _, dfkoi, dfk2, _, _ = load_data(updated_koi_params=False,
                                     updated_k2_params=False)

    ukics = np.unique(dfkoi['IC'])

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

    uepics = np.unique(dfk2['IC'])

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
    for ii, ik2 in enumerate(uepics):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            names = Simbad.query_objectids(f"EPIC {ik2}")
        itic = 0
        if names is not None:
            for iname in names:
                if iname[0][:3] == 'TIC':
                    assert itic == 0
                    itic = int(iname[0][3:])
        if itic == 0:
            assert ((xmatch['epic'] == ik2).sum() == 1) or ik2 in knownbadepic
            if ik2 in knownbadepic:
                itic = -1 * ik2
            else:
                itic = int(xmatch.loc[xmatch['epic'] == ik2, 'tid'])
        cat = Catalogs.query_criteria(catalog='tic', ID=itic)
        assert ((len(cat) == 1 and int(cat['ID'][0]) == itic) or
                (itic in knownbadtic))
        if len(cat) == 0:
            k2dists.append(np.nan)
        else:
            k2dists.append(cat['d'][0])
        if (ii % 100) == 0:
            print(ii, uepics.size)

    k2dists = np.array(k2dists)

    np.savetxt('data/k2oi_distances.txt', np.vstack((uepics, k2dists)).T,
               fmt='%d  %f')
