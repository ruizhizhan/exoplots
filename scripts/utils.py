"""
Utility functions needed for every figure. Load and process the data in a
uniform way as well as run tests to ensure sensible results.
"""


def get_update_time():
    """
    Return a datetime object representing the last time all the data files
    were generated.

    Returns
    -------
    datetime.datetime

    """
    import datetime
    dateloc = 'data/last_update_time.txt'
    with open(dateloc, 'r') as ff:
        lines = ff.readlines()
    return datetime.datetime.strptime(lines[0], '%Y-%m-%d %H:%M:%S.%f')


def get_new_koi_params(dfcon, dfkoi):
    import numpy as np
    import warnings

    kk, mm, rr, tt, ll, dd = np.loadtxt('data/koi_params_berger2020.txt',
                                        unpack=True)
    kk = kk.astype(int)

    koicon = dfkoi['koi_disposition'] == 'Confirmed'
    koican = dfkoi['koi_disposition'] == 'Candidate'

    # all these just have very slight differences in period (< 0.2 days)
    # which my conservative selection rejects but are good enough
    # (KOI-4441 and 5475 was a KOI at half the period of the confirmed planet
    # and 5568 a KOI at 1/3 the confirmed period)
    excluded = ['KOI-806.01', 'KOI-806.03', 'KOI-142.01', 'KOI-1274.01',
                'KOI-1474.01', 'KOI-1599.01', 'KOI-377.01', 'KOI-377.02',
                'KOI-4441.01', 'KOI-5568.01', 'KOI-5475.01', 'KOI-5622.01',
                'KOI-277.02', 'KOI-523.02', 'KOI-1783.02']
    # what the name is in the confirmed planets table
    real = ['Kepler-30 d', 'Kepler-30 b', 'KOI-142 b', 'Kepler-421 b',
            'Kepler-419 b', 'KOI-1599.01', 'Kepler-9 b', 'Kepler-9 c',
            'Kepler-1604 b', 'Kepler-1633 b', 'Kepler-1632 b', 'Kepler-1635 b',
            'Kepler-36 b', 'Kepler-177 b', 'KOI-1783.02']

    cononly = dfcon['pl_kepflag'].values * 1
    cononly = cononly.astype(bool)

    # match the confirmed KOIs to the appropriate confirmed planets
    for index, icon in dfkoi[koicon].iterrows():
        res = np.where((np.abs(dfcon['ra'] - icon['ra']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - icon['dec']) < 1. / 60) &
                       (np.abs(
                           dfcon['pl_orbper'] - icon['koi_period']) < 1 / 60))
        res = res[0]
        if len(res) != 1:
            # special cases I know about that we can match up manually
            assert icon['kepoi_name'] in excluded
            rname = real[excluded.index(icon['kepoi_name'])]
            res = np.where(dfcon['pl_name'] == rname)[0]
            assert len(res) == 1
        cononly[res[0]] = False
        # make sure both tables have the new parameters
        fd = np.where(kk == icon['kepid'])[0]
        if len(fd) != 1:
            warnings.warn(f"Can't find parameters for KIC {icon['kepid']}")
        elif ~np.isfinite(mm[fd[0]]):
            continue
        else:
            fd = fd[0]
            dfkoi.at[index, 'koi_smass'] = mm[fd]
            oldrad = dfkoi.at[index, 'koi_srad'] * 1
            dfkoi.at[index, 'koi_srad'] = rr[fd]
            dfkoi.at[index, 'koi_steff'] = tt[fd]
            dfkoi.at[index, 'log_lum'] = ll[fd]
            dfkoi.at[index, 'distance_pc'] = dd[fd]
            srat = rr[fd] / oldrad
            if np.isfinite(srat):
                dfkoi.at[index, 'koi_prad'] *= srat
                dfkoi.at[index, 'koi_pradj'] *= srat

            res = res[0]
            dfcon.at[res, 'st_mass'] = mm[fd]
            oldrad = dfcon.at[res, 'st_rad'] * 1
            dfcon.at[res, 'st_rad'] = rr[fd]
            dfcon.at[res, 'st_teff'] = tt[fd]
            dfcon.at[res, 'st_lum'] = ll[fd]
            dfcon.at[res, 'distance_pc'] = dd[fd]
            srat = rr[fd] / oldrad
            if np.isfinite(srat):
                dfcon.at[res, 'pl_rade'] *= srat
                dfcon.at[res, 'pl_radj'] *= srat

    # make sure all candidate KOIs have the new parameters
    for index, ican in dfkoi[koican].iterrows():
        res = np.where((np.abs(dfcon['ra'] - ican['ra']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - ican['dec']) < 1. / 60) &
                       (np.abs(
                           dfcon['pl_orbper'] - ican['koi_period']) < 1 / 60))
        res = res[0]
        assert len(res) == 0

        fd = np.where(kk == ican['kepid'])[0]
        if len(fd) != 1:
            warnings.warn(f"Can't find parameters for KIC {ican['kepid']}")
        elif ~np.isfinite(mm[fd[0]]):
            continue
        else:
            fd = fd[0]
            dfkoi.at[index, 'koi_smass'] = mm[fd]
            oldrad = dfkoi.at[index, 'koi_srad'] * 1
            dfkoi.at[index, 'koi_srad'] = rr[fd]
            dfkoi.at[index, 'koi_steff'] = tt[fd]
            dfkoi.at[index, 'log_lum'] = ll[fd]
            dfkoi.at[index, 'distance_pc'] = dd[fd]
            srat = rr[fd] / oldrad
            if np.isfinite(srat):
                dfkoi.at[index, 'koi_prad'] *= srat
                dfkoi.at[index, 'koi_pradj'] *= srat

    missing = ['KOI-142 c', 'Kepler-78 b', 'Kepler-16 b', 'Kepler-34 b',
               'Kepler-35 b', 'Kepler-38 b', 'Kepler-47 b', 'Kepler-47 c',
               'PH1 b', 'KOI-55 b', 'KOI-55 c', 'Kepler-1647 b', 'Kepler-413 b',
               '2MASS J19383260+4603591 b', 'Kepler-453 b', 'Kepler-1654 b',
               'Kepler-1661 b', 'Kepler-448 c', 'Kepler-88 d', 'Kepler-47 d',
               'HAT-P-11 c', 'Kepler-90 i']
    fillkics = [5446285, 8435766, 12644769, 8572936, 9837578, 6762829, 10020423,
                10020423, 4862625, 5807616, 5807616, 5473556, 12351927, 9472174,
                9632895, 8410697, 6504534, 5812701, 5446285, 10020423, 10748390,
                11442793]

    # for Kepler planets only on the confirmed list, try to find their KIC
    # from other KOIs in the system
    for index, icon in dfcon[cononly].iterrows():
        # include the trailing space so we match Kepler 90 but not Kepler 900
        iname = icon['pl_name'][:-1]
        # easy case where the planet name is KIC #### b
        if iname[:3] == 'KIC':
            iskic = int(iname[4:])
        else:
            # get the host name of all similar planets
            isin = dfcon['pl_name'].str.contains(iname, regex=False)
            matches = dfcon['pl_hostname'][isin]

            # look for those host names in the KOI list
            haskoi = np.zeros(dfkoi['kepler_name'].size).astype(bool)
            for ik in matches:
                tmp = dfkoi['kepler_name'].str.contains(ik + ' ')
                tmp.replace(np.nan, False, inplace=True)
                haskoi |= tmp

            # make sure they all agree on the KIC
            if haskoi.sum() > 0:
                iskic = np.unique(dfkoi['kepid'][haskoi])
                assert iskic.size == 1
            # we can't find it, so it's one of these special cases
            else:
                assert icon['pl_name'] in missing
                iskic = fillkics[missing.index(icon['pl_name'])]

        fd = np.where(kk == iskic)[0]
        if len(fd) != 1:
            warnings.warn(f"Can't find parameters for planet {icon['pl_name']}")
        elif ~np.isfinite(mm[fd[0]]):
            continue
        else:
            fd = fd[0]
            dfcon.at[index, 'st_mass'] = mm[fd]
            oldrad = dfcon.at[index, 'st_rad'] * 1
            dfcon.at[index, 'st_rad'] = rr[fd]
            dfcon.at[index, 'st_teff'] = tt[fd]
            dfcon.at[index, 'st_lum'] = ll[fd]
            dfcon.at[index, 'distance_pc'] = dd[fd]
            srat = rr[fd] / oldrad
            if np.isfinite(srat):
                dfcon.at[index, 'pl_rade'] *= srat
                dfcon.at[index, 'pl_radj'] *= srat


def get_new_k2_params(dfcon, dfk2):
    import numpy as np
    import warnings

    ee, mm, rr, tt, ll, dd = np.loadtxt(
        'data/k2_params_hardegree-ullman2020.txt',
        unpack=True)
    ee = ee.astype(int)

    k2con = dfk2['k2c_disp'] == 'Confirmed'
    k2can = dfk2['k2c_disp'] == 'Candidate'

    # all K2 confirmed planets are already in the confirmed planets table
    notfound = ~np.in1d(dfk2['pl_name'][k2con], dfcon['pl_name'])
    assert notfound.sum() == 0
    assert dfk2['pl_name'][k2con].isna().sum() == 0

    # anything with a planet name in the K2 table but still a candidate hasn't
    # already shown up in the confirmed planets table
    hasname = ~dfk2['pl_name'][k2can].isna()
    assert np.in1d(dfk2['pl_name'][k2can][hasname], dfcon['pl_name']).sum() == 0

    cononly = dfcon['pl_k2flag'].values * 1
    cononly = cononly.astype(bool)

    # match the confirmed K2 candidates to the appropriate confirmed planets
    for index, icon in dfk2[k2con].iterrows():
        res = np.where(icon['pl_name'] == dfcon['pl_name'])
        res = res[0]
        assert len(res) == 1
        cononly[res[0]] = False
        # make sure both tables have the new parameters
        fd = np.where(ee == icon['epic'])[0]
        if len(fd) != 1:
            warnings.warn(f"Can't find parameters for EPIC {icon['epic']}")
        elif ~np.isfinite(mm[fd[0]]):
            continue
        else:
            fd = fd[0]
            dfk2.at[index, 'st_mass'] = mm[fd]
            oldrad = dfk2.at[index, 'st_rad'] * 1
            dfk2.at[index, 'st_rad'] = rr[fd]
            dfk2.at[index, 'st_teff'] = tt[fd]
            dfk2.at[index, 'log_lum'] = ll[fd]
            dfk2.at[index, 'distance_pc'] = dd[fd]
            srat = rr[fd] / oldrad
            if np.isfinite(srat):
                dfk2.at[index, 'pl_rade'] *= srat
                dfk2.at[index, 'pl_radj'] *= srat

            res = res[0]
            dfcon.at[res, 'st_mass'] = mm[fd]
            oldrad = dfcon.at[res, 'st_rad'] * 1
            dfcon.at[res, 'st_rad'] = rr[fd]
            dfcon.at[res, 'st_teff'] = tt[fd]
            dfcon.at[res, 'st_lum'] = ll[fd]
            dfcon.at[res, 'distance_pc'] = dd[fd]
            srat = rr[fd] / oldrad
            if np.isfinite(srat):
                dfcon.at[res, 'pl_rade'] *= srat
                dfcon.at[res, 'pl_radj'] *= srat

    # make sure all candidate K2 planets have the new parameters
    for index, ican in dfk2[k2can].iterrows():
        res = np.where((np.abs(dfcon['ra'] - ican['ra']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - ican['dec']) < 1. / 60) &
                       (np.abs(
                           dfcon['pl_orbper'] - ican['pl_orbper']) < 1 / 60))
        res = res[0]
        assert len(res) == 0

        fd = np.where(ee == ican['epic'])[0]
        if len(fd) != 1:
            warnings.warn(f"Can't find parameters for EPIC {ican['epic']}")
        elif ~np.isfinite(mm[fd[0]]):
            continue
        else:
            fd = fd[0]
            dfk2.at[index, 'st_mass'] = mm[fd]
            oldrad = dfk2.at[index, 'st_rad'] * 1
            dfk2.at[index, 'st_rad'] = rr[fd]
            dfk2.at[index, 'st_teff'] = tt[fd]
            dfk2.at[index, 'log_lum'] = ll[fd]
            dfk2.at[index, 'distance_pc'] = dd[fd]
            srat = rr[fd] / oldrad
            if np.isfinite(srat):
                dfk2.at[index, 'pl_rade'] *= srat
                dfk2.at[index, 'pl_radj'] *= srat

    missing = ['GJ 9827 b', 'GJ 9827 c', 'GJ 9827 d', 'HD 72490 b',
               'HD 89345 b', 'HIP 116454 b', 'HIP 41378 b', 'HIP 41378 c',
               'HIP 41378 d', 'HIP 41378 e', 'HIP 41378 f', 'K2-133 b',
               'K2-133 c', 'K2-133 d', 'K2-133 e', 'K2-136 b', 'K2-136 c',
               'K2-136 d', 'K2-137 b', 'K2-138 b', 'K2-138 c', 'K2-138 d',
               'K2-138 e', 'K2-138 f', 'K2-141 b', 'K2-141 c', 'K2-149 b',
               'K2-155 b', 'K2-155 c', 'K2-155 d', 'K2-232 b', 'K2-233 b',
               'K2-233 c', 'K2-233 d', 'K2-237 b', 'K2-238 b', 'K2-239 b',
               'K2-239 c', 'K2-239 d', 'K2-240 b', 'K2-240 c', 'K2-260 b',
               'K2-261 b', 'K2-264 b', 'K2-264 c', 'K2-266 b', 'K2-266 c',
               'K2-266 d', 'K2-266 e', 'K2-284 b', 'K2-285 b', 'K2-285 c',
               'K2-285 d', 'K2-285 e', 'K2-286 b', 'K2-287 b', 'K2-290 b',
               'K2-290 c', 'K2-291 b', 'K2-292 b', 'K2-293 b', 'K2-294 b',
               'K2-308 b', 'Ross 128 b', 'TRAPPIST-1 b', 'TRAPPIST-1 c',
               'TRAPPIST-1 d', 'TRAPPIST-1 e', 'TRAPPIST-1 f', 'TRAPPIST-1 g',
               'TRAPPIST-1 h', 'V1298 Tau b', 'V1298 Tau c', 'V1298 Tau d',
               'V1298 Tau e', 'WASP-151 b', 'WASP-28 b', 'Wolf 503 b']
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
                 246441449, 60017806, 212779563]

    # for K2 planets only on the confirmed list, try to find their EPIC
    # from other KOIs in the system
    for index, icon in dfcon[cononly].iterrows():
        # include the trailing space so we match K2 23 but not K2 230
        iname = icon['pl_name'][:-1]
        # easy case where the planet name is EPIC #### b
        if iname[:4] == 'EPIC':
            isepic = int(iname[4:])
        else:
            # get the host name of all similar planets
            isin = dfcon['pl_name'].str.contains(iname, regex=False)
            matches = dfcon['pl_hostname'][isin]

            # look for those host names in the K2 candidate list
            haskoi = np.zeros(dfk2['pl_name'].size).astype(bool)
            for ik in matches:
                tmp = dfk2['pl_name'].str.contains(ik + ' ')
                tmp.replace(np.nan, False, inplace=True)
                haskoi |= tmp

            # make sure they all agree on the KIC
            if haskoi.sum() > 0:
                isepic = np.unique(dfk2['epic'][haskoi])
                assert isepic.size == 1
            # we can't find it, so it's one of these special cases
            else:
                assert icon['pl_name'] in missing
                isepic = fillepics[missing.index(icon['pl_name'])]

        fd = np.where(ee == isepic)[0]
        if len(fd) != 1:
            warnings.warn(f"Can't find parameters for planet {icon['pl_name']}")
        elif ~np.isfinite(mm[fd[0]]):
            continue
        else:
            fd = fd[0]
            dfcon.at[index, 'st_mass'] = mm[fd]
            oldrad = dfcon.at[index, 'st_rad'] * 1
            dfcon.at[index, 'st_rad'] = rr[fd]
            dfcon.at[index, 'st_teff'] = tt[fd]
            dfcon.at[index, 'st_lum'] = ll[fd]
            dfcon.at[index, 'distance_pc'] = dd[fd]
            srat = rr[fd] / oldrad
            if np.isfinite(srat):
                dfcon.at[index, 'pl_rade'] *= srat
                dfcon.at[index, 'pl_radj'] *= srat


def set_discovery_year(dfcon, dfkoi, dfk2, dftoi):
    """
    Simultaneously test the data to make sure we're counting each planet
    exactly once and also set up the necessary links between planets on the
    different tables to get year of discovery instead of year of confirmation
    that is listed in the confirmed planets table.
    """
    from glob import glob
    import numpy as np
    import pandas as pd

    # set up the appropriate columns
    dfcon['year_disc'] = dfcon['pl_disc'] * 1
    dfk2['year_disc'] = dfk2['year'] * 1
    dfkoi['year_disc'] = 1950
    dftoi['year_disc'] = dftoi['year'] * 1

    # check that we're including all K2 planets, but only counting them once
    k2con = dfk2['k2c_disp'] == 'Confirmed'
    k2can = dfk2['k2c_disp'] == 'Candidate'

    # all K2 confirmed planets are already in the confirmed planets table
    notfound = ~np.in1d(dfk2['pl_name'][k2con], dfcon['pl_name'])
    assert notfound.sum() == 0

    # anything with a planet name in the K2 table but still a candidate hasn't
    # already shown up in the confirmed planets table
    hasname = ~dfk2['pl_name'][k2can].isna()
    assert np.in1d(dfk2['pl_name'][k2can][hasname], dfcon['pl_name']).sum() == 0

    # also test explicitly by RA/Dec/Period

    # all these just have very slight differences in period (< 0.2 days) or are
    # off by factors of 2 since different groups found different things for the
    # same planet
    k2exclude = ['EPIC 201505350.01', 'EPIC 201596316.01', 'EPIC 201629650.01',
                 'EPIC 201637175.01', 'EPIC 201647718.01', 'EPIC 203771098.01',
                 'EPIC 206348688.02', 'EPIC 210968143.01', 'EPIC 212394689.02',
                 'EPIC 212672300.01']

    # XXX: until this is fixed (the Kruse and Heller .03 are different planets)
    k2exclude.append('EPIC 201497682.03')

    # make sure all confirmed K2 planets are in the confirmed table exactly once
    for index, icon in dfk2[k2con].iterrows():
        res = np.where((np.abs(dfcon['ra'] - icon['ra']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - icon['dec']) < 1. / 60) &
                       (np.abs(
                           dfcon['pl_orbper'] - icon['pl_orbper']) < 1. / 60))
        res = res[0]
        if len(res) != 1:
            # special cases I know about that we can ignore
            assert icon['epic_candname'] in k2exclude
            # for now set its discovery year to be late
            icon['year_disc'] = 2050
        else:
            found = dfk2['epic_candname'] == icon['epic_candname']
            k2yr = dfk2['year'][found].min()
            conyr = dfcon.at[res[0], 'pl_disc']
            # set the confirmed planet and this candidate to have the same
            # discovery year
            dfcon.at[res[0], 'year_disc'] = min(k2yr, conyr)
            dfk2.at[index, 'year_disc'] = min(k2yr, conyr)

    # deal with the ones we skipped
    # in theory this shouldn't matter because we'll only be drawing from the
    # confirmed planets table anyway for these
    for ival in k2exclude:
        srch = np.where(dfk2['epic_candname'] == ival)[0]
        k2yr = dfk2['year_disc'][dfk2['epic_candname'] == ival].min()
        for isrch in srch:
            dfk2.at[isrch, 'year_disc'] = k2yr
    assert dfk2['year_disc'].max() < 2040

    # make sure all candidate K2 planets aren't in the confirmed table
    for index, ican in dfk2[k2can].iterrows():
        res = np.where((np.abs(dfcon['ra'] - ican['ra']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - ican['dec']) < 1. / 60) &
                       (np.abs(
                           dfcon['pl_orbper'] - ican['pl_orbper']) < 1. / 60))
        res = res[0]
        assert len(res) == 0

        found = dfk2['epic_candname'] == ican['epic_candname']
        k2yr = dfk2['year'][found].min()
        dfk2.at[index, 'year_disc'] = k2yr

    # first go through and assign KOIs the year they first showed up in a KOI
    # catalog. We're assuming in this process that a particular KOI number will
    # always refer to the same planet.
    dfkoi['koi_year'] = 1990

    # these first 2 KOI tables aren't archived on Exoplanet Archive
    earlykois = ['data/koi1.txt', 'data/koi2.txt']
    allkois = glob('data/kepler-kois-q*')
    allkois.sort()
    # year the KOI tables were published
    koiyears = [2013, 2014, 2015, 2015, 2016, 2018]

    # load the two early KOI tables. Just use KOI names
    k1 = np.loadtxt(earlykois[0], dtype='<U12', usecols=(0,))
    for ii in np.arange(k1.size):
        k1[ii] = 'KOI-' + k1[ii]

    k2 = np.loadtxt(earlykois[1], dtype='<U12', usecols=(0,), skiprows=73)
    for ii in np.arange(k2.size):
        k2[ii] = 'KOI-' + k2[ii]

    # load the archived KOI tables
    dfs = []
    for ifile in allkois:
        df = pd.read_csv(ifile)
        df['kepoi_name'].replace(to_replace='K0+', value='KOI-',
                                 regex=True, inplace=True)
        dfs.append(df)

    # find the first time a particular KOI number is mentioned and set its year
    for index, row in dfkoi.iterrows():
        ikoi = row['kepoi_name']
        if ikoi in k1 or ikoi in k2:
            dfkoi.at[index, 'koi_year'] = 2011
            continue
        for ii, df in enumerate(dfs):
            if ikoi in df['kepoi_name'].values:
                dfkoi.at[index, 'koi_year'] = koiyears[ii]
                break

    assert dfkoi['koi_year'].min() == 2011 and dfkoi['koi_year'].max() == 2018

    dfkoi['year_disc'] = dfkoi['koi_year'] * 1

    # there's not an easy way to tie confirmed planets in the KOI table to
    # entries in the confirmed planets table. instead match by RA/Dec/Period
    koicon = dfkoi['koi_disposition'] == 'Confirmed'
    koican = dfkoi['koi_disposition'] == 'Candidate'

    # all these just have very slight differences in period (< 0.2 days)
    # which my conservative selection rejects but are good enough
    # (KOI-4441 and 5475 was a KOI at half the period of the confirmed planet
    # and 5568 a KOI at 1/3 the confirmed period)
    excluded = ['KOI-806.01', 'KOI-806.03', 'KOI-142.01', 'KOI-1274.01',
                'KOI-1474.01', 'KOI-1599.01', 'KOI-377.01', 'KOI-377.02',
                'KOI-4441.01', 'KOI-5568.01', 'KOI-5475.01', 'KOI-5622.01',
                'KOI-277.02', 'KOI-523.02', 'KOI-1783.02']
    # what the name is in the confirmed planets table
    real = ['Kepler-30 d', 'Kepler-30 b', 'KOI-142 b', 'Kepler-421 b',
            'Kepler-419 b', 'KOI-1599.01', 'Kepler-9 b', 'Kepler-9 c',
            'Kepler-1604 b', 'Kepler-1633 b', 'Kepler-1632 b', 'Kepler-1635 b',
            'Kepler-36 b', 'Kepler-177 b', 'KOI-1783.02']

    # make sure all confirmed KOIs are in the confirmed table exactly once
    for index, icon in dfkoi[koicon].iterrows():
        res = np.where((np.abs(dfcon['ra'] - icon['ra']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - icon['dec']) < 1. / 60) &
                       (np.abs(
                           dfcon['pl_orbper'] - icon['koi_period']) < 1 / 60))
        res = res[0]
        if len(res) != 1:
            # special cases I know about that we can match up manually
            assert icon['kepoi_name'] in excluded
            rname = real[excluded.index(icon['kepoi_name'])]
            res = np.where(dfcon['pl_name'] == rname)[0]
            assert len(res) == 1
        # make sure both tables have the same discovery year
        koiyr = icon['koi_year']
        conyr = dfcon.at[res[0], 'pl_disc']
        dfcon.at[res[0], 'year_disc'] = min(koiyr, conyr)
        dfkoi.at[index, 'year_disc'] = min(koiyr, conyr)

    # make sure all candidate KOIs aren't in the confirmed table
    for index, ican in dfkoi[koican].iterrows():
        res = np.where((np.abs(dfcon['ra'] - ican['ra']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - ican['dec']) < 1. / 60) &
                       (np.abs(
                           dfcon['pl_orbper'] - ican['koi_period']) < 1 / 60))
        res = res[0]
        assert len(res) == 0

    # there's not an easy way to tie confirmed planets in the TOI table to
    # entries in the confirmed planets table. instead match by RA/Dec/Period
    toicon = dftoi['disp'] == 'Confirmed'
    toican = dftoi['disp'] == 'Candidate'

    # make sure all confirmed TOIs are in the confirmed table exactly once
    for index, icon in dftoi[toicon].iterrows():
        res = np.where((np.abs(dfcon['ra'] - icon['RA']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - icon['Dec']) < 1. / 60) &
                       (np.abs(dfcon['pl_orbper'] - icon['period']) < 1. / 60))[
            0]
        assert len(res) == 1
        tessyr = icon['year']
        conyr = dfcon.at[res[0], 'pl_disc']
        dfcon.at[res[0], 'year_disc'] = min(tessyr, conyr)
        dftoi.at[index, 'year_disc'] = min(tessyr, conyr)

    # make sure all candidate TOIs aren't in the confirmed table
    for index, ican in dftoi[toican].iterrows():
        res = np.where((np.abs(dfcon['ra'] - ican['RA']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - ican['Dec']) < 1. / 60) &
                       (np.abs(dfcon['pl_orbper'] - ican['period']) < 1. / 60))[
            0]
        assert len(res) == 0


def set_insolations(dfcon, dfkoi, dfk2, dftoi, updated_koi_params=True,
                    updated_k2_params=True):
    import numpy as np
    import warnings

    # work out insolation for all the planets

    # Earth insolations = L/Ls * (a/AU)**-2

    # if L is not given, get it from (R/Rs)**2 * (Teff/Tsun)**4
    # if a is not given, get it from a/Rs * Rs
    # or if a/Rs is not given, get it from stellar mass and period with
    # Kepler's third law
    # a/AU = ((P/yr)**2 * M/Msun)**1/3

    # TOI list doesn't give stellar mass or a/R* so I can't calculate my own
    # insolations for any missing values
    dftoi['insol'] = dftoi['Planet Insolation (Earth Flux)'].values * 1

    # KOIs already do this calculation for all possible planets (only omitting
    # the ones without a stellar radius)
    dfkoi['insol'] = dfkoi['koi_insol'].values * 1

    # things have changed, so do it ourselves
    if updated_koi_params:
        lum = 10.**dfkoi['log_lum']
        tmpau = (((dfkoi['koi_period']/365.256)**2) *
                 dfkoi['koi_smass'])**(1./3.)
        outinsol = lum * (tmpau**-2)
        isf = np.isfinite(outinsol)
        dfkoi.loc[isf, 'insol'] = outinsol[isf]

    # Confirmed planets
    insolcon = dfcon['pl_insol'].values * 1
    lumcon = 10. ** (dfcon['st_lum'].values * 1)
    teffcon = dfcon['st_teff'].values * 1
    rstarcon = dfcon['st_rad'].values * 1

    iskep = dfcon['pl_kepflag'].astype(bool)
    isk2 = dfcon['pl_k2flag'].astype(bool)
    # fill in any missing luminosities with our own calculation
    tmplums = (rstarcon ** 2) * ((teffcon / 5772) ** 4)
    lumcon[~np.isfinite(lumcon)] = tmplums[~np.isfinite(lumcon)]

    # fill in any missing semi-major axes from Kepler's third law first
    aucon = dfcon['pl_orbsmax'].values * 1
    pdcon = dfcon['pl_orbper'].values * 1
    mstarcon = dfcon['st_mass'].values * 1
    tmpau2 = (((pdcon / 365.256)**2) * mstarcon)**(1./3.)
    aucon[~np.isfinite(aucon)] = tmpau2[~np.isfinite(aucon)]
    if updated_koi_params:
        torep = iskep & np.isfinite(tmpau2)
        aucon[torep] = tmpau2[torep]
    if updated_k2_params:
        torep = isk2 & np.isfinite(tmpau2)
        aucon[torep] = tmpau2[torep]

    # then fill in any missing semi-major axes with a/R* * R*
    arstarcon = dfcon['pl_ratdor'].values * 1
    tmpau = arstarcon * rstarcon / 215  # convert to AU; 1 AU = 215 Rsun
    aucon[~np.isfinite(aucon)] = tmpau[~np.isfinite(aucon)]

    # calculate insolations ourselves and fill in any missing that we can
    tmpinsol = lumcon * (aucon ** -2)
    insolcon[~np.isfinite(insolcon)] = tmpinsol[~np.isfinite(insolcon)]
    if updated_koi_params:
        torep = iskep & np.isfinite(tmpinsol)
        insolcon[torep] = tmpinsol[torep]
    if updated_k2_params:
        torep = isk2 & np.isfinite(tmpinsol)
        insolcon[torep] = tmpinsol[torep]
    dfcon['insol'] = insolcon

    # K2

    # not every group reports all needed parameters, so take averages and apply
    # them to all rows for that planet
    k2names = dfk2['epic_candname'].values * 1
    rstark2 = dfk2['st_rad'].values * 1
    teffk2 = dfk2['st_teff'].values * 1
    arstark2 = dfk2['pl_ratdor'].values * 1
    mstark2 = dfk2['st_mass'].values * 1
    lumk2 = dfk2['log_lum'].values * 1
    lumk2 = 10.**lumk2
    pdk2 = dfk2['pl_orbper'].values * 1

    ids = np.unique(dfk2['epic_candname'])
    insolk2 = np.zeros(k2names.size) * np.nan

    for iid in ids:
        srch = np.where(k2names == iid)[0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rstar = np.nanmean(rstark2[srch])
            teff = np.nanmean(teffk2[srch])
            arstar = np.nanmean(arstark2[srch])
            mstar = np.nanmean(mstark2[srch])
            lum = np.nanmean(lumk2[srch])
            pd = np.nanmean(pdk2[srch])

            if ~np.isfinite(lum):
                lum = (rstar**2) * ((teff/5772)**4)

            au = (((pd / 365.256) ** 2) * mstar) ** (1. / 3.)
            if ~np.isfinite(au):
                au = arstar * rstar / 215  # convert to AU; 1 AU = 215 Rsun

            insolk2[srch] = lum * (au**-2)

    dfk2['insol'] = insolk2


def load_data(discovery_year=False, updated_koi_params=True,
              updated_k2_params=True, new=False):
    """
    Load our data tables and perform some data cleansing/updating to make them
    ready for use in our interactive figures.

    Parameters
    ----------
    discovery_year : bool
        Whether we need to go through the time intensive step of calculating
        the discovery years of confirmed planets and not just their year of
        confirmation.
    updated_koi_params : bool
        If True, for all stars in the Kepler field, use the updated stellar
        parameters from Berger 2020. Recalculate planet radii, insolations, etc
        using these new Gaia assisted stellar parameters.
    updated_k2_params : bool
        If True, for all stars in the K2 fields, use the updated stellar
        parameters from Hardegree-Ullman 2020. Recalculate planet radii,
        insolations, etc using these new Gaia assisted stellar parameters.
    new : bool
        Whether we're using the new planetary systems composite table instead
        of the old/deprecated confirmed planets table

    Returns
    -------
    dfcon : DataFrame
        All planets in the Exoplanet Archive confirmed planets table.
    dfkoi : DataFrame
        All planets in the Exoplanet Archive KOI planets table.
    dfk2 : DataFrame
        All planets in the Exoplanet Archive K2 planet candidates table.
    dftoi : DataFrame
        All planets in the ExoFOP-TESS planet candidates table.

    """
    import pandas as pd
    import numpy as np
    from astropy.coordinates import Angle
    import warnings

    # load the data files
    if new:
        datafile = 'data/new-confirmed-planets.csv'
    else:
        datafile = 'data/confirmed-planets.csv'
    k2file = 'data/k2-candidates-table.csv'
    koifile = 'data/kepler-kois-full.csv'
    toifile = 'data/tess-candidates.csv'

    k2distfile = 'data/k2oi_distances.txt'
    koidistfile = 'data/koi_distances.txt'

    # the dtype is to silence a pandas warning
    if new:
        ignore_warns = {'hd_name': 'string',  'hip_name': 'string',
                        'pl_orbtperstr': 'string', 'pl_occdepstr': 'string',
                        'pl_projobliqstr': 'string', 'sy_icmagstr': 'string',
                        'pl_trueobliqstr': 'string', 'pl_msinijstr': 'string',
                        'pl_msiniestr': 'string'}
        dfcon = pd.read_csv(datafile, dtype=ignore_warns)
    else:
        dfcon = pd.read_csv(datafile, dtype={'pl_edelink': 'string'})
    dfk2 = pd.read_csv(k2file)
    dfkoi = pd.read_csv(koifile)
    dftoi = pd.read_csv(toifile)

    #########################
    # CONFIRMED PLANET PREP #
    #########################

    # replace the long name with just TESS
    full = 'Transiting Exoplanet Survey Satellite (TESS)'
    if new:
        dfcon['disc_facility'].replace(full, 'TESS', inplace=True)
    else:
        dfcon['pl_facility'].replace(full, 'TESS', inplace=True)
    # set all of these planets as confirmed
    dfcon['status'] = 'Confirmed'

    # where do we want to point people to on clicking?
    if new:
        dfcon['url'] = ('https://exoplanetarchive.ipac.caltech.edu/overview/' +
                        dfcon['hostname'])
    else:
        dfcon['url'] = ('https://exoplanetarchive.ipac.caltech.edu/overview/' +
                        dfcon['pl_hostname'])

    # set up a distance field that is the same in all 4 groups
    if new:
        dfcon['distance_pc'] = dfcon['sy_dist'].values * 1
    else:
        # for confirmed planets, trust Gaia over published values if possible
        condists = dfcon['gaia_dist'].values * 1
        ncd = ~np.isfinite(condists)
        condists[~np.isfinite(condists)] = dfcon['st_dist'][ncd]
        dfcon['distance_pc'] = condists

    #################
    # TOI LIST PREP #
    #################

    # jupiter/earth radius ratio
    radratio = 11.21

    # get easier to reference names for things in the ExoFOP listing
    renames = {'TFOPWG Disposition': 'disp', 'TIC ID': 'TIC',
               'Period (days)': 'period',
               'Planet Radius (R_Earth)': 'prade'}
    dftoi.rename(columns=renames, inplace=True)

    # things that don't have a disposition get PC
    dftoi['disp'].replace(np.nan, 'PC', inplace=True)
    # change this to the status we want to report
    dftoi['disp'].replace('PC', 'Candidate', inplace=True)
    dftoi['disp'].replace('KP', 'Confirmed', inplace=True)
    dftoi['disp'].replace('CP', 'Confirmed', inplace=True)

    # make these useful degrees like all the other catalogs
    dftoi['RA'] = Angle(dftoi['RA'], unit='hourangle').degree
    dftoi['Dec'] = Angle(dftoi['Dec'], unit='degree').degree

    # set these to strings we'd want to show in a figure
    dftoi['TOI'] = 'TOI-' + dftoi['TOI'].astype(str)
    dftoi['host'] = 'TIC ' + dftoi['TIC'].astype(str)

    # give TOIs units of Jupiter radii
    dftoi['pradj'] = dftoi['prade'] / radratio

    # set the appropriate discovery facility for candidates
    dftoi['pl_facility'] = 'TESS'

    # where do we want to point people to on clicking?
    dftoi['url'] = ('https://exofop.ipac.caltech.edu/tess/target.php?id=' +
                    dftoi['TIC'].astype(str))

    # the TOI list from ExoFOP isn't always kept synced with the confirmed
    # planets table, so do some shifting of categories here.
    # match planets between tables by RA/Dec/Period
    toicon = dftoi['disp'] == 'Confirmed'
    toican = dftoi['disp'] == 'Candidate'

    # any supposedly confirmed TOIs that aren't in the table get demoted back
    # to candidate
    for index, icon in dftoi[toicon].iterrows():
        res = np.where((np.abs(dfcon['ra'] - icon['RA']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - icon['Dec']) < 1. / 60) &
                       (np.abs(dfcon['pl_orbper'] - icon['period']) < 1. / 60))
        res = res[0]
        if len(res) == 0:
            dftoi.at[index, 'disp'] = 'Candidate'

    # any candidates in the confirmed table get set as such
    for index, ican in dftoi[toican].iterrows():
        res = np.where((np.abs(dfcon['ra'] - ican['RA']) < 1. / 60) &
                       (np.abs(dfcon['dec'] - ican['Dec']) < 1. / 60) &
                       (np.abs(dfcon['pl_orbper'] - ican['period']) < 1. / 60))
        res = res[0]
        if len(res) == 1:
            dftoi.at[index, 'disp'] = 'Confirmed'

    # the year the TOI was found
    yrs = []
    for ival in dftoi['Date TOI Alerted (UTC)']:
        yrs.append(int(ival[:4]))
    dftoi['year'] = yrs

    # set up a distance field that is the same in all 4 groups
    dftoi['distance_pc'] = dftoi['Stellar Distance (pc)']

    #################
    # KOI LIST PREP #
    #################

    # make these not all caps
    dfkoi['koi_disposition'] = dfkoi['koi_disposition'].str.title()

    # make KOI strings into the format we expect
    dfkoi['kepoi_name'].replace(to_replace='K0+', value='KOI-',
                                regex=True, inplace=True)

    # give KOIs units of Jupiter radii
    dfkoi['koi_pradj'] = dfkoi['koi_prad'] / radratio

    # set the appropriate discovery facility for candidates
    dfkoi['pl_facility'] = 'Kepler'

    # where do we want to point people to on clicking?
    exo = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Display' \
          'Overview/nph-DisplayOverview?objname='
    dfkoi['url'] = (exo + dfkoi['kepoi_name'].str.slice(0, -3) +
                    '&type=KEPLER_TCE_HOST')

    # KOI-1101.02 is a known duplicate of 1101.01. Remove it.
    dfkoi.drop(dfkoi[dfkoi['kepoi_name'] == 'KOI-1101.02'].index, inplace=True)

    # set up a distance field that is the same in all 4 groups
    koidists = np.zeros(dfkoi['kepid'].size)
    kics, k1dists = np.loadtxt(koidistfile, unpack=True)
    kics = kics.astype(int)
    for ii, ikoi in enumerate(dfkoi['kepid']):
        srch = np.where(kics == ikoi)[0]
        if len(srch) == 1:
            koidists[ii] = k1dists[srch[0]]
        elif len(srch) == 0:
            warnings.warn(f'Can not find distance for KIC {ikoi}')
            koidists[ii] = np.nan
        else:
            raise Exception('Multiple distances for KIC {ikoi}?')
    dfkoi['distance_pc'] = koidists

    koicon = dfkoi['koi_disposition'] == 'Confirmed'
    koican = dfkoi['koi_disposition'] == 'Candidate'
    # candidates where the planet is bigger than the star shouldn't exist
    badfit = (dfkoi['koi_ror'] > 1) & (koicon | koican)
    # use depth instead of r/R for radius
    sunearth = 109.1
    rr = np.sqrt(dfkoi['koi_depth']/1e6) * dfkoi['koi_srad'] * sunearth
    badfit = badfit & np.isfinite(rr)
    dfkoi.loc[badfit, 'koi_prad'] = rr[badfit]
    dfkoi.loc[badfit, 'koi_pradj'] = rr[badfit] / radratio

    #################
    # K2 LIST PREP #
    #################

    # make these not all caps
    dfk2['k2c_disp'] = dfk2['k2c_disp'].str.title()

    # K2 tables don't have both columns always filled in
    noearth = (~np.isfinite(dfk2['pl_rade']) & np.isfinite(dfk2['pl_radj']))
    dfk2.loc[noearth, 'pl_rade'] = dfk2.loc[noearth, 'pl_radj'] * radratio

    nojup = (np.isfinite(dfk2['pl_rade']) & (~np.isfinite(dfk2['pl_radj'])))
    dfk2.loc[nojup, 'pl_radj'] = dfk2.loc[nojup, 'pl_rade'] / radratio

    # make an int column of EPICs
    epics = []
    for iep in dfk2['epic_name']:
        epics.append(int(iep[4:]))
    epics = np.array(epics)
    dfk2['epic'] = epics

    # set the appropriate discovery facility for candidates
    dfk2['pl_facility'] = 'K2'

    # where do we want to point people to on clicking?
    dfk2['url'] = ('https://exofop.ipac.caltech.edu/k2/edit_target.php?id=' +
                   dfk2['epic_name'].str.slice(5))

    # these are confirmed planets that aren't listed as such, so match them up
    k2known = ['EPIC 202126849.01', 'EPIC 212555594.02', 'EPIC 201357835.01']
    plname = ['HAT-P-54 b', 'K2-192 b', 'K2-245 b']
    for ii, ik2 in enumerate(k2known):
        srch = np.where(dfk2['epic_candname'] == ik2)[0]

        # make sure these haven't been fixed yet
        assert ~(dfk2.loc[srch, 'k2c_disp'] == 'Confirmed').all()

        dfk2.loc[srch, 'k2c_disp'] = 'Confirmed'
        dfk2.loc[srch, 'pl_name'] = plname[ii]

        # EPIC 201357835.01 is K2-245, but that has a different EPIC: 201357643
        if ik2 == 'EPIC 201357835.01':
            dfk2.loc[srch, 'epic_name'] = 'EPIC 201357643'
            dfk2.loc[srch, 'epic'] = 201357643
            dfk2.loc[srch, 'epic_candname'] = 'EPIC 201357643.01'

    # add in a column for the publication year of the K2 candidates
    yrs = []
    for ival in dfk2['k2c_reflink']:
        yrs.append(int(ival.split('ET_AL__')[1][:4]))
    dfk2['year'] = yrs

    # set up a distance field that is the same in all 4 groups
    k2dists = np.zeros(dfk2['epic'].size)
    epics, epdists = np.loadtxt(k2distfile, unpack=True)
    epics = epics.astype(int)
    for ii, ik2 in enumerate(dfk2['epic']):
        srch = np.where(epics == ik2)[0]
        if len(srch) == 1:
            k2dists[ii] = epdists[srch[0]]
        elif len(srch) == 0:
            warnings.warn(f'Can not find distance for EPIC {ik2}')
            k2dists[ii] = np.nan
        else:
            raise Exception('Multiple distances for EPIC {ik2}?')
    dfk2['distance_pc'] = k2dists

    if updated_koi_params:
        get_new_koi_params(dfcon, dfkoi)
    else:
        dfkoi['log_lum'] = np.nan

    if updated_k2_params:
        get_new_k2_params(dfcon, dfk2)
    else:
        dfk2['log_lum'] = np.nan
        dfk2['st_mass'] = np.nan

    # work out insolation for all the planets
    set_insolations(dfcon, dfkoi, dfk2, dftoi,
                    updated_koi_params=updated_koi_params,
                    updated_k2_params=updated_k2_params)

    if discovery_year:
        set_discovery_year(dfcon, dfkoi, dfk2, dftoi)

    return dfcon, dfkoi, dfk2, dftoi


def log_axis_labels(min_tick=-2.001, max_tick=3.):
    """
    Bokeh can't do subscript or superscript text, which includes scientific
    notation in axis labels. This is a hack script that uses unicode
    superscript values and manually creates pseudo-scientific notation axis
    labels. Any values within log10(min_tick) and log10(max_tick) will be
    displayed as normal, while outside those bounds in either direction will
    be converted to scientific notation.

    Parameters
    ----------
    min_tick : float, optional
        Maximum small log(10) value that will display in scientific notation
        instead of the full decimal representation. The default is -2.001,
        meaning axis labels will go from 9x10^-3 to 0.01.
    max_tick : float, optional
        Minimum large log(10) value that will display in scientific notation
        instead of the full decimal representation. The default is 3, meaning
        axis labels will go from 999 to 10^3.

    Returns
    -------
    str:
        JavaScript code function that generates the appropriate tick labels.

    """
    return f"""
var logtick = Math.log10(tick);
if ((logtick > {min_tick}) && (logtick < {max_tick})){{
    return tick.toLocaleString();
}} else {{
    var power = Math.floor(logtick);
    var retval = 10 + (power.toString()
             .split('')
             .map(function (d) {{ return d === '-' ? '⁻' : '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]; }})
             .join(''));
    var front = (tick/Math.pow(10, power)).toPrecision(2).toString().slice(0,3);
    
    if (front == '1.0'){{
        return retval
    }}
    else if (front.slice(1,3) == '.0'){{
        return front[0] + 'x' + retval
    }}
    
    return front + 'x' + retval
}}
    """


"""
    JavaScript code to create a CSV file from selected data points in a plot

    Parameters
    ----------

"""
csv_creation = """
    function makeCSV(sources, keys, header) {
        const nsources = sources.length;
        const ncolumns = keys.length;
        const lines = [header];
        for (let nn = 0; nn < nsources; nn++) {
            var source = sources[nn];
            const nrows = source.selected.indices.length;
            
            for (let ii = 0; ii < nrows; ii++) {
                let row = [];
                for (let jj = 0; jj < ncolumns; jj++) {
                    const column = keys[jj];
                    var ind = source.selected.indices[ii];
                    row.push(source.data[column][ind].toString());
                }
                lines.push(row.join(', '));
            }
        }
        return lines.join('\\n').concat('\\n');
    };

    // adapted from:
    // https://stackoverflow.com/questions/21012580/is-it-possible-to-write-
    // data-to-file-using-only-javascript
    var textFile = null;
    function makeTextFile(text) {
      var data = new Blob([text], {type: 'text/plain'});

      // If we are replacing a previously generated file we need to
      // manually revoke the object URL to avoid memory leaks.
      if (textFile !== null) {
        window.URL.revokeObjectURL(textFile);
      }

      textFile = window.URL.createObjectURL(data);

      return textFile;
    };

    var link = document.createElement('a');
    //link.setAttribute('download', 'exoplots_download.txt');
    link.setAttribute('target', '_blank');
    link.href = makeTextFile(makeCSV(sources, keys, header));
    document.body.appendChild(link);

    // wait for the link to be added to the document
    window.requestAnimationFrame(function() {
      var event = new MouseEvent('click');
      link.dispatchEvent(event);
      document.body.removeChild(link);
    });
    """
