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


def get_equilibrium_temperature(df, albedo: float = 0.3,
                                radiative_fraction: float = 1):
    """
    Calculate the equilibrium temperature of a planet given the star's
    luminosity and the planet's distance away. Takes into account the planet's
    albedo and the fraction of the planet assumed to be heated and radiating
    at the equilibrium temperature.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe assumed to have columns of 'st_log_lum' and 'semi_au' that
        contain a star's log10 solar luminosity and the planet's semi-major
        axis in AU respectively. These are the only parameters needed to get
        an equilibrium temperature
    albedo : float, optional
        What fraction of a star's incoming light the planet reflects.
    radiative_fraction : float, optional
        What fraction of a planet is heated by the star and radiates at the
        returned equilibrium temperature. Full heat redistribution (entire
        planet is at the same temperature) has a fraction of 1; a tidally
        locked world where only half the planet is heated would be 0.5. The
        cold side is assumed to be negligible (temperature of 0 K).

    Returns
    -------
    astropy Quantity array
    """
    from astropy import constants as const
    import numpy as np

    if albedo < 0 or albedo > 1:
        raise Exception('Invalid albedo. Must be between 0 and 1.')
    if radiative_fraction < 0 or radiative_fraction > 1:
        raise Exception('Invalid radiative fraction. Must be between 0 and 1.')
    num = (10.**df['st_log_lum'].values) * const.L_sun * (1. - albedo)
    semisq = (df['semi_au'].values * const.au)**2
    denom = radiative_fraction * 16 * np.pi * const.sigma_sb * semisq
    return (num / denom)**0.25


def get_esm(df, wavelength_micron: float = 7.5, scale: float = 4.29, **kwargs):
    """
    Calculate the Emission Spectroscopy Metric for planets. The reference
    wavelength, overall scaling, and parameters for calculating the equilibrium
    temperature can all be changed.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe assumed to have columns needed to calculate equilibrium
        temperature and ESM, namely: 'st_log_lum', 'semi_au', 'st_teff',
        'tran_depth_ppm', and 'Kmag' that respectively contain a star's log10
        solar luminosity, the planet's semi-major axis in AU, stellar effective
        temperature, the planet's transit depth in ppm, and the star's K band
        magnitude.
    wavelength_micron : float, optional
        The wavelength to calculate the ESM at, in microns.
    scale : float, optional
        The overall ESM scaling, defaulting to the 4.29 used in
        Kempton et al., 2018.
    kwargs
        Any other parameters are passed to get_equilibrium_temperature.

    Returns
    -------
    astropy Quantity array
    """
    from astropy import constants as const
    from astropy import units as uu
    import numpy as np

    teq = get_equilibrium_temperature(df, **kwargs)
    exp1 = np.exp((const.h * const.c)/(wavelength_micron * uu.um * const.k_B *
                                       df['st_teff'].values * uu.K)) - 1
    exp2 = np.exp((const.h * const.c)/(wavelength_micron * uu.um * const.k_B *
                                       1.1 * teq)) - 1
    ecldep = df['tran_depth_ppm'].values * exp1 / exp2
    return scale * ecldep * (10.**(-0.2 * df['Kmag'].values))


def get_tsm(df, scale: float = 0.19, **kwargs):
    """
    Calculate the Transmission Spectroscopy Metric for planets. The 
    overall S/N scaling value and parameters for calculating the equilibrium
    temperature can all be changed.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe assumed to have columns needed to calculate equilibrium
        temperature and TSM, namely: 'st_log_lum', 'semi_au', 'rade',
        'masse', 'masse_est', 'st_rad', and 'Jmag' that respectively contain 
        a star's log10 solar luminosity, the planet's semi-major axis in AU, 
        the planet's radius, the planet's measured mass, an estimate of
        the planet's mass from M-R relations when measured mass is unavailable,
        the star's radius, and the star's J band magnitude.
    scale : float, optional
        The overall TSM scaling, defaulting to the 0.19 used in
        Kempton et al., 2018 for terrestrial planets.
    kwargs
        Any other parameters are passed to get_equilibrium_temperature.

    Returns
    -------
    ndarray
    """
    import numpy as np
    teq = get_equilibrium_temperature(df, **kwargs)

    num = scale * (df['rade']**3) * teq * (10.**(-0.2*df['Jmag']))
    
    combmass = df['masse_est'].values * 1
    isreal = np.isfinite(df['masse'])
    combmass[isreal] = df.loc[isreal, 'masse']
    
    denom = combmass * (df['st_rad']**2)
    
    return num / denom


def load_data(updated_koi_params=True, updated_k2_params=True, new=True):
    """
    Load our data tables and perform some data cleansing/updating to make them
    ready for use in our interactive figures.

    Parameters
    ----------
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
    comp : DataFrame
        The composite data frame used in making all plots

    """
    import pandas as pd
    import numpy as np
    from astropy.coordinates import Angle
    import warnings
    from glob import glob
    import os

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

    ticparams = 'data/full_tic.txt'

    # the dtype is to silence a pandas warning
    if new:
        ignore_warns = {'hd_name': 'string', 'hip_name': 'string',
                        'pl_orbtperstr': 'string', 'pl_occdepstr': 'string',
                        'pl_projobliqstr': 'string', 'sy_icmagstr': 'string',
                        'pl_trueobliqstr': 'string', 'pl_msinijstr': 'string',
                        'pl_msiniestr': 'string'}
        dfcon = pd.read_csv(datafile, dtype=ignore_warns)
    else:
        dfcon = pd.read_csv(datafile, dtype={'pl_edelink': 'string',
                                             'swasp_id': 'string'})
    dfk2 = pd.read_csv(k2file)
    dfkoi = pd.read_csv(koifile)
    dftoi = pd.read_csv(toifile)
    if os.path.exists(ticparams):
        fulltic = pd.read_csv(ticparams)
    else:
        fulltic = None

    # what columns do we want/need in the final dataframe
    cols = ['name', 'hostname', 'IC', 'disposition', 'period', 'rade', 'radj',
            'masse', 'massj', 'tran_depth_ppm', 'tran_dur_hr', 'semi_au',
            'insol', 'distance_pc', 'year_discovered', 'year_confirmed',
            'discoverymethod', 'facility', 'st_mass', 'st_rad', 'st_teff',
            'st_log_lum', 'Jmag', 'Kmag', 'ra', 'dec', 'flag_tran',
            'flag_kepler', 'flag_k2', 'url']

    #########################
    # CONFIRMED PLANET PREP #
    #########################

    # replace the long name with just TESS
    full = 'Transiting Exoplanet Survey Satellite (TESS)'
    if new:
        dfcon['facility'] = dfcon['disc_facility']
        dfcon['facility'].replace(full, 'TESS', inplace=True)
    else:
        dfcon['facility'] = dfcon['pl_facility']
        dfcon['facility'].replace(full, 'TESS', inplace=True)

    # set all of these planets as confirmed
    dfcon['disposition'] = 'Confirmed'

    # where do we want to point people to on clicking?
    if new:
        dfcon['url'] = ('https://exoplanetarchive.ipac.caltech.edu/overview/' +
                        dfcon['hostname'])
        dfcon['hostname'] = dfcon['hostname']
        dfcon['flag_tran'] = dfcon['tran_flag']
    else:
        dfcon['url'] = ('https://exoplanetarchive.ipac.caltech.edu/overview/' +
                        dfcon['pl_hostname'])
        dfcon['hostname'] = dfcon['pl_hostname']
        dfcon['flag_tran'] = dfcon['pl_tranflag']

    # all transit flags are 0 or 1
    assert ((dfcon['flag_tran'] == 0) | (dfcon['flag_tran'] == 1)).all()
    # now make them bool flags as desired
    dfcon['flag_tran'] = dfcon['flag_tran'].astype(bool)

    # set up a distance field that is the same in all 4 groups and the
    # discovery year
    if new:
        dfcon['distance_pc'] = dfcon['sy_dist'].values * 1
        dfcon['year_discovered'] = dfcon['disc_year'] * 1
    else:
        # for confirmed planets, trust Gaia over published values if possible
        condists = dfcon['gaia_dist'].values * 1
        ncd = ~np.isfinite(condists)
        condists[~np.isfinite(condists)] = dfcon['st_dist'][ncd]
        dfcon['distance_pc'] = condists
        dfcon['year_discovered'] = dfcon['pl_disc'] * 1

    # set our common names for all of these
    renames = {'pl_name': 'name', 'pl_orbper': 'period', 'pl_rade': 'rade',
               'pl_radj': 'radj', 'st_lum': 'st_log_lum', 'pl_insol': 'insol',
               'pl_orbsmax': 'semi_au', 'pl_bmasse': 'masse',
               'pl_bmassj': 'massj', 'sy_kmag': 'Kmag', 'sy_jmag': 'Jmag',
               'pl_trandep': 'tran_depth_ppm', 'pl_trandur': 'tran_dur_hr'}
    dfcon.rename(columns=renames, inplace=True)

    # upper/lower limits are given values at that limit and we need to remove
    # them for now
    dfcon.loc[dfcon['pl_orbperlim'] != 0, 'period'] = np.nan
    dfcon.loc[dfcon['pl_radelim'] != 0, 'rade'] = np.nan
    dfcon.loc[dfcon['pl_radjlim'] != 0, 'radj'] = np.nan
    dfcon.loc[dfcon['st_radlim'] != 0, 'st_rad'] = np.nan
    dfcon.loc[dfcon['st_masslim'] != 0, 'st_mass'] = np.nan
    dfcon.loc[dfcon['st_lumlim'] != 0, 'st_log_lum'] = np.nan
    dfcon.loc[dfcon['st_tefflim'] != 0, 'st_teff'] = np.nan
    dfcon.loc[dfcon['pl_insollim'] != 0, 'insol'] = np.nan
    dfcon.loc[dfcon['pl_orbsmaxlim'] != 0, 'semi_au'] = np.nan
    dfcon.loc[dfcon['pl_ratdorlim'] != 0, 'pl_ratdor'] = np.nan
    dfcon.loc[dfcon['pl_bmasselim'] != 0, 'masse'] = np.nan
    dfcon.loc[dfcon['pl_bmassjlim'] != 0, 'massj'] = np.nan
    dfcon.loc[dfcon['pl_trandeplim'] != 0, 'tran_depth_ppm'] = np.nan
    dfcon.loc[dfcon['pl_trandurlim'] != 0, 'tran_dur_hr'] = np.nan

    ct = 0
    # one planet (OGLE-TR-111 b) has different references for the two masses
    for ii in np.arange(dfcon['masse'].size):
        if (dfcon.at[ii, 'pl_bmasse_reflink'] !=
                dfcon.at[ii, 'pl_bmassj_reflink']):
            ct += 1
    assert ct == 1

    # both always exist or not together
    badm = (np.isfinite(dfcon['masse']) ^ np.isfinite(dfcon['massj']))
    assert badm.sum() == 0

    # remove calculated values from true masses. we'll make a calculated
    # column later
    badme = dfcon['pl_bmasse_reflink'].str.contains('Calculated')
    badmj = dfcon['pl_bmassj_reflink'].str.contains('Calculated')
    assert not (badme ^ badmj).any()
    dfcon.loc[badme, ['masse', 'massj']] = np.nan

    massrat = 317.83
    # XXX: because earth and jup radii don't always agree, make them
    # uniform and treat Earth as truth
    dfcon['massj'] = dfcon['masse'] / massrat

    # jupiter/earth radius ratio
    radratio = 11.21
    badrj = (np.isfinite(dfcon['radj']) ^ np.isfinite(dfcon['rade']))
    # XXX: these bad 2 need to be fixed
    if new:
        assert badrj.sum() == 2
    dfcon.loc[badrj, 'radj'] = dfcon.loc[badrj, 'rade'] / radratio

    # remove calculated values from true masses. we'll make a calculated
    # column later
    badre = dfcon['pl_rade_reflink'].str.contains('Calculated')
    badrj = dfcon['pl_radj_reflink'].str.contains('Calculated')
    assert not (badre ^ badrj).any()
    dfcon.loc[badre, ['rade', 'radj']] = np.nan

    # XXX: because earth and jup radii use different sources, make them
    # uniform and treat Earth as truth
    dfcon['radj'] = dfcon['rade'] / radratio

    # convert their depth in % to depth in ppm
    dfcon['tran_depth_ppm'] *= 1e4
    # we should have transit depths for these
    getdep = (dfcon['flag_tran'] & (~np.isfinite(dfcon['tran_depth_ppm'])) &
              np.isfinite(dfcon['rade']) & np.isfinite(dfcon['st_rad']))

    tranrat = dfcon['rade']**2 / (dfcon['st_rad'] * 109.1)**2
    tranrat *= 1e6
    dfcon.loc[getdep, 'tran_depth_ppm'] = tranrat[getdep]

    # for some reason these claim to have transits but either no
    # stellar or planet radius measurement
    getdep = (dfcon['flag_tran'] & (~np.isfinite(dfcon['tran_depth_ppm'])) &
              np.isfinite(dfcon['rade']) & np.isfinite(dfcon['st_rad']))
    assert getdep.sum() == 0

    # K2-22 b
    getrad = (dfcon['flag_tran'] & np.isfinite(dfcon['tran_depth_ppm']) &
              (~np.isfinite(dfcon['rade'])) & np.isfinite(dfcon['st_rad']))
    assert getrad.sum() == 1
    # don't assume radius from depth because it's a disintegrating planet
    # where depth doesn't equal true radius

    # fix these inconsistencies by hand for now
    baddep = ((dfcon['tran_depth_ppm'] < tranrat/3) |
              (dfcon['tran_depth_ppm'] > tranrat*3)) & (dfcon['rade'] < 4)
    dfcon.loc[baddep, 'tran_depth_ppm'] = tranrat[baddep]

    # set whether or not these were observed by Kepler or K2
    if new:
        dfcon['flag_kepler'] = False
        # these were labeled in the old table as being in the Kepler field
        # but didn't have default host names indicating such
        others = ['PH1', 'PH2', 'TrES-2', 'HAT-P-7', 'HAT-P-11',
                  '2MASS J19383260+4603591']
        for ind, icon in dfcon.iterrows():
            iskep = (icon['hostname'][:3] in ['KOI', 'Kep', 'KIC'] or
                     icon['hostname'] in others)
            if iskep:
                dfcon.loc[ind, 'flag_kepler'] = True

        dfcon['flag_k2'] = False
        # these were labeled in the old table as being in the K2 fields
        # but didn't have default host names indicating such
        others = ['BD+20 594', 'G 9-40', 'GJ 9827', 'HAT-P-56', 'HATS-11',
                  'HATS-12', 'HATS-36', 'HATS-9', 'HD 106315', 'HD 3167',
                  'HD 72490', 'HD 80653', 'HD 89345', 'HIP 116454',
                  'HIP 41378', 'Qatar-2', 'Ross 128', 'TRAPPIST-1',
                  'V1298 Tau', 'WASP-151', 'WASP-157', 'WASP-28', 'WASP-47',
                  'WASP-75', 'WASP-85 A', 'Wolf 503']
        # note to self: EPIC 211945201 b and HD 3167 d are both in the K2
        # fields but unlisted as such in the old confirmed planets table
        for ind, icon in dfcon.iterrows():
            isk2 = (icon['hostname'][:2] == 'K2' or
                    icon['hostname'][:4] == 'EPIC' or
                    icon['hostname'] in others)
            if isk2:
                dfcon.loc[ind, 'flag_k2'] = True
    else:
        dfcon['flag_kepler'] = dfcon['pl_kepflag'].astype(bool)
        dfcon['flag_k2'] = dfcon['pl_k2flag'].astype(bool)

    # fill in any missing luminosities with our own calculation
    # (archive claims they already do this)
    tmplums = (dfcon['st_rad']**2) * ((dfcon['st_teff'] / 5772)**4)
    toadd = (~np.isfinite(dfcon['st_log_lum'])) & np.isfinite(tmplums)
    assert toadd.sum() == 0

    # fill in any missing semi-major axes from Kepler's third law first
    tmpau = (((dfcon['period'] / 365.256)**2) * dfcon['st_mass'])**(1./3.)
    repau = (~np.isfinite(dfcon['semi_au'])) & np.isfinite(tmpau)
    dfcon.loc[repau, 'semi_au'] = tmpau[repau]

    # then fill in any missing semi-major axes with a/R* * R*
    # convert to AU; 1 AU = 215 Rsun
    tmpau2 = dfcon['pl_ratdor'] * dfcon['st_rad'] / 215.03216
    repau2 = (~np.isfinite(dfcon['semi_au'])) & np.isfinite(tmpau2)
    # this so far isn't actually necessary
    assert repau2.sum() == 0
    dfcon.loc[repau2, 'semi_au'] = tmpau2[repau2]

    # calculate insolations ourselves and fill in any missing that we can
    tmpinsol = (10.**dfcon['st_log_lum']) * (dfcon['semi_au']**-2)
    repinsol = (~np.isfinite(dfcon['insol'])) & np.isfinite(tmpinsol)
    dfcon.loc[repinsol, 'insol'] = tmpinsol[repinsol]

    # confirmed planets don't have Input Catalog numbers, so set them to nan
    dfcon['IC'] = float('nan')

    # make a confirmation year column as well
    dfcon['year_confirmed'] = dfcon['year_discovered']

    # do all the tests to make sure things are working

    # everything has a facility name
    assert (np.array([len(xi) for xi in dfcon['facility']]) > 0).all()
    assert (dfcon['disposition'] == 'Confirmed').all()
    # everything has a host name and URL
    assert (np.array([len(xi) for xi in dfcon['hostname']]) > 0).all()
    assert (np.array([len(xi) for xi in dfcon['url']]) > 0).all()
    # everything has a planet name
    assert (np.array([len(xi) for xi in dfcon['name']]) > 0).all()
    # Input Catalog numbers are correct
    assert not np.isfinite(dfcon['IC']).any()
    assert (np.array([len(xi) for xi in dfcon['discoverymethod']]) > 0).all()

    # distances are either NaN or > 1 pc
    assert (~np.isfinite(dfcon['distance_pc']) |
            (dfcon['distance_pc'] > 1)).all()
    # stellar parameters make sense
    assert (~np.isfinite(dfcon['st_rad']) | (dfcon['st_rad'] > 0)).all()
    assert (~np.isfinite(dfcon['st_mass']) | (dfcon['st_mass'] > 0)).all()
    assert (~np.isfinite(dfcon['st_teff']) | (dfcon['st_teff'] > 100)).all()
    assert (~np.isfinite(dfcon['st_log_lum']) |
            ((dfcon['st_log_lum'] > -8) & (dfcon['st_log_lum'] < 5))).all()
    assert (~np.isfinite(dfcon['Kmag']) | (dfcon['Kmag'] > -5)).all()
    assert (~np.isfinite(dfcon['Jmag']) | (dfcon['Jmag'] > -5)).all()

    # RA and Dec are both valid
    assert ((dfcon['ra'] >= 0) & (dfcon['ra'] <= 360.)).all()
    assert ((dfcon['dec'] >= -90) & (dfcon['dec'] <= 90.)).all()

    # planet parameters are either NaN or > 0
    assert (~np.isfinite(dfcon['period']) | (dfcon['period'] > 0)).all()
    assert (~np.isfinite(dfcon['semi_au']) | (dfcon['semi_au'] > 0)).all()
    assert (~np.isfinite(dfcon['insol']) | (dfcon['insol'] > 0)).all()
    assert (~np.isfinite(dfcon['rade']) | (dfcon['rade'] > 0)).all()
    assert (~np.isfinite(dfcon['radj']) | (dfcon['radj'] > 0)).all()
    assert (~np.isfinite(dfcon['masse']) | (dfcon['masse'] > 0)).all()
    assert (~np.isfinite(dfcon['massj']) | (dfcon['massj'] > 0)).all()
    assert (~np.isfinite(dfcon['tran_depth_ppm']) |
            (dfcon['tran_depth_ppm'] > 0)).all()
    assert (~np.isfinite(dfcon['tran_dur_hr']) |
            (dfcon['tran_dur_hr'] > 0)).all()

    # Jup and Earth radii and masses are either defined or not together
    assert np.allclose(np.isfinite(dfcon['radj']), np.isfinite(dfcon['rade']))
    assert ((~np.isfinite(dfcon['rade'])) |
            ((dfcon['rade'] / dfcon['radj'] > 0.99 * radratio) &
             (dfcon['rade'] / dfcon['radj'] < 1.01 * radratio))).all()
    assert np.allclose(np.isfinite(dfcon['massj']),
                       np.isfinite(dfcon['masse']))
    assert ((~np.isfinite(dfcon['masse'])) |
            ((dfcon['masse'] / dfcon['massj'] > 0.99 * massrat) &
             (dfcon['masse'] / dfcon['massj'] < 1.01 * massrat))).all()

    # these flags at least have the right number of good values
    assert dfcon['flag_kepler'].sum() > 2000
    assert dfcon['flag_k2'].sum() > 400
    assert dfcon['flag_tran'].sum() > 3000

    # discovery and confirmation years make sense
    assert (dfcon['year_discovered'] >= 1989).all()
    assert np.allclose(dfcon['year_confirmed'], dfcon['year_discovered'])

    # create the composite, single data frame for all the planets and
    # planet candidates
    comp = dfcon[cols].copy()

    #################
    # KOI LIST PREP #
    #################

    # make these not all caps
    dfkoi['disposition'] = dfkoi['koi_disposition'].str.title()
    assert np.unique(dfkoi['disposition']).size == 3

    # set our common names for all of these
    renames = {'koi_period': 'period', 'koi_prad': 'rade', 'kepid': 'IC',
               'koi_insol': 'insol', 'koi_sma': 'semi_au',
               'koi_smass': 'st_mass', 'koi_srad': 'st_rad',
               'koi_steff': 'st_teff', 'kepoi_name': 'name',
               'koi_kmag': 'Kmag', 'koi_depth': 'tran_depth_ppm',
               'koi_duration': 'tran_dur_hr', 'koi_jmag': 'Jmag'}
    dfkoi.rename(columns=renames, inplace=True)

    # make KOI strings into the format we expect
    dfkoi['name'].replace(to_replace='K0+', value='KOI-', regex=True,
                          inplace=True)

    assert (~np.isfinite(dfkoi['insol']) | (dfkoi['insol'] >= 0)).all()
    assert (~np.isfinite(dfkoi['semi_au']) | (dfkoi['semi_au'] > 0)).all()
    assert (~np.isfinite(dfkoi['st_rad']) | (dfkoi['st_rad'] > 0)).all()
    assert (~np.isfinite(dfkoi['st_mass']) | (dfkoi['st_mass'] >= 0)).all()
    assert (~np.isfinite(dfkoi['st_teff']) | (dfkoi['st_teff'] > 100)).all()
    assert (dfkoi['period'] > 0).all()

    # go through and assign KOIs the year they first showed up in a KOI
    # catalog. We're assuming in this process that a particular KOI number will
    # always refer to the same planet.
    dfkoi['year_discovered'] = 1990

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
        ikoi = row['name']
        if ikoi in k1 or ikoi in k2:
            dfkoi.at[index, 'year_discovered'] = 2011
            continue
        for ii, df in enumerate(dfs):
            if ikoi in df['kepoi_name'].values:
                dfkoi.at[index, 'year_discovered'] = koiyears[ii]
                break

    # we got them all
    assert (dfkoi['year_discovered'].min() == 2011 and
            dfkoi['year_discovered'].max() == 2018)

    # give KOIs units of Jupiter radii
    dfkoi['radj'] = dfkoi['rade'] / radratio

    # set the appropriate discovery facility for candidates
    dfkoi['facility'] = 'Kepler'

    # where do we want to point people to on clicking?
    exo = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Display' \
          'Overview/nph-DisplayOverview?objname='
    dfkoi['url'] = (exo + dfkoi['name'].str.slice(0, -3) +
                    '&type=KEPLER_TCE_HOST')

    # KOI-1101.02 is a known duplicate of 1101.01. Remove it.
    assert len(dfkoi[dfkoi['name'] == 'KOI-1101.02']) == 1
    dfkoi.drop(dfkoi[dfkoi['name'] == 'KOI-1101.02'].index, inplace=True)
    assert len(dfkoi[dfkoi['name'] == 'KOI-1101.02']) == 0

    # there's not an easy way to tie confirmed planets in the KOI table to
    # entries in the confirmed planets table. instead match by RA/Dec/Period
    koicon = dfkoi['disposition'] == 'Confirmed'
    koican = dfkoi['disposition'] == 'Candidate'

    # KOI-4441 and 5475 was a KOI at half the period of the confirmed planet
    # and 5568 a KOI at 1/3 the confirmed period. KOI-523 was 1 day off
    excluded = ['KOI-4441.01', 'KOI-5568.01', 'KOI-5475.01', 'KOI-523.02']
    # what the name is in the confirmed planets table
    real = ['Kepler-1604 b', 'Kepler-1633 b', 'Kepler-1632 b', 'Kepler-177 b']
    fixed = np.zeros(len(excluded), dtype=bool)

    # make sure all confirmed KOIs are in the confirmed table exactly once
    for index, icon in dfkoi[koicon].iterrows():
        res = np.where((np.abs(comp['ra'] - icon['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - icon['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - icon['period']) /
                               icon['period']) < 0.01))
        res = res[0]
        if len(res) != 1:
            # special cases I know about that we can match up manually
            assert icon['name'] in excluded
            rname = real[excluded.index(icon['name'])]
            res = np.where(comp['name'] == rname)[0]
            assert len(res) == 1
            fixed[excluded.index(icon['name'])] = True
        # update and sync the discovery year in both tables
        res = res[0]
        myr = min(comp.at[res, 'year_discovered'], icon['year_discovered'])
        comp.at[res, 'year_discovered'] = myr
        dfkoi.at[index, 'year_discovered'] = myr

    assert fixed.all()

    # these are now confirmed but they didn't update it
    # in the KOI table
    newconf = []
    newconf2 = []
    isconf = np.zeros(len(newconf), dtype=bool)

    # make sure all candidate KOIs aren't in the confirmed table
    for index, ican in dfkoi[koican].iterrows():
        res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - ican['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - ican['period']) /
                               ican['period']) < 0.01))
        res = res[0]
        if len(res) > 0:
            assert len(res) == 1
            assert ican['name'] in newconf
            dfkoi.loc[index, 'disposition'] = 'Confirmed'
            short = newconf2[newconf.index(ican['name'])]
            dfkoi.loc[index, 'kepler_name'] = short
            isconf[newconf.index(ican['name'])] = True
            # update and sync the discovery year in both tables
            res = res[0]
            myr = min(comp.at[res, 'year_discovered'], ican['year_discovered'])
            comp.at[res, 'year_discovered'] = myr
            dfkoi.at[index, 'year_discovered'] = myr

    assert isconf.all()

    # make the KIC the host name here
    dfkoi['hostname'] = 'KIC ' + dfkoi['IC'].astype(str)

    # set up a distance field that is the same in all 4 groups
    koidists = np.zeros(dfkoi['IC'].size)
    kics, k1dists = np.loadtxt(koidistfile, unpack=True)
    kics = kics.astype(int)
    for ii, ikoi in enumerate(dfkoi['IC']):
        srch = np.where(kics == ikoi)[0]
        if len(srch) == 1:
            koidists[ii] = k1dists[srch[0]]
        elif len(srch) == 0:
            warnings.warn(f'Can not find distance for KIC {ikoi}')
            koidists[ii] = np.nan
        else:
            raise Exception(f'Multiple distances for KIC {ikoi}?')
    dfkoi['distance_pc'] = koidists

    # fill in missing luminosities with our own calculation
    tmplums = (dfkoi['st_rad'] ** 2) * ((dfkoi['st_teff'] / 5772) ** 4)
    dfkoi['st_log_lum'] = np.log10(tmplums)

    # KOI insolations only go to 2 sig figs so gets the distant ones wrong
    # so calculate our own in a consistent way
    tmpau = (((dfkoi['period'] / 365.256)**2) * dfkoi['st_mass'])**(1./3.)
    dfkoi.loc[np.isfinite(tmpau), 'semi_au'] = tmpau[np.isfinite(tmpau)]

    tmpinsol = (10.**dfkoi['st_log_lum']) * (dfkoi['semi_au']**-2)
    dfkoi.loc[np.isfinite(tmpinsol), 'insol'] = tmpinsol[np.isfinite(tmpinsol)]

    if updated_koi_params:
        # load the new parameters using Gaia etc
        kk, mm, rr, tt, ll, dd = np.loadtxt('data/koi_params_berger2020.txt',
                                            unpack=True)
        kk = kk.astype(int)

        koicon = dfkoi['disposition'] == 'Confirmed'
        koican = dfkoi['disposition'] == 'Candidate'

        # keep track of which planets in the confirmed list don't have a KOI,
        # so we have to find its KIC/get stellar parameters a fancier way
        cononly = comp['flag_kepler'] & True

        # match the confirmed KOIs to the appropriate confirmed planets and
        # get that KOIs new stellar values
        for index, icon in dfkoi[koicon].iterrows():
            res = np.where((np.abs(comp['ra'] - icon['ra']) < 1. / 60) &
                           (np.abs(comp['dec'] - icon['dec']) < 1. / 60) &
                           (np.abs((comp['period'] - icon['period']) /
                                   icon['period']) < 0.01))
            res = res[0]
            if len(res) != 1:
                # special cases I know about that we can match up manually
                assert icon['name'] in excluded
                rname = real[excluded.index(icon['name'])]
                res = np.where(comp['name'] == rname)[0]
                assert len(res) == 1

            cononly[res[0]] = False

            fd = np.where(kk == icon['IC'])[0]
            if len(fd) != 1:
                warnings.warn(f"Can't find parameters for KIC {icon['IC']}")
            elif ~np.isfinite(mm[fd[0]]):
                continue
            else:
                # only care about updating the confirmed table
                res = res[0]
                comp.at[res, 'st_mass'] = mm[fd]
                oldrad = comp.at[res, 'st_rad'] * 1
                comp.at[res, 'st_rad'] = rr[fd]
                comp.at[res, 'st_teff'] = tt[fd]
                comp.at[res, 'st_log_lum'] = ll[fd]
                comp.at[res, 'distance_pc'] = dd[fd]
                iau = (((comp.at[res, 'period'] / 365.256)**2) * mm[fd])**(1./3)
                iinsol = (10.**ll[fd]) * (iau**-2)
                comp.at[res, 'semi_au'] = iau
                comp.at[res, 'insol'] = iinsol
                srat = rr[fd] / oldrad
                if np.isfinite(srat):
                    comp.at[res, 'rade'] *= srat
                    comp.at[res, 'radj'] *= srat

        # make sure all candidate KOIs have the new parameters
        for index, ican in dfkoi[koican].iterrows():
            res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                           (np.abs(comp['dec'] - ican['dec']) < 1. / 60) &
                           (np.abs((comp['period'] - ican['period']) /
                                   ican['period']) < 0.01))
            res = res[0]
            assert len(res) == 0

            fd = np.where(kk == ican['IC'])[0]
            if len(fd) != 1:
                warnings.warn(f"Can't find parameters for KIC {ican['IC']}")
            elif ~np.isfinite(mm[fd[0]]):
                continue
            else:
                fd = fd[0]
                dfkoi.at[index, 'st_mass'] = mm[fd]
                oldrad = dfkoi.at[index, 'st_rad'] * 1
                dfkoi.at[index, 'st_rad'] = rr[fd]
                dfkoi.at[index, 'st_teff'] = tt[fd]
                dfkoi.at[index, 'st_log_lum'] = ll[fd]
                dfkoi.at[index, 'distance_pc'] = dd[fd]
                iau = (((dfkoi.at[index, 'period'] / 365.256)**2) * mm[fd])
                iau = iau**(1./3.)
                iinsol = (10.**ll[fd]) * (iau**-2)
                dfkoi.at[index, 'semi_au'] = iau
                dfkoi.at[index, 'insol'] = iinsol
                srat = rr[fd] / oldrad
                if np.isfinite(srat):
                    dfkoi.at[index, 'rade'] *= srat
                    dfkoi.at[index, 'radj'] *= srat

        missing = ['KOI-142 c', 'Kepler-78 b', 'Kepler-16 b', 'Kepler-34 b',
                   'Kepler-35 b', 'Kepler-38 b', 'Kepler-47 b', 'Kepler-47 c',
                   'PH1 b', 'KOI-55 b', 'KOI-55 c', 'Kepler-1647 b',
                   'Kepler-413 b', '2MASS J19383260+4603591 b', 'Kepler-453 b',
                   'Kepler-1654 b', 'Kepler-1661 b', 'Kepler-448 c',
                   'Kepler-88 d', 'Kepler-47 d', 'HAT-P-11 c', 'Kepler-90 i']
        fillkics = [5446285, 8435766, 12644769, 8572936, 9837578, 6762829,
                    10020423, 10020423, 4862625, 5807616, 5807616, 5473556,
                    12351927, 9472174, 9632895, 8410697, 6504534, 5812701,
                    5446285, 10020423, 10748390, 11442793]

        # for Kepler planets only on the confirmed list, try to find their KIC
        # from other KOIs in the system
        for index, icon in comp[cononly].iterrows():
            # include the trailing space so we match Kepler 90 & not Kepler 900
            iname = icon['name'][:-1]
            # easy case where the planet name is KIC #### b
            if iname[:3] == 'KIC':
                iskic = int(iname[4:])
            else:
                # get the host name of all similar planets
                isin = comp['name'].str.contains(iname, regex=False)
                matches = comp['hostname'][isin]

                # look for those host names in the KOI list
                haskoi = np.zeros(dfkoi['kepler_name'].size).astype(bool)
                for ik in matches:
                    tmp = dfkoi['kepler_name'].str.contains(ik + ' ')
                    tmp.replace(np.nan, False, inplace=True)
                    haskoi |= tmp

                # make sure they all agree on the KIC
                if haskoi.sum() > 0:
                    iskic = np.unique(dfkoi['IC'][haskoi])
                    assert iskic.size == 1
                # we can't find it, so it's one of these special cases
                else:
                    assert icon['name'] in missing
                    iskic = fillkics[missing.index(icon['name'])]

            fd = np.where(kk == iskic)[0]
            if len(fd) != 1:
                warnings.warn(f"Can't find parameters for {icon['name']}")
            elif ~np.isfinite(mm[fd[0]]):
                continue
            else:
                fd = fd[0]
                comp.at[index, 'st_mass'] = mm[fd]
                oldrad = comp.at[index, 'st_rad'] * 1
                comp.at[index, 'st_rad'] = rr[fd]
                comp.at[index, 'st_teff'] = tt[fd]
                comp.at[index, 'st_log_lum'] = ll[fd]
                comp.at[index, 'distance_pc'] = dd[fd]
                iau = (((comp.at[index, 'period'] / 365.256)**2) * mm[fd])
                iau = iau**(1./3.)
                iinsol = (10.**ll[fd]) * (iau**-2)
                comp.at[index, 'semi_au'] = iau
                comp.at[index, 'insol'] = iinsol
                srat = rr[fd] / oldrad
                if np.isfinite(srat):
                    comp.at[index, 'rade'] *= srat
                    comp.at[index, 'radj'] *= srat

    koicon = dfkoi['disposition'] == 'Confirmed'
    koican = dfkoi['disposition'] == 'Candidate'
    # candidates where the planet is bigger than the star shouldn't exist
    badfit = dfkoi['koi_ror'] > 1
    assert not (badfit & koicon).any()
    # use depth instead of r/R for radius
    sunearth = 109.1
    rr = np.sqrt(dfkoi['tran_depth_ppm'] / 1e6) * dfkoi['st_rad'] * sunearth
    badfit = badfit & koican & np.isfinite(rr)
    dfkoi.loc[badfit, 'rade'] = rr[badfit]
    dfkoi.loc[badfit, 'radj'] = rr[badfit] / radratio

    # make sure things that have radii also have depths and vice versa
    cons = ((~np.isfinite(dfkoi['tran_depth_ppm'])) &
            np.isfinite(dfkoi['rade'])).sum()
    assert cons == 0

    # fix these inconsistencies by hand for now
    tranrat = dfkoi['rade']**2 / (dfkoi['st_rad'] * 109.1)**2
    tranrat *= 1e6
    baddep = ((dfkoi['tran_depth_ppm'] < tranrat/3) |
              (dfkoi['tran_depth_ppm'] > tranrat*3)) & (dfkoi['rade'] < 4)
    dfkoi.loc[baddep, 'tran_depth_ppm'] = tranrat[baddep]

    # all KOIs transit
    dfkoi['flag_tran'] = True

    # all KOIs were observed by Kepler but not K2
    dfkoi['flag_kepler'] = True
    dfkoi['flag_k2'] = False

    # these have not been confirmed
    dfkoi['year_confirmed'] = np.nan

    # all discovered via transit
    dfkoi['discoverymethod'] = 'Transit'

    # no masses
    dfkoi['masse'] = np.nan
    dfkoi['massj'] = np.nan

    # do all the tests to make sure things are working, only focusing on
    # candidates since that's all we're adding to output. FPs can be weird.
    canonly = dfkoi[koican]

    # everything has a facility name
    assert (np.array([len(xi) for xi in canonly['facility']]) > 0).all()
    assert (canonly['disposition'] == 'Candidate').all()
    # everything has a host name and URL
    assert (np.array([len(xi) for xi in canonly['hostname']]) > 0).all()
    assert (np.array([len(xi) for xi in canonly['url']]) > 0).all()
    # everything has a planet name
    assert (np.array([len(xi) for xi in canonly['name']]) > 0).all()
    # Input Catalog numbers are correct
    assert (canonly['IC'] > 0).all()
    assert (canonly['discoverymethod'] == 'Transit').all()

    # distances are either NaN or > 1 pc
    assert (~np.isfinite(canonly['distance_pc']) |
            (canonly['distance_pc'] > 1)).all()
    # stellar parameters make sense
    assert (~np.isfinite(canonly['st_rad']) | (canonly['st_rad'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_mass']) |
            (canonly['st_mass'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_teff']) |
            (canonly['st_teff'] > 3000)).all()
    assert (~np.isfinite(canonly['st_log_lum']) |
            ((canonly['st_log_lum'] > -3) & (canonly['st_log_lum'] < 5))).all()
    assert (~np.isfinite(canonly['Kmag']) | (canonly['Kmag'] > 0)).all()
    assert (~np.isfinite(canonly['Jmag']) | (canonly['Jmag'] > 0)).all()

    # RA and Dec are both valid
    assert ((canonly['ra'] >= 0) & (canonly['ra'] <= 360.)).all()
    assert ((canonly['dec'] >= -90) & (canonly['dec'] <= 90.)).all()

    # planet parameters are either NaN or > 0
    assert (canonly['period'] > 0).all()
    assert (~np.isfinite(canonly['semi_au']) | (canonly['semi_au'] > 0)).all()
    assert (~np.isfinite(canonly['insol']) | (canonly['insol'] > 0)).all()
    assert (~np.isfinite(canonly['rade']) | (canonly['rade'] > 0)).all()
    assert (~np.isfinite(canonly['radj']) | (canonly['radj'] > 0)).all()
    assert (~np.isfinite(canonly['masse'])).all()
    assert (~np.isfinite(canonly['massj'])).all()
    assert (~np.isfinite(canonly['tran_depth_ppm']) |
            (canonly['tran_depth_ppm'] >= 0)).all()
    assert (canonly['tran_dur_hr'] > 0).all()

    # Jup and Earth radii are either defined or not together
    assert np.allclose(np.isfinite(canonly['radj']),
                       np.isfinite(canonly['rade']))
    assert ((~np.isfinite(canonly['rade'])) |
            ((canonly['rade'] / canonly['radj'] > 0.99 * radratio) &
             (canonly['rade'] / canonly['radj'] < 1.01 * radratio))).all()

    # these flags at least have the right number of good values
    assert canonly['flag_kepler'].all()
    assert not canonly['flag_k2'].any()
    assert canonly['flag_tran'].all()

    # discovery and confirmation years make sense
    assert ((canonly['year_discovered'] >= 2011) &
            (canonly['year_discovered'] <= 2018)).all()
    assert not np.isfinite(canonly['year_confirmed']).any()

    # add the KOIs to our final composite table
    koiadd = dfkoi[cols][koican].copy()
    comp = comp.append(koiadd, verify_integrity=True, ignore_index=True)

    ################
    # K2 LIST PREP #
    ################

    # put these into our keywords
    renames = {'k2c_disp': 'disposition', 'pl_rade': 'rade',
               'pl_orbper': 'period', 'pl_radj': 'radj', 'st_k2': 'Kmag',
               'epic_candname': 'name', 'epic_name': 'hostname',
               'pl_trandep': 'tran_depth_ppm', 'pl_trandur': 'tran_dur_hr',
               'st_j2': 'Jmag'}
    dfk2.rename(columns=renames, inplace=True)

    # upper/lower limits are given values at that limit and we need to remove
    # them for now
    dfk2.loc[dfk2['pl_orbperlim'] != 0, 'period'] = np.nan
    dfk2.loc[dfk2['pl_radelim'] != 0, 'rade'] = np.nan
    dfk2.loc[dfk2['pl_radjlim'] != 0, 'radj'] = np.nan
    dfk2.loc[dfk2['st_radlim'] != 0, 'st_rad'] = np.nan
    dfk2.loc[dfk2['st_tefflim'] != 0, 'st_teff'] = np.nan
    dfk2.loc[dfk2['pl_ratdorlim'] != 0, 'pl_ratdor'] = np.nan
    dfk2.loc[dfcon['pl_trandeplim'] != 0, 'tran_depth_ppm'] = np.nan
    dfk2.loc[dfcon['pl_trandurlim'] != 0, 'tran_dur_hr'] = np.nan
    dfk2.loc[dfcon['pl_ratrorlim'] != 0, 'pl_ratror'] = np.nan

    # K2 tables don't have both columns always filled in
    noearth = (~np.isfinite(dfk2['rade']) & np.isfinite(dfk2['radj']))
    dfk2.loc[noearth, 'rade'] = dfk2.loc[noearth, 'radj'] * radratio

    nojup = (np.isfinite(dfk2['rade']) & (~np.isfinite(dfk2['radj'])))
    dfk2.loc[nojup, 'radj'] = dfk2.loc[nojup, 'rade'] / radratio

    # XXX: because earth and jup radii don't always agree, make them
    # uniform and treat Earth as truth
    dfk2['radj'] = dfk2['rade'] / radratio

    # make these not all caps
    dfk2['disposition'] = dfk2['disposition'].str.title()
    assert np.unique(dfk2['disposition']).size == 3

    # make an int column of EPICs
    epics = []
    for iep in dfk2['hostname']:
        epics.append(int(iep[4:]))
    epics = np.array(epics)
    dfk2['IC'] = epics

    # set the appropriate discovery facility for candidates
    dfk2['facility'] = 'K2'

    # where do we want to point people to on clicking?
    dfk2['url'] = ('https://exofop.ipac.caltech.edu/k2/edit_target.php?id=' +
                   dfk2['hostname'].str.slice(5))

    # until this is fixed (the Kruse and Heller .03 are different planets)
    srch = np.where(dfk2['name'] == 'EPIC 201497682.03')[0]
    assert len(srch) == 1
    srch = srch[0]
    assert dfk2.at[srch, 'disposition'] == 'Confirmed'
    assert dfk2.at[srch, 'pl_name'] == 'EPIC 201497682 b'
    dfk2.at[srch, 'disposition'] = 'Candidate'
    dfk2.at[srch, 'pl_name'] = float('nan')
    dfk2.at[srch, 'name'] = 'EPIC 201497682.04'
    srch = np.where(dfk2['name'] == 'EPIC 201497682.04')[0]
    assert len(srch) == 1
    srch = srch[0]
    assert dfk2.at[srch, 'disposition'] == 'Candidate'
    assert ~np.isfinite(dfk2.at[srch, 'pl_name'])

    # add in a column for the publication year of the K2 candidates
    yrs = []
    for ival in dfk2['k2c_reflink']:
        yrs.append(int(ival.split('ET_AL__')[1][:4]))
    dfk2['year_discovered'] = yrs
    assert (dfk2['year_discovered'] > 2014).all()

    # set the discovery year to be the same for all rows of the same planet
    for iplan in np.unique(dfk2['name']):
        srch = dfk2['name'] == iplan
        myr = np.min(dfk2.loc[srch, 'year_discovered'])
        dfk2.loc[srch, 'year_discovered'] = myr

    # check that we're including all K2 planets, but only counting them once
    k2con = dfk2['disposition'] == 'Confirmed'
    k2can = dfk2['disposition'] == 'Candidate'

    # all K2 confirmed planets are already in the confirmed planets table
    notfound = ~np.in1d(dfk2['pl_name'][k2con], comp['name'])
    assert notfound.sum() == 0

    # anything with a planet name in the K2 table but still a candidate hasn't
    # already shown up in the confirmed planets table
    hasname = ~dfk2['pl_name'][k2can].isna()
    assert np.in1d(dfk2['pl_name'][k2can][hasname], comp['name']).sum() == 0

    # XXX: way too many objects don't have a single disposition
    uobjs = np.unique(dfk2['name'])
    bad = []
    for iobj in uobjs:
        vv = np.where(dfk2['name'] == iobj)[0]
        if np.unique(dfk2['disposition'][vv]).size != 1:
            bad.append(iobj)
        # assert np.unique(dfk2['disposition'][vv]).size == 1

    # also test explicitly by RA/Dec/Period

    # the Kruse sample got 201505350 and 203771098 periods wrong,
    # Crossfield got 201637175 wrong, Kruse/Mayo disagree on 212394689.02 by 2x
    k2exclude = ['EPIC 201505350.01', 'EPIC 203771098.01', 'EPIC 201637175.01',
                 'EPIC 212394689.02']
    isexclude = np.zeros(len(k2exclude), dtype=bool)

    # make sure all confirmed K2 planets are in the confirmed table exactly once
    # and update their discovery year
    for index, icon in dfk2[k2con].iterrows():
        res = np.where((np.abs(comp['ra'] - icon['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - icon['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - icon['period']) /
                               icon['period']) < 0.01))
        res = res[0]
        if len(res) != 1:
            # special cases I know about that we can ignore
            assert len(res) == 0
            assert icon['name'] in k2exclude
            isexclude[k2exclude.index(icon['name'])] = True
            res = np.where(comp['name'] == icon['pl_name'])[0]
            assert len(res) == 1
        # update and sync the discovery year in both tables
        res = res[0]
        assert comp.at[res, 'name'] == icon['pl_name']
        myr = min(comp.at[res, 'year_discovered'], icon['year_discovered'])
        comp.at[res, 'year_discovered'] = myr
        dfk2.at[index, 'year_discovered'] = myr

    assert isexclude.all()

    # these are confirmed planets that aren't listed as such, so match them up
    # and set them as confirmed
    k2known = ['EPIC 202126849.01', 'EPIC 212555594.02', 'EPIC 201357835.01']
    plname = ['HAT-P-54 b', 'K2-192 b', 'K2-245 b']
    reknown = np.zeros(len(k2known), dtype=bool)

    # make sure all candidate K2 planets aren't in the confirmed table
    for index, ican in dfk2[k2can].iterrows():
        res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - ican['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - ican['period']) /
                               ican['period']) < 0.01))
        res = res[0]
        if len(res) != 0:
            assert ican['name'] in k2known
            pall = dfk2['name'] == ican['name']
            dfk2.loc[pall, 'disposition'] = 'Confirmed'
            dfk2.loc[pall, 'pl_name'] = plname[k2known.index(ican['name'])]
            reknown[k2known.index(ican['name'])] = True

            # EPIC 201357835.01 is K2-245, but that has a different
            # EPIC: 201357643
            if ican['name'] == 'EPIC 201357835.01':
                dfk2.at[index, 'IC'] = 201357643
                dfk2.at[index, 'name'] = 'EPIC 201357643.01'
                dfk2.at[index, 'hostname'] = 'EPIC 201357643'
                dfk2.at[index, 'k2c_recentflag'] = 0
                u1 = 'https://exofop.ipac.caltech.edu/k2/edit_target.php?id='
                dfk2.at[index, 'url'] = (u1 + dfk2.at[index, 'hostname'][5:])

            # update and sync the discovery year in both tables
            res = res[0]
            myr = min(comp.at[res, 'year_discovered'], ican['year_discovered'])
            comp.at[res, 'year_discovered'] = myr
            dfk2.at[index, 'year_discovered'] = myr

    assert reknown.all()

    # set up a distance field that is the same in all 4 groups
    k2dists = np.zeros(dfk2['IC'].size)
    epics, epdists = np.loadtxt(k2distfile, unpack=True)
    epics = epics.astype(int)
    for ii, ik2 in enumerate(dfk2['IC']):
        srch = np.where(epics == ik2)[0]
        if len(srch) == 1:
            k2dists[ii] = epdists[srch[0]]
        elif len(srch) == 0:
            warnings.warn(f'Can not find distance for EPIC {ik2}')
            k2dists[ii] = np.nan
        else:
            raise Exception('Multiple distances for EPIC {ik2}?')
    dfk2['distance_pc'] = k2dists

    # all K2 candidates transit
    dfk2['flag_tran'] = True

    # convert their duration in days to hours
    dfk2['tran_dur_hr'] *= 24

    # convert their depth in % to depth in ppm
    dfk2['tran_depth_ppm'] *= 1e4
    # we should have transit depths for these
    getdep = (dfk2['flag_tran'] & (~np.isfinite(dfk2['tran_depth_ppm'])) &
              np.isfinite(dfk2['rade']) & np.isfinite(dfk2['st_rad']))

    tranrat = dfk2['rade']**2 / (dfk2['st_rad'] * 109.1)**2
    dfk2.loc[getdep, 'tran_depth_ppm'] = tranrat[getdep] * 1e6

    # we should have a radius if they gave a depth
    getrad = (dfk2['flag_tran'] & np.isfinite(dfk2['tran_depth_ppm']) &
              (~np.isfinite(dfk2['rade'])) & np.isfinite(dfk2['st_rad']))
    tranrad = np.sqrt((dfk2['tran_depth_ppm']/1e6) * (dfk2['st_rad']**2))
    tranrad *= 109.1
    dfk2.loc[getrad, 'rade'] = tranrad[getrad]
    dfk2.loc[getrad, 'radj'] = tranrad[getrad] / radratio

    # fix these inconsistencies by hand for now
    tranrat = dfk2['rade']**2 / (dfk2['st_rad'] * 109.1)**2
    tranrat *= 1e6
    baddep = ((dfk2['tran_depth_ppm'] < tranrat/3) |
              (dfk2['tran_depth_ppm'] > tranrat*3)) & (dfk2['rade'] < 4)
    dfk2.loc[baddep, 'tran_depth_ppm'] = tranrat[baddep]

    # fill in missing luminosities with our own calculation
    tmplums = (dfk2['st_rad'] ** 2) * ((dfk2['st_teff'] / 5772) ** 4)
    dfk2['st_log_lum'] = np.log10(tmplums)

    # these columns aren't in the table by default, and we can't calculate
    # them without a stellar mass
    dfk2['st_mass'] = np.nan
    dfk2['semi_au'] = np.nan
    dfk2['insol'] = np.nan

    if updated_k2_params:
        k2paramfile = 'data/k2_params_hardegree-ullman2020.txt'
        ee, mm, rr, tt, ll, dd = np.loadtxt(k2paramfile, unpack=True)
        ee = ee.astype(int)

        # these are bad fits
        hubad = np.where(mm < 0.06)
        mm[hubad] = np.nan
        rr[hubad] = np.nan
        tt[hubad] = np.nan
        ll[hubad] = np.nan
        dd[hubad] = np.nan

        k2con = dfk2['disposition'] == 'Confirmed'
        k2can = dfk2['disposition'] == 'Candidate'

        # all K2 confirmed planets are already in the confirmed planets table
        notfound = ~np.in1d(dfk2['pl_name'][k2con], comp['name'])
        assert notfound.sum() == 0
        assert dfk2['pl_name'][k2con].isna().sum() == 0

        # anything with a planet name in the K2 table but still a candidate
        # hasnt' already shown up in the confirmed planets table
        hasname = ~dfk2['pl_name'][k2can].isna()
        assert np.in1d(dfk2['pl_name'][k2can][hasname], comp['name']).sum() == 0

        # keep track of which planets in the confirmed list don't have a K2
        # cand, so we have to find its EPIC/stellar parameters a fancier way
        cononly = comp['flag_k2'] & True

        # match the confirmed K2 candidates to the appropriate confirmed planets
        for index, icon in dfk2[k2con].iterrows():
            res = np.where(icon['pl_name'] == comp['name'])
            res = res[0]
            assert len(res) == 1
            cononly[res[0]] = False
            # make sure both tables have the new parameters
            fd = np.where(ee == icon['IC'])[0]
            if len(fd) != 1:
                warnings.warn(f"Can't find parameters for EPIC {icon['IC']}")
            elif ~np.isfinite(mm[fd[0]]):
                continue
            else:
                # only care about updating the confirmed table
                res = res[0]
                comp.at[res, 'st_mass'] = mm[fd]
                oldrad = comp.at[res, 'st_rad'] * 1
                comp.at[res, 'st_rad'] = rr[fd]
                comp.at[res, 'st_teff'] = tt[fd]
                comp.at[res, 'st_log_lum'] = ll[fd]
                comp.at[res, 'distance_pc'] = dd[fd]
                iau = (((comp.at[res, 'period'] / 365.256)**2) * mm[fd])
                iau = iau**(1./3.)
                iinsol = (10.**ll[fd]) * (iau**-2)
                comp.at[res, 'semi_au'] = iau
                comp.at[res, 'insol'] = iinsol
                srat = rr[fd] / oldrad
                if np.isfinite(srat):
                    comp.at[res, 'rade'] *= srat
                    comp.at[res, 'radj'] *= srat

        # make sure all candidate K2 planets have the new parameters
        for index, ican in dfk2[k2can].iterrows():
            res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                           (np.abs(comp['dec'] - ican['dec']) < 1. / 60) &
                           (np.abs((comp['period'] - ican['period']) /
                                   ican['period']) < 0.01))
            res = res[0]
            assert len(res) == 0

            fd = np.where(ee == ican['IC'])[0]
            if len(fd) != 1:
                warnings.warn(f"Can't find parameters for EPIC {ican['IC']}")
            elif ~np.isfinite(mm[fd[0]]):
                continue
            else:
                fd = fd[0]
                dfk2.at[index, 'st_mass'] = mm[fd]
                oldrad = dfk2.at[index, 'st_rad'] * 1
                dfk2.at[index, 'st_rad'] = rr[fd]
                dfk2.at[index, 'st_teff'] = tt[fd]
                dfk2.at[index, 'st_log_lum'] = ll[fd]
                dfk2.at[index, 'distance_pc'] = dd[fd]
                iau = (((dfk2.at[index, 'period'] / 365.256)**2) * mm[fd])
                iau = iau**(1./3.)
                iinsol = (10.**ll[fd]) * (iau**-2)
                dfk2.at[index, 'semi_au'] = iau
                dfk2.at[index, 'insol'] = iinsol
                srat = rr[fd] / oldrad
                if np.isfinite(srat):
                    dfk2.at[index, 'rade'] *= srat
                    dfk2.at[index, 'radj'] *= srat

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
                   'TRAPPIST-1 d', 'TRAPPIST-1 e', 'TRAPPIST-1 f',
                   'TRAPPIST-1 g', 'TRAPPIST-1 h', 'V1298 Tau b', 'V1298 Tau c',
                   'V1298 Tau d', 'V1298 Tau e', 'WASP-151 b', 'WASP-28 b',
                   'Wolf 503 b', 'K2-315 b', 'K2-316 b', 'K2-316 c', 'K2-317 b',
                   'K2-318 b', 'K2-319 b', 'K2-320 b', 'K2-321 b', 'K2-322 b',
                   'K2-323 b', 'K2-324 b', 'K2-325 b', 'K2-326 b']
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
                     246074965, 246472939]

        # for K2 planets only on the confirmed list, try to find their EPIC
        # from other KOIs in the system
        for index, icon in comp[cononly].iterrows():
            # include the trailing space so we match K2 23 but not K2 230
            iname = icon['name'][:-1]
            # easy case where the planet name is EPIC #### b
            if iname[:4] == 'EPIC':
                isepic = int(iname[4:])
            else:
                # get the host name of all similar planets
                isin = comp['name'].str.contains(iname, regex=False)
                matches = comp['hostname'][isin]

                # look for those host names in the K2 candidate list
                haskoi = np.zeros(dfk2['pl_name'].size).astype(bool)
                for ik in matches:
                    tmp = dfk2['pl_name'].str.contains(ik + ' ')
                    tmp.replace(np.nan, False, inplace=True)
                    haskoi |= tmp

                # make sure they all agree on the KIC
                if haskoi.sum() > 0:
                    isepic = np.unique(dfk2['IC'][haskoi])
                    assert isepic.size == 1
                # we can't find it, so it's one of these special cases
                else:
                    assert icon['name'] in missing
                    isepic = fillepics[missing.index(icon['name'])]

            fd = np.where(ee == isepic)[0]
            if len(fd) != 1:
                warnings.warn(f"Can't find parameters for {icon['name']}")
            elif ~np.isfinite(mm[fd[0]]):
                continue
            else:
                fd = fd[0]
                comp.at[index, 'st_mass'] = mm[fd]
                oldrad = comp.at[index, 'st_rad'] * 1
                comp.at[index, 'st_rad'] = rr[fd]
                comp.at[index, 'st_teff'] = tt[fd]
                comp.at[index, 'st_log_lum'] = ll[fd]
                comp.at[index, 'distance_pc'] = dd[fd]
                iau = (((comp.at[index, 'period'] / 365.256)**2) * mm[fd])
                iau = iau**(1./3.)
                iinsol = (10.**ll[fd]) * (iau**-2)
                comp.at[index, 'semi_au'] = iau
                comp.at[index, 'insol'] = iinsol
                srat = rr[fd] / oldrad
                if np.isfinite(srat):
                    comp.at[index, 'rade'] *= srat
                    comp.at[index, 'radj'] *= srat

    # no K2 candidate observed by Kepler prime but all by K2
    dfk2['flag_kepler'] = False
    dfk2['flag_k2'] = True

    # these have not been confirmed
    dfk2['year_confirmed'] = np.nan

    # all found via transit
    dfk2['discoverymethod'] = 'Transit'

    k2can = (dfk2['disposition'] == 'Candidate') & (dfk2['k2c_recentflag'] == 1)

    # XXX: once the dispositions get fixed
    # assert k2can.sum() == np.unique(dfk2[dfk2['disposition'] ==
    #                                 'Candidate']['name']).size

    # go through the ones we're going to add to the table and fill in any
    # missing values if possible from other entries in the table
    ichk = ['period', 'rade', 'radj', 'st_rad', 'st_teff', 'st_log_lum',
            'st_mass', 'insol', 'semi_au', 'tran_depth_ppm']

    for index, ican in dfk2[k2can].iterrows():
        srch = np.where((dfk2['name'] == ican['name']) &
                        (dfk2['k2c_recentflag'] == 0))[0]
        if len(srch) == 0:
            continue
        for icol in ichk:
            if ((not np.isfinite(dfk2.at[index, icol])) and
                    np.isfinite(dfk2[icol][srch]).any()):
                dfk2.at[index, icol] = np.nanmedian(dfk2[icol][srch])

    # this EPIC doesn't really exist and thus doesn't have properties.
    # at least give it a sky position from MAST
    dfk2.loc[dfk2['IC'] == 229228348, 'ra'] = 289.5273417
    dfk2.loc[dfk2['IC'] == 229228348, 'dec'] = -16.3430889

    # no masses
    dfk2['masse'] = np.nan
    dfk2['massj'] = np.nan

    # do all the tests to make sure things are working, only focusing on
    # candidates since that's all we're adding to output. FPs can be weird.
    canonly = dfk2[k2can]

    # everything has a facility name
    assert (np.array([len(xi) for xi in canonly['facility']]) > 0).all()
    assert (canonly['disposition'] == 'Candidate').all()
    # everything has a host name and URL
    assert (np.array([len(xi) for xi in canonly['hostname']]) > 0).all()
    assert (np.array([len(xi) for xi in canonly['url']]) > 0).all()
    # everything has a planet name
    assert (np.array([len(xi) for xi in canonly['name']]) > 0).all()
    # Input Catalog numbers are correct
    assert (canonly['IC'] > 0).all()
    assert (canonly['discoverymethod'] == 'Transit').all()

    # distances are either NaN or > 1 pc
    assert (~np.isfinite(canonly['distance_pc']) |
            (canonly['distance_pc'] > 1)).all()
    # stellar parameters make sense
    assert (~np.isfinite(canonly['st_rad']) | (canonly['st_rad'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_mass']) |
            (canonly['st_mass'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_teff']) |
            (canonly['st_teff'] > 2900)).all()
    assert (~np.isfinite(canonly['st_log_lum']) |
            ((canonly['st_log_lum'] > -3) & (canonly['st_log_lum'] < 7))).all()
    assert (~np.isfinite(canonly['Kmag']) | (canonly['Kmag'] > 4)).all()
    assert (~np.isfinite(canonly['Jmag']) | (canonly['Jmag'] > 4)).all()

    # RA and Dec are both valid
    assert ((canonly['ra'] >= 0) & (canonly['ra'] <= 360.)).all()
    assert ((canonly['dec'] >= -90) & (canonly['dec'] <= 90.)).all()

    # planet parameters are either NaN or > 0
    assert ((~np.isfinite(canonly['period'])) | (canonly['period'] > 0)).all()
    assert (~np.isfinite(canonly['semi_au']) | (canonly['semi_au'] > 0)).all()
    assert (~np.isfinite(canonly['insol']) | (canonly['insol'] > 0)).all()
    assert (~np.isfinite(canonly['rade']) | (canonly['rade'] > 0)).all()
    assert (~np.isfinite(canonly['radj']) | (canonly['radj'] > 0)).all()
    assert (~np.isfinite(canonly['masse'])).all()
    assert (~np.isfinite(canonly['massj'])).all()
    assert (~np.isfinite(canonly['tran_depth_ppm']) |
            (canonly['tran_depth_ppm'] > 0)).all()
    assert (~np.isfinite(canonly['tran_dur_hr']) |
            (canonly['tran_dur_hr'] >= 0)).all()

    # Jup and Earth radii are either defined or not together
    assert np.allclose(np.isfinite(canonly['radj']),
                       np.isfinite(canonly['rade']))
    assert ((~np.isfinite(canonly['rade'])) |
            ((canonly['rade'] / canonly['radj'] > 0.99 * radratio) &
             (canonly['rade'] / canonly['radj'] < 1.01 * radratio))).all()

    # these flags at least have the right number of good values
    assert not canonly['flag_kepler'].any()
    assert canonly['flag_k2'].all()
    assert canonly['flag_tran'].all()

    # discovery and confirmation years make sense
    assert (canonly['year_discovered'] >= 2015).all()
    assert not np.isfinite(canonly['year_confirmed']).any()

    # add the K2 candidates to our final composite table
    k2add = dfk2[cols][k2can].copy()
    comp = comp.append(k2add, verify_integrity=True, ignore_index=True)

    #################
    # TOI LIST PREP #
    #################

    # get easier to reference names for things in the ExoFOP listing
    renames = {'TFOPWG Disposition': 'disposition', 'TIC ID': 'IC',
               'Period (days)': 'period',
               'Planet Radius (R_Earth)': 'rade',
               'Stellar Radius (R_Sun)': 'st_rad',
               'Stellar Eff Temp (K)': 'st_teff',
               'Stellar Distance (pc)': 'distance_pc',
               'Planet Insolation (Earth Flux)': 'insol',
               'Depth (ppm)': 'tran_depth_ppm',
               'Duration (hours)': 'tran_dur_hr'}
    dftoi.rename(columns=renames, inplace=True)

    # set these to strings we'd want to show in a figure
    dftoi['name'] = 'TOI-' + dftoi['TOI'].astype(str)
    dftoi['hostname'] = 'TIC ' + dftoi['IC'].astype(str)

    # download TIC info for any new entries
    if (fulltic is None) or (not np.in1d(dftoi['IC'], fulltic['ID']).all()):
        from astroquery.mast import Catalogs
        for index, irow in dftoi.iterrows():
            if (fulltic is None) or (irow['IC'] not in fulltic['ID'].values):
                print('Getting TIC info', index, dftoi['IC'].size)
                cat = Catalogs.query_criteria(catalog='tic', ID=irow['IC'])
                assert len(cat) == 1 and int(cat['ID'][0]) == irow['IC']
                head, istr = cat.to_pandas().to_csv().split()
                if not os.path.exists(ticparams):
                    with open(ticparams, 'w') as off:
                        off.write(head + '\n')
                with open(ticparams, 'a') as off:
                    off.write(istr + '\n')
                fulltic = pd.read_csv(ticparams)

    assert np.in1d(dftoi['IC'], fulltic['ID']).all()

    # by default, we assume new candidates aren't in Kepler/K2 data
    dftoi['flag_kepler'] = False
    dftoi['flag_k2'] = False

    # orbital periods are either NaN or > 0 days
    noper = dftoi['period'] == 0
    dftoi.loc[noper, 'period'] = np.nan
    assert (~np.isfinite(dftoi['period']) | (dftoi['period'] > 0)).all()

    # things that don't have a disposition get PC
    dftoi['disposition'].replace(np.nan, 'PC', inplace=True)
    # change this to the status we want to report
    dftoi['disposition'].replace('PC', 'Candidate', inplace=True)
    dftoi['disposition'].replace('KP', 'Confirmed', inplace=True)
    dftoi['disposition'].replace('CP', 'Confirmed', inplace=True)
    dftoi['disposition'].replace('APC', 'Candidate', inplace=True)
    dftoi['disposition'].replace('FA', 'False Positive', inplace=True)
    dftoi['disposition'].replace('FP', 'False Positive', inplace=True)
    assert np.unique(dftoi['disposition']).size == 3

    # make these useful degrees like all the other catalogs
    dftoi['ra'] = Angle(dftoi['RA'], unit='hourangle').degree
    dftoi['dec'] = Angle(dftoi['Dec'], unit='degree').degree

    # give TOIs units of Jupiter radii
    dftoi['radj'] = dftoi['rade'] / radratio

    # set the appropriate discovery facility for candidates
    dftoi['facility'] = 'TESS'

    # where do we want to point people to on clicking?
    u2 = 'https://exofop.ipac.caltech.edu/tess/target.php?id='
    dftoi['url'] = (u2 + dftoi['IC'].astype(str))

    # the year the TOI was found
    yrs = []
    for ival in dftoi['Date TOI Alerted (UTC)']:
        yrs.append(int(ival[:4]))
    dftoi['year_discovered'] = yrs
    assert (np.array(yrs) > 2017).all()

    # these are WASP-30 and LP 261-75, brown dwarfs and not a real planet
    bds = ['TOI-239.01', 'TOI-1779.01']
    for ibd in bds:
        bd = np.where(dftoi['name'] == ibd)[0][0]
        assert dftoi.loc[bd, 'disposition'] == 'Confirmed'
        dftoi.loc[bd, 'disposition'] = 'False Positive'

    # TOI-515 and 844.01 are also K2 planet candidates, so remove it from
    # counting as a TESS planet. should I do anything else about this?
    # TOI-1241 is KOI-5
    prevs = ['TOI-515.01', 'TOI-844.01', 'TOI-1241.01']
    for ipr in prevs:
        pr = np.where(dftoi['name'] == ipr)[0][0]
        assert dftoi.loc[pr, 'disposition'] == 'Candidate'
        dftoi.loc[pr, 'disposition'] = 'False Positive'
        if ipr == 'TOI-1241.01':
            dftoi.loc[pr, 'flag_kepler'] = True
        else:
            dftoi.loc[pr, 'flag_k2'] = True

    # the TOI list from ExoFOP isn't always kept synced with the confirmed
    # planets table, so do some shifting of categories here.
    # match planets between tables by RA/Dec/Period
    toicon = dftoi['disposition'] == 'Confirmed'
    toican = dftoi['disposition'] == 'Candidate'

    # any supposedly confirmed TOIs that aren't in the table get demoted back
    # to candidate

    # these ones we have an explanation for and know they're properly in the
    # confirmed table.
    # 186, 1793, 1899, 2011, 2221 are single transits, so no period
    # matching
    # 1338 is the CBP, so not sure where its wrong TOI period came from
    # 1456 was a single obvious transit and the second hidden in scattered
    # light causing SPOC to miss it and get the period wrong
    # 351 TESS got the period wrong by 2x
    ignores = ['TOI-186.01', 'TOI-1338.01', 'TOI-1456.01',
               'TOI-1793.01', 'TOI-1899.01', 'TOI-2011.01', 'TOI-2221.01',
               'TOI-351.01']
    conname = ['GJ 143 b', 'TOI-1338 b', 'HD 332231 b', 'HD 95338 b',
               'TOI-1899 b', 'HD 136352 b', 'AU Mic b', 'WASP-165 b']
    # we know what these are and they have paper trails of submitted papers
    # though some were submitted way back in 2014 and still in limbo
    # only TOI-1918/481/892 I can't figure out what it is yet
    waiting = ['TOI-126.01', 'TOI-143.01', 'TOI-257.01',
               'TOI-295.01', 'TOI-626.01', 'TOI-657.01',
               'TOI-834.01', 'TOI-840.01', 'TOI-857.01',
               'TOI-1071.01', 'TOI-1493.01', 'TOI-1494.01', 'TOI-1580.01',
               'TOI-1603.01', 'TOI-1826.01',
               'TOI-1918.01', 'TOI-481.01', 'TOI-892.01',
               'TOI-2179.01', 'TOI-2330.01']

    stillbad = np.zeros(len(ignores), dtype=bool)
    stillwaiting = np.zeros(len(waiting), dtype=bool)
    for index, icon in dftoi[toicon].iterrows():
        res = np.where((np.abs(comp['ra'] - icon['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - icon['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - icon['period']) /
                               comp['period']) < 0.01))
        res = res[0]
        if len(res) == 0:
            # keep these as confirmed, they just failed matching
            if icon['name'] in ignores:
                stillbad[ignores.index(icon['name'])] = True
                res = np.where(comp['name'] ==
                               conname[ignores.index(icon['name'])])[0]
            # these haven't made the confirmed table yet, so they're
            # officially still candidates
            else:
                assert icon['name'] in waiting
                stillwaiting[waiting.index(icon['name'])] = True
                dftoi.at[index, 'disposition'] = 'Candidate'

            # try looking for just same location on the sky and make sure
            # nothing new comes up
            if ~np.isfinite(icon['period']):
                res2 = np.where((np.abs(comp['ra'] - icon['ra']) < 1. / 60) &
                                (np.abs(comp['dec'] - icon['dec']) < 1. / 60))
                res2 = res2[0]
                if len(res2) > 0:
                    assert icon['name'] in ignores
        assert len(res) < 2
        if len(res) == 1:
            # update and sync the discovery year in both tables
            res = res[0]
            myr = min(comp.at[res, 'year_discovered'], icon['year_discovered'])
            comp.at[res, 'year_discovered'] = myr
            dftoi.at[index, 'year_discovered'] = myr

    assert stillbad.all()
    assert stillwaiting.all()

    # any candidates that appear in the confirmed table need to be upgraded

    # these are now confirmed and need to be updated as such
    tobeconf = ['TOI-125.03', 'TOI-129.01', 'TOI-132.01', 'TOI-134.01',
                'TOI-150.01', 'TOI-157.01', 'TOI-169.01', 'TOI-186.02',
                'TOI-294.01', 'TOI-448.01', 'TOI-652.01', 'TOI-700.01',
                'TOI-700.02', 'TOI-700.03', 'TOI-704.01', 'TOI-732.01',
                'TOI-732.02', 'TOI-736.01', 'TOI-736.02', 'TOI-1078.01',
                'TOI-1339.01', 'TOI-1339.02', 'TOI-1462.01', 'TOI-1728.01',
                'TOI-1690.01', 'TOI-193.01', 'TOI-824.01', 'TOI-1339.03',
                'TOI-421.01', 'TOI-540.01', 'TOI-1266.01', 'TOI-1266.02',
                'TOI-488.01', 'TOI-837.01']
    tbc = np.zeros(len(tobeconf), dtype=bool)
    # single transits that should be set as confirmed
    nopermatch = ['TOI-1847.01']
    confmatch = ['NGTS-11 b']
    singconf = np.zeros(len(nopermatch), dtype=bool)

    # any candidates in the confirmed table get set as such
    for index, ican in dftoi[toican].iterrows():
        res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                       (np.abs(comp['dec'] - ican['dec']) < 1. / 60) &
                       (np.abs((comp['period'] - ican['period']) /
                               comp['period']) < 0.01))
        res = res[0]
        # period and location match, so mark this as confirmed
        if len(res) == 1:
            dftoi.at[index, 'disposition'] = 'Confirmed'
            tbc[tobeconf.index(ican['name'])] = True
            # update and sync the discovery year in both tables
            res = res[0]
            myr = min(comp.at[res, 'year_discovered'], ican['year_discovered'])
            comp.at[res, 'year_discovered'] = myr
            dftoi.at[index, 'year_discovered'] = myr
        else:
            assert len(res) == 0
            # try looking for just same location on the sky
            if ~np.isfinite(ican['period']):
                res = np.where((np.abs(comp['ra'] - ican['ra']) < 1. / 60) &
                               (np.abs(comp['dec'] - ican['dec']) < 1. / 60))[0]
                if len(res) > 0:
                    assert ican['name'] in nopermatch
                    res = np.where(comp['name'] ==
                                   confmatch[nopermatch.index(ican['name'])])[0]
                    assert len(res) == 1
                    # these two are single transits of confirmed planets
                    dftoi.at[index, 'disposition'] = 'Confirmed'
                    singconf[nopermatch.index(ican['name'])] = True
                    # update and sync the discovery year in both tables
                    res = res[0]
                    myr = min(comp.at[res, 'year_discovered'],
                              ican['year_discovered'])
                    comp.at[res, 'year_discovered'] = myr
                    dftoi.at[index, 'year_discovered'] = myr

    assert tbc.all()
    assert singconf.all()

    newcan = dftoi['disposition'] == 'Candidate'
    newcon = dftoi['disposition'] == 'Confirmed'

    assert newcan.sum() == (toican.sum() + len(waiting) - len(tobeconf) -
                            len(nopermatch))
    assert newcon.sum() == (toicon.sum() + len(tobeconf) + len(nopermatch) -
                            len(waiting))

    # replace everything with new TIC parameters because apparently some TOIs
    # were using TIC v7 and things aren't self-consistent
    dftoi['Kmag'] = np.nan
    dftoi['Jmag'] = np.nan
    for index, itoi in dftoi.iterrows():
        mt = np.where(itoi['IC'] == fulltic['ID'])[0][0]
        if np.isfinite(fulltic['Teff'][mt]):
            dftoi.at[index, 'st_teff'] = fulltic['Teff'][mt]
        if np.isfinite(fulltic['mass'][mt]):
            dftoi.at[index, 'st_mass'] = fulltic['mass'][mt]
        if np.isfinite(fulltic['lum'][mt]) and fulltic['lum'][mt] > 0:
            dftoi.at[index, 'st_log_lum'] = np.log10(fulltic['lum'][mt])
        if np.isfinite(fulltic['rad'][mt]):
            prev = dftoi.at[index, 'st_rad']
            dftoi.at[index, 'st_rad'] = fulltic['rad'][mt]
            dftoi.at[index, 'rade'] *= fulltic['rad'][mt]/prev
            dftoi.at[index, 'radj'] *= fulltic['rad'][mt]/prev
        dftoi.at[index, 'Kmag'] = fulltic['Kmag'][mt]
        dftoi.at[index, 'Jmag'] = fulltic['Jmag'][mt]

    # fill in missing luminosities with our own calculation
    tmplums = (dftoi['st_rad'] ** 2) * ((dftoi['st_teff'] / 5772) ** 4)
    tofill = ~np.isfinite(dftoi['st_log_lum'])
    dftoi.loc[tofill, 'st_log_lum'] = np.log10(tmplums[tofill])

    # get the semi-major axes and insolations
    iau = (((dftoi['period'] / 365.256)**2) * dftoi['st_mass'])**(1./3.)
    iinsol = (10.**dftoi['st_log_lum']) * (iau**-2)

    # all TESS candidates transit
    dftoi['flag_tran'] = True

    # we should have transit depths for these (all TOIs so far have had
    # depth listed so this doesn't do anything)
    getdep = (dftoi['flag_tran'] & (~np.isfinite(dftoi['tran_depth_ppm'])) &
              np.isfinite(dftoi['rade']) & np.isfinite(dftoi['st_rad']))
    assert getdep.sum() == 0
    tranrat = dftoi['rade']**2 / (dftoi['st_rad'] * 109.1)**2
    dftoi.loc[getdep, 'tran_depth_ppm'] = tranrat[getdep] * 1e6

    # we should have a radius if they gave a depth
    getrad = (dftoi['flag_tran'] & np.isfinite(dftoi['tran_depth_ppm']) &
              (~np.isfinite(dftoi['rade'])) & np.isfinite(dftoi['st_rad']))
    tranrad = np.sqrt((dftoi['tran_depth_ppm']/1e6) * (dftoi['st_rad']**2))
    tranrad *= 109.1
    dftoi.loc[getrad, 'rade'] = tranrad[getrad]
    dftoi.loc[getrad, 'radj'] = tranrad[getrad] / radratio

    # fix these by hand for now
    tranrat = dftoi['rade']**2 / (dftoi['st_rad'] * 109.1)**2
    tranrat *= 1e6
    baddep = ((dftoi['tran_depth_ppm'] < tranrat/3) |
              (dftoi['tran_depth_ppm'] > tranrat*3)) & (dftoi['rade'] < 4)
    dftoi.loc[baddep, 'tran_depth_ppm'] = tranrat[baddep]

    igd = np.isfinite(iinsol)
    dftoi['semi_au'] = iau
    dftoi.loc[igd, 'insol'] = iinsol[igd]

    # these have not been confirmed
    dftoi['year_confirmed'] = np.nan

    # no masses
    dftoi['masse'] = np.nan
    dftoi['massj'] = np.nan

    # all found via transit
    dftoi['discoverymethod'] = 'Transit'

    # do all the tests to make sure things are working, only focusing on
    # candidates since that's all we're adding to output. FPs can be weird.
    canonly = dftoi[newcan]

    # everything has a facility name
    assert (np.array([len(xi) for xi in canonly['facility']]) > 0).all()
    assert (canonly['disposition'] == 'Candidate').all()
    # everything has a host name and URL
    assert (np.array([len(xi) for xi in canonly['hostname']]) > 0).all()
    assert (np.array([len(xi) for xi in canonly['url']]) > 0).all()
    # everything has a planet name
    assert (np.array([len(xi) for xi in canonly['name']]) > 0).all()
    # Input Catalog numbers are correct
    assert (canonly['IC'] > 0).all()
    assert (canonly['discoverymethod'] == 'Transit').all()

    # distances are either NaN or > 1 pc
    assert (~np.isfinite(canonly['distance_pc']) |
            (canonly['distance_pc'] > 1)).all()
    # stellar parameters make sense
    assert (~np.isfinite(canonly['st_rad']) | (canonly['st_rad'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_mass']) |
            (canonly['st_mass'] > 0.07)).all()
    assert (~np.isfinite(canonly['st_teff']) |
            (canonly['st_teff'] > 2700)).all()
    assert (~np.isfinite(canonly['st_log_lum']) |
            ((canonly['st_log_lum'] > -3) & (canonly['st_log_lum'] < 7))).all()
    assert (~np.isfinite(canonly['Kmag']) | (canonly['Kmag'] > 0)).all()
    assert (~np.isfinite(canonly['Jmag']) | (canonly['Jmag'] > 0)).all()

    # RA and Dec are both valid
    assert ((canonly['ra'] >= 0) & (canonly['ra'] <= 360.)).all()
    assert ((canonly['dec'] >= -90) & (canonly['dec'] <= 90.)).all()

    # planet parameters are either NaN or > 0
    assert ((~np.isfinite(canonly['period'])) | (canonly['period'] > 0)).all()
    assert (~np.isfinite(canonly['semi_au']) | (canonly['semi_au'] > 0)).all()
    assert (~np.isfinite(canonly['insol']) | (canonly['insol'] > 0)).all()
    assert (~np.isfinite(canonly['rade']) | (canonly['rade'] > 0)).all()
    assert (~np.isfinite(canonly['radj']) | (canonly['radj'] > 0)).all()
    assert (~np.isfinite(canonly['masse'])).all()
    assert (~np.isfinite(canonly['massj'])).all()
    assert (~np.isfinite(canonly['tran_depth_ppm']) |
            (canonly['tran_depth_ppm'] > 0)).all()
    assert (canonly['tran_dur_hr'] > 0).all()

    # Jup and Earth radii are either defined or not together
    assert np.allclose(np.isfinite(canonly['radj']),
                       np.isfinite(canonly['rade']))
    assert ((~np.isfinite(canonly['rade'])) |
            ((canonly['rade'] / canonly['radj'] > 0.99 * radratio) &
             (canonly['rade'] / canonly['radj'] < 1.01 * radratio))).all()

    # these flags at least have the right number of good values
    assert not canonly['flag_kepler'].any()
    assert not canonly['flag_k2'].any()
    assert canonly['flag_tran'].all()

    # discovery and confirmation years make sense
    assert (canonly['year_discovered'] >= 2018).all()
    assert not np.isfinite(canonly['year_confirmed']).any()

    # add the K2 candidates to our final composite table
    toiadd = dftoi[cols][newcan].copy()
    comp = comp.append(toiadd, verify_integrity=True, ignore_index=True)

    ###################
    # FINAL ADDITIONS #
    ###################

    # create the estimate mass/radius columns
    badm = (np.isfinite(comp['masse']) ^ np.isfinite(comp['massj']))
    assert badm.sum() == 0

    badr = (np.isfinite(comp['rade']) ^ np.isfinite(comp['radj']))
    assert badr.sum() == 0

    getrad = np.isfinite(comp['masse']) & (~np.isfinite(comp['rade']))

    r1 = getrad & (comp['masse'] < 2.04)
    r2 = getrad & (comp['masse'] < 132) & (comp['masse'] >= 2.04)
    r3 = getrad & (comp['masse'] < 26600) & (comp['masse'] >= 132)
    r4 = getrad & (comp['masse'] >= 26600)

    comp.insert(comp.columns.get_loc('rade')+1, 'rade_est', np.nan)
    comp.insert(comp.columns.get_loc('radj')+1, 'radj_est', np.nan)
    comp.loc[r1, 'rade_est'] = 10.**(np.log10(comp.loc[r1, 'masse']) * 0.279 +
                                     0.00346)
    comp.loc[r2, 'rade_est'] = 10.**(np.log10(comp.loc[r2, 'masse']) * 0.589 -
                                     0.0925)
    comp.loc[r3, 'rade_est'] = 10.**(np.log10(comp.loc[r3, 'masse']) * -0.044 +
                                     1.25)
    comp.loc[r4, 'rade_est'] = 10.**(np.log10(comp.loc[r4, 'masse']) * 0.881 -
                                     2.85)

    getmass = np.isfinite(comp['rade']) & (~np.isfinite(comp['masse']))

    m1 = getmass & (comp['rade'] < 1.23)
    m2 = getmass & (comp['rade'] < 11.1) & (comp['rade'] >= 1.23)
    # m3 = getmass & (comp['rade'] < 14.3) & (comp['rade'] >= 11.1)
    m4 = getmass & (comp['rade'] >= 14.3)

    comp.insert(comp.columns.get_loc('masse')+1, 'masse_est', np.nan)
    comp.insert(comp.columns.get_loc('massj')+1, 'massj_est', np.nan)
    comp.loc[m1, 'masse_est'] = 10.**((np.log10(comp.loc[m1, 'rade']) -
                                       0.00346) / 0.2790)
    comp.loc[m2, 'masse_est'] = 10.**((np.log10(comp.loc[m2, 'rade']) +
                                       0.0925) / 0.589)
    comp.loc[m4, 'masse_est'] = 10.**((np.log10(comp.loc[m4, 'rade']) +
                                       2.85) / 0.881)

    # save our final version of the data frame to use in making all the plots
    comp.to_csv('data/exoplots_data.csv', index=False)

    # XXX: check that the planet is smaller than the star in all cases

    return dfcon, dfkoi, dfk2, dftoi, comp


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
    // link.setAttribute('download', 'exoplots_download.txt');
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

deselect = """
const nglyphs = glyphs.length;
var some = 0;
for (let nn = 0; nn < nglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    var legend = legends[nn];
    
    if (!glyph.visible){
        source.selected.indices = [];
    }

    if (source.selected.indices.length == 0){
        glyph.glyph.line_alpha = glyph.nonselection_glyph.line_alpha;
        glyph.glyph.fill_alpha = glyph.nonselection_glyph.fill_alpha;
        glyph.change.emit();
    }
    else {
        glyph.glyph.line_alpha = alphas[nn];
        glyph.glyph.fill_alpha = alphas[nn];
        glyph.change.emit();
        some = 1;
    }
    var newnum = source.selected.indices.length.toLocaleString('en-US');
    newnum = '(' + newnum + ')';
    var newstr = legend.label['value'].replace(/\([\d,]+\)/, newnum);
    legend.label['value'] = newstr;
}
if (some == 0){
    for (let nn = 0; nn < nglyphs; nn++) {
        var glyph = glyphs[nn];
        var source = glyph.data_source;
        var legend = legends[nn];
        
        glyph.glyph.line_alpha = alphas[nn];
        glyph.glyph.fill_alpha = alphas[nn];
        glyph.change.emit();
        
        var newnum = source.data['planet'].length.toLocaleString('en-US');
        newnum = '(' + newnum + ')';
        var newstr = legend.label['value'].replace(/\([\d,]+\)/, newnum);
        legend.label['value'] = newstr;
    }
}
"""

sldeselect = """
const nglyphs = glyphs.length;
var some = 0;
var loop = 0;
while (some == 0 && loop < 3){
    for (let nn = 0; nn < nglyphs; nn++) {
        var glyph = glyphs[nn];
        var source = glyph.data_source;
        var legend = legends[nn];
        
        if (!glyph.visible){
            source.selected.indices = [];
        }
    
        if (source.selected.indices.length == 0){
            glyph.glyph.line_alpha = glyph.nonselection_glyph.line_alpha;
            glyph.glyph.fill_alpha = glyph.nonselection_glyph.fill_alpha;
            glyph.change.emit();
        }
        else {
            glyph.glyph.line_alpha = alphas[nn];
            glyph.glyph.fill_alpha = alphas[nn];
            glyph.change.emit();
            some = 1;
        }
        var newnum = source.selected.indices.length.toLocaleString('en-US');
        newnum = '(' + newnum + ')';
        var newstr = legend.label['value'].replace(/\([\d,]+\)/, newnum);
        legend.label['value'] = newstr;
    }
    if (some == 0){
        var minyr = slider.value[0];
        var maxyr = slider.value[1];
        for (let nn = 0; nn < nglyphs; nn++) {
            var glyph = glyphs[nn];
            var source = glyph.data_source;
            var selected = [];
            for (let ii = 0; ii < source.data['planet'].length; ii++) {
                if (source.data['year'][ii] >= minyr && 
                        source.data['year'][ii] <= maxyr){
                    selected.push(ii);
                }
            }
            if (glyph.visible){
                source.selected.indices = selected;
            }
            else {
                source.selected.indices = [];
            }
            source.change.emit();
        }
    }
    loop = loop + 1;
}
"""

reset = """
const nglyphs = glyphs.length;
for (let nn = 0; nn < nglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    var legend = legends[nn];
    
    glyph.glyph.line_alpha = alphas[nn];
    glyph.glyph.fill_alpha = alphas[nn];
    glyph.change.emit();
    
    var newnum = source.data['planet'].length.toLocaleString('en-US');
    newnum = '(' + newnum + ')';
    var newstr = legend.label['value'].replace(/\([\d,]+\)/, newnum);
    legend.label['value'] = newstr;
}
"""

yearselect = """
const mglyphs = glyphs.length;
var minyr = slider.value[0];
var maxyr = slider.value[1];
for (let nn = 0; nn < mglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    var selected = [];
    for (let ii = 0; ii < source.data['planet'].length; ii++) {
        if (source.data['year'][ii] >= minyr && 
                source.data['year'][ii] <= maxyr){
            selected.push(ii);
        }
    }
    if (glyph.visible){
        source.selected.indices = selected;
    }
    else {
        source.selected.indices = [];
    }
    source.change.emit();
}

label.text = minyr + '\u2013' + maxyr;
label.change.emit();

""" + sldeselect

unselect = """
const mglyphs = glyphs.length;
for (let nn = 0; nn < mglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    source.selected.indices = [];
    source.change.emit();
}
""" + deselect

sliderselect = """
const mglyphs = glyphs.length;
var minyr = slider.value[0];
var maxyr = slider.value[1];
for (let nn = 0; nn < mglyphs; nn++) {
    var glyph = glyphs[nn];
    var source = glyph.data_source;
    var legend = legends[nn];
    
    if (!glyph.visible){
        source.selected.indices = [];
    }
    else {
        var selected = [];
        for (let ii = 0; ii < source.selected.indices.length; ii++) {
            var iyr = source.data['year'][source.selected.indices[ii]];
            if (iyr >= minyr &&  iyr <= maxyr){
                selected.push(source.selected.indices[ii]);
            }
        }
        source.selected.indices = selected;
    }
    source.change.emit();
}

""" + sldeselect

nowvis = """
var tmp = [];
for (var i=0;i<cb_obj.data_source.data['year'].length;i++) {
  tmp.push(i);
}
cb_obj.data_source.selected.indices=tmp;

""" + sliderselect

playpause = """

async function advance(slider, button) {
    var i = 0;
    var mpause = 0;
    while (button.active){
        i = i + 1;
        var vals = slider.value;
        var nextval = vals[1] + 1;
        if (nextval > slider.end){
            if (mpause > 0){
                nextval = vals[0];
                mpause = 0;
            }
            else{
                nextval = nextval - 1;
                mpause = mpause + 1;
            }
        }
        slider.value = [vals[0], nextval];
        await new Promise(r => setTimeout(r, 1000));
    }
}

async function startloop(slider, button) {
    const result = await advance(slider, button);    
}

if (cb_obj.active){
    cb_obj.label = "\u2759\u2759 Pause";
    startloop(slider, cb_obj);
}
else {
    cb_obj.label = "\u25b6 Play";
}
"""
