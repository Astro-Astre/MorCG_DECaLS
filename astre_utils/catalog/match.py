import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord


def match(df_1: pd.DataFrame, df_2: pd.DataFrame, pixel: int, df1_name: list, resolution=0.262) -> pd.DataFrame:
    """
    match two catalog.
    suggestion: df_1 is the real coordinates locally, df_2 is a DataFrame with something wrong in coordinates
    :param df_1:
    :param df_2:
    :param pixel:
    :param df1_name:
    :param resolution: telescope resolution , arcsec/pixel
    :return: using coord in df_1
    """
    sc1 = SkyCoord(ra=df_1.ra * u.degree, dec=df_1.dec * u.degree)
    sc2 = SkyCoord(ra=df_2.ra * u.degree, dec=df_2.dec * u.degree)
    idx, d2d, d3d = sc1.match_to_catalog_sky(sc2)
    distance_idx = d2d < (pixel * resolution * u.arcsec)

    sc1_matches = df_1.iloc[distance_idx]
    sc2_matches = df_2.iloc[idx[distance_idx]]

    test = sc1_matches.loc[:].rename(columns={"ra": "%s" % df1_name[0], "dec": "%s" % df1_name[1]})
    test.insert(0, 'ID', range(len(test)))
    sc2_matches.insert(0, 'ID', range(len(sc2_matches)))
    new_df = pd.merge(test, sc2_matches, how="inner", on=["ID"])
    return new_df.drop("ID", axis=1)
