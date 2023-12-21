import logging
import sqlite3

import numpy as np
import pandas as pd
import yaml

from fecommon.fe_data_structure import find_year_columns, order_columns, transform_year_columns_to_int
from fecommon.fe_process import ForecastElementProcess

log = logging.getLogger(__name__)
log_summary = logging.getLogger('Process Summary')

country_mapping = {"European Union - 27 countries (from 2020)": 0,
                               "Belgium": 2,
                                "Bulgaria": 27,
                                "Czechia": 4,
                                "Denmark": 5,
                                "Germany (until 1990 former territory of the FRG)": 9,
                                "Estonia": 6,
                                "Ireland": 12,
                                "Greece": 10,
                                "Spain": 23,
                                "France": 8,
                                "Croatia": 32,
                                "Italy": 13,
                                "Cyprus": 3,
                                "Latvia": 14,
                                "Lithuania": 15,
                                "Luxembourg": 16,
                                "Hungary": 11,
                                "Malta": 17,
                                "Netherlands": 18,
                                "Austria": 1,
                                "Poland": 19,
                                "Portugal": 20,
                                "Romania": 26,
                                "Slovenia": 22,
                                "Slovakia": 21,
                                "Finland": 7,
                                "Sweden": 24,
                                "Iceland": 33,
                                "Norway": 29,
                                "United Kingdom": 25
            }

subsector_mapping = {'Wholesale or retail': 1,
                     'Real estate': 4,
                     'Business service': 8,
                     'Financial or insurance': 4,
                     'Education': 6,
                     'Health': 5,
                     'Public administration': 7,
                     'Information or communication': 3,
                     'Other service': 8,
                     'Transport and storage': 3,
                     'Resturant, hotel and coffees': 2
                     }

office_floor_weights = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
office_empl_weights = {1: 0.35, 2: 0.05, 3: 0.85, 4: 0.85, 5: 0.05, 6: 0.20, 7: 0.90, 8: 0.70}

def polish_country_column(df):
    # rename the country column
    cols = list(df.columns)
    cols[0] = 'country'
    df.columns = cols

    # filter countries and replace names by ids
    df['country'] = df['country'].apply(lambda x: country_mapping[x] if x in country_mapping.keys() else np.nan)
    df = df[df['country'] >= 0]
    return df

def polish_subsector_column(df):
    subsector_mapping = {'Working in business services': [8],
         'Working in financial or insurance activities': [4],
         'Working in information or communication': [3],
         'Working in other service activities': [8],
         'Working in public administration, defence, education, human health or social work activities': [5, 6, 7],
         'Working in real estate activities': [8],
         'Working in wholesale or retail trade, transport, accommodation or food service activities': [1, 2, 3]}

    countries = list(set(df['ID_Country']))
    df2a = pd.DataFrame({'ID_Country': countries})
    df2b = pd.DataFrame({'ID_Subsector': list(range(1, 9))})
    df2 = df2a.merge(df2b, how='cross')
    df2['Unit'] = ''
    df2['ID_Sector'] = 3
    df2[2018] = 0
    df2.set_index(['ID_Country', 'ID_Subsector'], inplace=True)
    df.set_index(['ID_Country', 'ind_type'], inplace=True)

    # manual fix for 2 missing values in statistic
    df.loc[(14, 'Working in business services'),:] = df.loc[(14, 'Working in real estate activities'),:]
    df.loc[(14, 'Working in business services'), 2018] = (df.loc[(14, 'Working in real estate activities'), 2018] + df.loc[(14, 'Working in other service activities'), 2018]) / 2
    df.loc[(32, 'Working in real estate activities'), :] = df.loc[(32, 'Working in business services'), :]
    df.loc[(32, 'Working in real estate activities'), 2018] = (df.loc[(32, 'Working in business services'), 2018] + df.loc[(32, 'Working in other service activities'), 2018]) / 2


    for cat in subsector_mapping.keys():
        for id_ss in subsector_mapping[cat]:
            factor = 1
            if cat in ['Working in business services', 'Working in other service activities', 'Working in real estate activities']:
                factor = 1/3
            if cat == 'Working in wholesale or retail trade, transport, accommodation or food service activities' and id_ss == 3:
                factor = 0
            for c in countries:
                if (c, cat) not in df.index:
                    log.warning(f"{(c, cat)} not found")
                    continue
                df2.loc[(c, id_ss), 2018] += df.loc[(c, cat), 2018] / 100 * factor

    df2.reset_index(inplace=True)
    return df2

def fill_gaps_2022(df):
    # fill gaps based on EU data
    pre_covid_years = list(range(2012, 2020))
    df['average'] = df.apply(lambda x: x[pre_covid_years].mean(), axis=1)
    for yr in [2020, 2021]:
        df[yr] = df.apply(lambda x: x[yr] if not np.isnan(x[yr])
                else df.loc[0, yr] / df.loc[0, 'average'] * x['average'], axis=1)
    df.drop('average', axis=1, inplace=True)
    return df


def fill_gaps_2018(df):
    eu_set = df[df['ID_Country'] == 0]
    if len(eu_set) == 0:
        log.warning(f'no eu data found, "fill gaps 2018" skipped')
        return df

    df.set_index(['ID_Country', 'ID_Subsector'], inplace=True)

    # check if whole countries are missing
    ids = []
    for cid in country_mapping.values():
        if cid == 0:
            continue
        if cid not in set([x[0] for x in df.index]):
            ids += [(cid, x) for x in range(1,9)]


    for id in ids:
        df.loc[id, 2018] = df.loc[(0, id[1]), 2018]

    # check if some values are nan
    ids = df[df[2018].isna()].index

    for id in ids:
        df.loc[id, 2018] = df.loc[(0, id[1]), 2018]

    df.reset_index(inplace=True)
    return df

def get_wfh_factor(df):
    """
    the wfh factor says how much higher/lower the wfh value of this subsector is compared to the average in this country
    e.g. wfh value in Austria over all subsectors: 20% (fictive number) and the wfh factor for Austria and finance subsector is 1.1
    then wfh value in Austria for Finance is 22% = 20% * 1.1
    here, the wfh factors are calculated based on the 2018 statistic data for all subsectors.
    it is calculated by wfh_value(country x, subsector y) / mean(wfh_value(countryx , all subsectors y1, y2,...))
    :param df:
    :return:
    """
    for c in set(df['ID_Country']):
        df_temp = df[df['ID_Country'] == c]
        nr = len(df_temp)
        country_sum = df_temp.loc[:, 2018].sum()
        for s in set(df['ID_Subsector']):
            log.debug(f"c {c}, s {s}")
            id = df_temp[df_temp['ID_Subsector'] == s].index[0]
            subsector_sum = df_temp.loc[df_temp['ID_Subsector'] == s, 2018].sum()
            df.loc[id, 'wfh_factor'] = subsector_sum / country_sum * nr
    return df


def do_trajectory_country(df, factors=(1, 2)):
    # factors say how much more wfh in 2030 and 2050 compared to 2021
    # time series df says what share sometimes or usually worked from home ("never" was inverted)
    # trajectory until 2030 and 2050
    df[2022] = df.apply(lambda x: x[-4:].mean(), axis=1)
    df[2030] = df[2021] * factors[0]
    df[2050] = df[2021] * factors[1]
    minv = 0
    maxv = 1

    cols = find_year_columns(df)

    yrs = list(range(2023, 2030))
    for yr in yrs:
        df[yr] = df[yr - 1] + (df[yrs[-1] + 1] - df[yrs[0] - 1]) / (len(yrs) + 1)

    yrs = list(range(2031, 2050))
    for yr in yrs:
        saturation_factor = 1
        # value in 2050 should be reached but on s-curve
        # added a second step below that increases all gradients by constant offset in order to reach target again
        if yr > 2040:
            saturation_factor = 1 - (yr - 2040) / 10
        df[yr] = df[yr - 1] + saturation_factor * (df[yrs[-1] + 1] - df[yrs[0] - 1]) / (len(yrs) + 1)

    offset = df[2050] - df[2049]
    i = 0
    for yr in yrs:
        i += 1
        df[yr] = df[yr] + offset * i / 19

    df[cols] = df[cols].clip(lower=minv, upper=maxv)
    df = order_columns(df)
    return df


def do_trajectory_subsector(df, df2):
    # important: parameter "never" is forecasted
    # folding the df by using merge "cross"
    # merge both df, but df only has id of countries, and df2 has countries and subsectors -> df3: with countries and subsectors
    # perform an apply that can work on each row and does calculation analog to excel

    df_res = df.merge(df2.drop(columns=[2018]), how='cross')
    df_res.set_index(['ID_Country_x', 'ID_Country_y', 'ID_Subsector'], inplace=True)
    ids = [i for i in df_res.index if i[0] == i[1]]
    df_res = df_res.loc[ids, :]
    df_res = df_res.reset_index()
    df_res.drop(columns=[col for col in df_res.columns if isinstance(col, str) and col.endswith("_y")], inplace=True)
    df_res.rename(columns={col: col[:-2] for col in df_res.columns if isinstance(col, str) and col.endswith("_x")}, inplace=True)
    for idx in df_res.index:
        for yr in range(1992, 2051):
            value = df_res.loc[idx, yr] * df_res.loc[idx, 'wfh_factor']
            # if value < 0:
            #     log.warning('')
            max_empl_share = office_empl_weights[df_res.loc[idx, 'ID_Subsector']]
            value = min(max(0, value), max_empl_share)
            df_res.loc[idx, yr] = value
    df_res = order_columns(df_res)
    return df_res

def get_specifc_floor_area_factor(share):
    """
    if share is low (20% or below), specific floor area (spfa) is unchanged (=1)
    if share is high (90% or above), spFa is 0.2
    if inbetween, linear
    :param spfa: specific floor area
    :param share of working from home
    :return: specific floor area in tertiary sector
    """
    if share <= 0.2:
        return 1
    elif share >= 0.9:
        return 0.2
    else:
        x_dist = (share - 0.2) / (0.9 - 0.2)
        y = 1 - (1 - 0.2) * x_dist
        return y



def calibrate3(df_spfa_org, df_spfa_new, calib_year, diff_year):
    '''
    in original spFA data, work from home is already included to some extent
    the wfh we want to forecast is the additional wfh
    there are 2 different df_spFA: idea is to apply difference of df_spfa_new to df_spfa_org
    and to keep df_spfa_org in the past
    :param df_spfa_org:
    :param df_spfa_new:
    :param calib_year:
    :param diff_year:
    :return:
    '''
    years = find_year_columns(df_spfa_org, 'Y')
    df_spfa_new[f'Y{calib_year}_org'] = df_spfa_org[f'Y{calib_year}']
    delta = df_spfa_new[years + [f'Y{calib_year}_org']].apply(lambda x: x + x[f'Y{calib_year}_org'] - x[f'Y{calib_year}'], axis=1)
    yrs2 = [f'Y{y}' for y in list(range(int(years[0][1:]), diff_year))]
    delta.loc[:, yrs2] = df_spfa_org.loc[:, yrs2]
    df_spfa_new[years] = delta[years]
    df_spfa_new.drop(columns=[f'Y{calib_year}_org'], inplace=True)
    return df_spfa_new


def weigh_subsectors(df_spfa_org, df_spfa_new):
    """
    the weights say how many offices floor space is in the total floor space of this subsector,
    (and that can be potentially moved to home office)
    :param df_spfa_org:
    :param df_spfa_new:
    :return:
    """

    df_spfa_org.set_index(['ID_Country', 'ID_Subsector'], inplace=True)
    df_spfa_new.set_index(['ID_Country', 'ID_Subsector'], inplace=True)
    years = find_year_columns(df_spfa_org, 'Y')
    for idx in df_spfa_new.index:
        ss = idx[1]
        df_spfa_new.loc[idx, years] = ((1 - office_floor_weights[ss]) * df_spfa_org.loc[idx, years] +
                                            office_floor_weights[ss] * df_spfa_new.loc[idx, years])

    df_spfa_org.reset_index(inplace=True)
    df_spfa_new.reset_index(inplace=True)
    return df_spfa_new


def invert_wfh(df):
    years = find_year_columns(df)
    df[years] = df[years].apply(lambda x: 1-x)
    return df


def get_wfh_time_series(factors, stat_lfsa_ehomp, stat_isoc_iw_hem, verbose=False):
    df = pd.read_csv(stat_lfsa_ehomp)
    df = transform_year_columns_to_int(df, prefix='Y')
    df = fill_gaps_2022(df)
    df = invert_wfh(df)
    df = do_trajectory_country(df, factors=factors)
    if verbose:
        df.to_csv(f'exports/trajectory_country_export_{factors[0]}_{factors[1]}.csv')

    # df2
    df2 = pd.read_csv(stat_isoc_iw_hem)
    df2 = transform_year_columns_to_int(df2, prefix='Y')
    df2 = polish_subsector_column(df2)
    df2 = fill_gaps_2018(df2)
    df2 = invert_wfh(df2)
    df2 = get_wfh_factor(df2)
    df2 = do_trajectory_subsector(df, df2)
    if verbose:
        df2.to_csv(f'exports/trajectory_subsector_export_{factors[0]}_{factors[1]}.csv')
    return df2


def csv_export(process):
    sqliteFullPath = process['database']
    con = sqlite3.connect(sqliteFullPath)
    ids = [process['spFA_org_ID_Scenario']] + [x['ID_Scenario'] for x in process['spFA_new']]
    where_constraint = ' WHERE ' + ' OR '.join([f'ID_Scenario={id}' for id in ids])
    query = f'SELECT * FROM ScenarioData_spFA {where_constraint};'
    df = pd.read_sql(query, con=con)
    df.to_csv('exports/test_export.csv')




def save_wfh_share_in_db(con, id_scenario, df_wfh):
    """
    this method doesn't write to ScenarioData_ShareEmploymentByKind
    but to new generated table ScenarioData_ShareEmployment_WFH

    ID_Kind
    0: normal/conventional
    1: wfh
    2: e-commerce
    3: co-working space

    table ScenarioData_ShareEmploymentByKind:
    ID_Scenario  integer,
    ID_Country   integer,
    ID_Sector    integer,
    ID_Subsector integer,
    Unit         text,
    Source       text,
    Comment      text,
    Y1990        real ...

    :param con:
    :param id_scenario:
    :param df_wfh:
    :return:
    """
    # add right ids
    df_export = df_wfh.copy()
    df_export.pop('wfh_factor')
    df_export['ID_Country'] = df_export['ID_Country'].apply(lambda x: int(x))
    df_export['Unit'] = '-'
    df_export['Source'] = 'auto generated by teleworking.py for newTrends'
    df_export['Comment'] = ''
    df_export['ID_Kind'] = 1
    df_export['ID_Scenario'] = id_scenario

    df_export.rename(columns={x: f'Y{x}' for x in find_year_columns(df_export)}, inplace=True)
    df_export.to_sql("ScenarioData_ShareEmployment_WFH", con=con, if_exists='append', index=False)

    log.info('save_wfh_share_in_db: done')



class Teleworking(ForecastElementProcess):
    mandatory_conf = ['stat_lfsa_ehomp', 'stat_isoc_iw_hem']

    def process(self, process):
            """
            This pre-process adjusts the data regarding teleworking.
            It reads in the statistical information about teleworking (share of employees that work from home),
            translates the qualitative statements into quantitative,
            completes the matrix for missing years and subsectors (statistics show only details for 2018),
            trajects the statistic up to 2050 for the total sum,
            calculates split, e.g ict = 5% of sum of class2 value of all branches,
            makes assumptions about its impact on floor area, utilization rate (eg h/day),
            and installed power regarding the different energy services

            In order to also model co-working areas, use the process "coworking" as well.

            This method is called as a pre-process in the course of the FORECAST simulation.
            Its parameters (dict "process") are defined in the configuration file of the simulation.

            :param process:
            :return:
            """
            log_summary.info(f'<<<STARTING>>> {__name__}: {process["id"]}')
            self.check_input(process)

            sqliteFullPath = process['database']
            verbose = process['verbose'] if 'verbose' in process.keys() else False
            spFA_org_ID_Scenario = process['spFA_org_ID_Scenario']
            yearpart = ','.join([f'Y{y}' for y in range(2012, 2051)])

            try:
                con = sqlite3.connect(sqliteFullPath)
                query = f'SELECT ID_Country, ID_Sector, ID_Subsector, ID_Kind, Unit, {yearpart} ' \
                        f'FROM ScenarioData_spFA WHERE ID_Scenario={spFA_org_ID_Scenario}'
                df_spfa_org = pd.read_sql(query, con=con)
            except Exception as e:
                msg = f"problems to load {sqliteFullPath}"
                log.error(msg)
                raise Exception(msg)

            for new_spfa_def in process['spFA_new']:
                factors = (new_spfa_def['factor_2030'], new_spfa_def['factor_2050'])
                df_wfh = get_wfh_time_series(factors, process['stat_lfsa_ehomp'] , process['stat_isoc_iw_hem'] , verbose)
                save_wfh_share_in_db(con, new_spfa_def['ID_Scenario'], df_wfh)
                df_spfa_new = df_spfa_org.copy()
                df_spfa_new = df_spfa_new.set_index(['ID_Country', 'ID_Subsector'])
                df_wfh = df_wfh.set_index(['ID_Country', 'ID_Subsector'])
                for idx in df_spfa_new.index:
                    for yr in range(2012, 2051):
                        id2 = (idx[0], idx[1])
                        if id2 not in df_wfh.index:
                            id2 = (0, idx[1])
                        value = df_wfh.loc[id2, [yr]]
                        try:
                            df_spfa_new.loc[idx, f"Y{yr}"] *= get_specifc_floor_area_factor(value.values[0])
                        except Exception as e:
                            log.error(e)
                df_spfa_new.reset_index(inplace=True)
                df_spfa_new["ID_Scenario"] = new_spfa_def['ID_Scenario']
                df_spfa_new["Unit"] = df_spfa_org['Unit'][0]
                df_spfa_new["Source"] = "TEP Energy"
                df_spfa_new["Comment"] = "auto generated2 by teleworking.py"
                df_spfa_new["ID"] = list(range(len(df_spfa_new)))
                calibrate3(df_spfa_org, df_spfa_new, 2019, 2019)
                weigh_subsectors(df_spfa_org, df_spfa_new)

                # remove all rows with ID_Sceanrio
                query = f'DELETE FROM ScenarioData_spFA WHERE ID_Scenario={new_spfa_def["ID_Scenario"]}'
                res = con.execute(query)
                con.commit()
                df_spfa_new.to_sql("ScenarioData_spFA", con=con, if_exists='append', index=False)

                log.info(f"done with ID_Scenario={new_spfa_def['ID_Scenario']}")
            log.info('done')
            if verbose:
                csv_export(process)

if __name__ == '__main__':
    """
    this section is only used for development and debugging. In the course of a regular simulation, 
    the method "process" of the class above is called directly, with the parameters defined in the configuration file 
    of the simulation.
    """
    with open('logging.conf') as f:
        d = yaml.safe_load(f)
        logging.config.dictConfig(d)

    log = logging.getLogger()
    p = {
        'database': 'C:/Users/msteck.TEP-WKS-NT006/TEP Energy/OneDrive - TEP/_projects/1201_NewTRENDS/_working/WP6_focusStudy_Digitalization/modelling/database/NewTrends-Tertiary_WIP.sqlite',
        'spFA_org_ID_Scenario': '520002',
        'spFA_new': [
            {'ID_Scenario': '520010', 'factor_2030': 1, 'factor_2050': 1.5},
            {'ID_Scenario': '520020', 'factor_2030': 1.2, 'factor_2050': 2.5},
            {'ID_Scenario': '520030', 'factor_2030': 1.5, 'factor_2050': 1000},
        ]
    }
    Teleworking().process(p)