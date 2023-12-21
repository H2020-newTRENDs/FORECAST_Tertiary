import logging
import sqlite3

import pandas as pd
import yaml

from fecommon.fe_data_structure import find_year_columns
from fecommon.fe_process import ForecastElementProcess

log = logging.getLogger(__name__)
log_summary = logging.getLogger('Process Summary')


class Coworking(ForecastElementProcess):
    mandatory_conf = ["dbfile", "ID_Scenario", "share_coworking", "factor_ict_remote", ]

    def process(self, process):
        """
        This pre-process increases the ICT workload in the FORECAST model, caused by co-working spaces and
        work from home (WFH) activity. It doesn't change the floor area in tertiary sector, as they are supposed to be
        the same as in on-site offices, but it increases the ICT work load:

        It uses these parameters:
        * share_coworking: share of workers in co-working spaces, relative to WFH-worker:
        Example: WFH rises from 30% to 60% in course of the years, and share_coworking = 20% => coworking rises
        from 6% (30% * 20%) to 12% (30% * 20%)
        [onsite goes down from 100%-30%-6% = 64% to 28% (100%-60%-12%) ]
        * factor_ict_remote: factor of additional workload in co-working and home office compared to on-site office, e.g. 1.1

        this process will increase the ScenarioData_EnServDriver 4 for coworking AND home-office:
        example with the given values above for year 2050:
        new_value = old_value * (12% + 60%) * 1.1

        This method is called as a pre-process in the course of the FORECAST simulation.
        Its parameters (dict "process") are defined in the configuration file of the simulation.

        :param process:
        :return:
        """
        log_summary.info(f'<<<STARTING>>> {__name__}: {process["id"]}')
        self.check_input(process)
        sqliteFullPath = process['dbfile']
        id_scenario = process['ID_Scenario']

        con = sqlite3.connect(sqliteFullPath)

        # get DB table ScenarioData_EnServDriver, ID_EnergyService = 4
        query = f"SELECT * FROM ScenarioData_EnServDriver WHERE ID_EnergyService = 4 and ID_Scenario = {id_scenario} and ID_Sector=3"
        df_enserv = pd.read_sql(query, con)
        query = f"SELECT * FROM ScenarioData_EnServDriver"
        df_enserv_old = pd.read_sql(query, con)
        df_enserv_old.set_index(['ID_Scenario', 'ID_Country', 'ID_Subsector', 'ID_Sector', 'ID_EnergyService'],
                                inplace=True)

        # get DB table ScenarioData_ShareEmployment_WFH, ID_Kind = 1
        query = f"SELECT * FROM ScenarioData_ShareEmployment_WFH WHERE ID_Kind = 1 and ID_Scenario = {id_scenario}  and ID_Sector=3"
        df_emplshare = pd.read_sql(query, con)

        yr_col = list(set(find_year_columns(df_enserv, prefix='Y')).difference(['Y1990', 'Y1991']))
        df_enserv.set_index(['ID_Scenario', 'ID_Country', 'ID_Subsector', 'ID_Sector', 'ID_EnergyService'],
                            inplace=True)
        df_emplshare['ID_Scenario'] = id_scenario
        df_emplshare['ID_Sector'] = 3
        df_emplshare['ID_EnergyService'] = 4

        df_emplshare.set_index(['ID_Scenario', 'ID_Country', 'ID_Subsector', 'ID_Sector', 'ID_EnergyService'],
                               inplace=True)

        df_enserv_new = df_enserv.copy()
        df_enserv_new.loc[:, yr_col] = df_enserv.loc[:, yr_col] * (
                1 + df_emplshare.loc[:, yr_col] * (1 + process['share_coworking']) * (
                process['factor_ict_remote'] - 1))

        df_enserv_new.pop('ID')
        df_enserv_new['Source'] = "coworking.py"

        df_old = df_enserv_old
        df_res = pd.concat([df_enserv_new, df_old[~df_old.index.isin(df_enserv_new.index)]]).reset_index()
        res = df_res.to_sql("ScenarioData_EnServDriver", con=con, if_exists='replace', index=False)
        con.close()


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
        "id": 1,
        "dbfile": "NewTrends/working_temp/NewTrends-Tertiary_WIP.sqlite",
        "ID_Scenario": 520002,
        "share_coworking": 0.2,
        "factor_ict_remote": 1.1
    }
    Coworking().process(p)
