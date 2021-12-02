import geodesics

from typing import Dict, List
from pandas.core.arrays import boolean
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

CARDINAL_GP = {
    'N': (-22.807226196465898, -47.0524966686184),\
    'NNE': (-22.807307677130677, -47.05222194447693),\
    'NE': (-22.80743275253845, -47.052023830957914),\
    'ENE': (-22.80758523686219, -47.0519203978644),\
    'E': (-22.807830425529488, -47.051824890303436),\
    'ESE': (-22.80804735840874, -47.05185104021198),\
    'SE': (-22.80819578477707, -47.051911540568284),\
    'SSE': (-22.808381365296672, -47.05211537588159),\
    'S': (-22.808501470367997, -47.05252878392342),\
    'SSW': (-22.80841515272454, -47.05287061280038),\
    'SW': (-22.808272924957787, -47.053052055794396),\
    'WSW': (-22.80807387004189, -47.05317285774496),\
    'W': (-22.807822432874918, -47.05320628229365),\
    'WNW': (-22.807602436867075, -47.0531658788379),\
    'NW': (-22.807379271552573, -47.05300710605838),\
    'NNW': (-22.80723725576768, -47.052729913577195)
}

MAPPING_CARDINAL_SECTOR = {
    'E': 'Q1P2',
    'ENE': 'Q1P4',
    'NE': 'Q1P6',
    'NNE': 'Q1P8',
    'N': 'Q1P10',
    'NNW': 'Q2P4',
    'NW': 'Q2P6',
    'WNW': 'Q2P8',
    'W': 'Q3P2',
    'WSW': 'Q3P4',
    'SW': 'Q3P6',
    'SSW': 'Q3P8',
    'S': 'Q4P2',
    'SSE': 'Q4P4',
    'SE': 'Q4P6',
    'ESE': 'Q4P8'
}

class Tides:
    data: Dict = {}
    coords: List
    coord_list: Dict
    num_of_days: int
    mapping_needed: boolean

    def __init__(self, point_names: List, mapping_needed: boolean = False) -> None:
        self.coords = point_names
        self.mapping_needed = mapping_needed
        # initializing data structure with empty DataFrames
        for coord in self.coords:
            self.data[coord] = pd.DataFrame()
        # initializing coordinate list with predefined latitude and longitude values
        self.coord_list = CARDINAL_GP
    
    def generate_tides(self, timespam: dict) -> None:
        # setting datelist according to method parameters
        datelist = pd.date_range(start=datetime(timespam['init']['year'], timespam['init']['month'], timespam['init']['day']),\
                                 end=datetime(timespam['end']['year'], timespam['end']['month'], timespam['end']['day']), freq='D').tolist()
        # setting number of days for future use
        # self.num_of_days =  (datelist[-1] - datelist[0]).days
        self.num_of_days =  len(datelist)
        # generating earth tides according to the timespam and position reference
        for date in datelist:
            for cardinal in self.coord_list:
                _, tides = geodesics.solid(date.year, date.month, date.day, self.coord_list[cardinal][0], self.coord_list[cardinal][1], 1)
                dummy = pd.DataFrame(tides, columns=['North', 'East', 'Up'])
                point_name = cardinal if not self.mapping_needed else MAPPING_CARDINAL_SECTOR[cardinal]
                self.data[point_name] = self.data[point_name].append(dummy, ignore_index=True)

        # correcting datetime index
        index = pd.date_range(start=datetime(timespam['init']['year'], timespam['init']['month'], timespam['init']['day']),\
                              end=datetime(timespam['end']['year'], timespam['end']['month'], timespam['end']['day']) + timedelta(days=1),\
                              freq='T', closed='left')

        for d in self.data:
            self.data[d].index = index
            # setting first record as 0 in tides series
            self.data[d] = self.data[d] - self.data[d].iloc[0,:]
            # filtering data to contemplate exactly the timespam
            datetime_init = datetime(timespam['init']['year'], timespam['init']["month"], timespam['init']["day"], timespam['init']["hour"], timespam['init']["minute"], timespam['init']["second"])
            datetime_end = datetime(timespam['end']['year'], timespam['end']["month"], timespam['end']["day"], timespam['end']["hour"], timespam['end']["minute"], timespam['end']["second"])
            self.data[d] = self.data[d][(self.data[d].index >= datetime_init) & (self.data[d].index <= datetime_end)]

    def plot_tide(self, position:str = None) -> None:
        if not position:
            for coord in self.coords:
                self.data[coord].plot()
            plt.show()
            return

        if position in self.coords:
            self.data[position].plot()
            plt.show()

    def get_timeseries(self) -> dict:
        return self.data

    def get_num_of_days(self) -> int:
        return self.num_of_days

    def get_num_of_records(self) -> int:
        return len(self.data[self.coords[0]])

    def get_datetime_index(self) -> pd.Index:
        return self.data[self.coords[0]].index