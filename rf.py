import pandas as pd
from datetime import datetime
import asyncio
from archiver import Archiver

PV = ['RF-Gen:GeneralFreq-RB']

class RF:
    filepath: str
    data_source: str
    timespam: dict

    def __init__(self, data_source, timespam: dict = None, filepath: str = None) -> None:
        self.data_source = data_source
        self.timespam = timespam
        self.filepath = filepath
        self.load_data()
    
    def get_local_data(self) -> pd.DataFrame:
        data = pd.read_excel(self.filepath)
        # data.index = pd.DatetimeIndex([datetime.strptime(ts_str, "%d.%m.%y %H:%M") for ts_str in data['datetime']])
        data.index = pd.DatetimeIndex(data['datetime'])
        data.drop(columns=['datetime'], inplace=True)
        return data

    def get_data_from_archiver(self) -> pd.DataFrame:
        data = asyncio.run(Archiver.request_data(PV, self.timespam, 1))
        return data

    def load_data(self) -> None:
        # extracting data from specified source
        if (self.data_source == 'local'):
            self.data = self.get_local_data()
        elif (self.data_source == 'archiver'):
            self.data = self.get_data_from_archiver()
        self.data = self.data - self.data.iloc[0,:]

    def get_data(self) -> pd.DataFrame:
        return self.data