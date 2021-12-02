import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import asyncio
from functools import partial

from archiver import Archiver
from perimeter import Perimeter
from plot import LegendPickablePlot

MAPPING_CARDINAL_SECTOR = {
    "concrete":
    {
        17: "E",
        16: "ENE",
        15: "NE",
        13: "NNE",
        12: "N",
        11: "NNW",
        10: "NW",
        8: "WNW",
        7: "W",
        6: "WSW",
        5: "SW",
        3: "SSW",
        2: "S", 
        1: "SSE",
        20: "SE",
        18: "ESE"
    },
    "hls":
    {
        17: "E",
        16: "ENE",
        15: "NE",
        13: "NNE",
        12: "N",
        11: "NNW",
        10: "NW",
        8: "WNW",
        7: "W",
        6: "WSW",
        5: "SW",
        3: "SSW",
        2: "S", 
        1: "SSE",
        20: "SE",
        18: "ESE"
    },
    "fancoil":
    {
        "16-18": "E",
        "16-18": "ENE",
        "14-16": "NE",
        "12-14": "NNE",
        "12-14": "N",
        "10-12": "NNW",
        "10-12": "NW",
        "08-10": "WNW",
        "06-08": "W",
        "06-08": "WSW",
        "04-06": "SW",
        "02-04": "SSW",
        "02-04": "S", 
        "20-22": "SSE",
        "20-22": "SE",
        "18-20": "ESE"
    }
}

PVS = {
    "concrete": {
        "comb1": {
            12: ['TU-12S:SS-Concrete-5AP:Temp-Mon'],
            13: ['TU-13S:SS-Concrete-5AP:Temp-Mon'],
            # ['TU-14S:SS-Concrete-5AP:Temp-Mon'],
            15: ['TU-15S:SS-Concrete-5AN:Temp-Mon', 'TU-15S:SS-Concrete-5CN:Temp-Mon'],
            16: ['TU-16S:SS-Concrete-5AP:Temp-Mon'],
            17: ['TU-17S:SS-Concrete-5AP:Temp-Mon'],
            18: ['TU-18S:SS-Concrete-5AP:Temp-Mon'],
            # ['TU-19S:SS-Concrete-5AP:Temp-Mon'],
            20: ['TU-20S:SS-Concrete-5AN:Temp-Mon', 'TU-20S:SS-Concrete-5CN:Temp-Mon'],
            1: ['TU-01S:SS-Concrete-5AP:Temp-Mon'],
            2: ['TU-02S:SS-Concrete-5AP:Temp-Mon'],
            3: ['TU-03S:SS-Concrete-5AP:Temp-Mon'],
            # ['TU-04S:SS-Concrete-5AP:Temp-Mon'],
            5: ['TU-05S:SS-Concrete-5AN:Temp-Mon', 'TU-05S:SS-Concrete-5CN:Temp-Mon'],
            6: ['TU-06S:SS-Concrete-5AP:Temp-Mon'],
            7: ['TU-07S:SS-Concrete-5AP:Temp-Mon'],
            8: ['TU-08S:SS-Concrete-5AP:Temp-Mon'],
            # ['TU-09S:SS-Concrete-5AP:Temp-Mon'],
            10: ['TU-10S:SS-Concrete-5AN:Temp-Mon', 'TU-10S:SS-Concrete-5CN:Temp-Mon'],
            11: ['TU-11S:SS-Concrete-5AP:Temp-Mon']
        },
        "comb2": {
            15: ['TU-15S:SS-Concrete-5AN:Temp-Mon'],
            20: ['TU-20S:SS-Concrete-5AN:Temp-Mon'],
            5: ['TU-05S:SS-Concrete-5AN:Temp-Mon'],
            10: ['TU-10S:SS-Concrete-5AN:Temp-Mon']
        },
        "comb3": {
            12: ['TU-12S:SS-Concrete-5AP:Temp-Mon', 'TU-12S:SS-Concrete-5BP:Temp-Mon', 'TU-12S:SS-Concrete-5CP:Temp-Mon'],
            13: ['TU-13S:SS-Concrete-5AP:Temp-Mon', 'TU-13S:SS-Concrete-5BP:Temp-Mon', 'TU-13S:SS-Concrete-5CP:Temp-Mon'],
            # ['TU-14S:SS-Concrete-5BP:Temp-Mon'],
            15: ['TU-15S:SS-Concrete-5AN:Temp-Mon', 'TU-15S:SS-Concrete-5CN:Temp-Mon'],
            16: ['TU-16S:SS-Concrete-5AP:Temp-Mon', 'TU-16S:SS-Concrete-5BP:Temp-Mon', 'TU-16S:SS-Concrete-5CP:Temp-Mon'],
            17: ['TU-17S:SS-Concrete-5AP:Temp-Mon', 'TU-17S:SS-Concrete-5BP:Temp-Mon', 'TU-17S:SS-Concrete-5CP:Temp-Mon'],
            18: ['TU-18S:SS-Concrete-5AP:Temp-Mon', 'TU-18S:SS-Concrete-5BP:Temp-Mon', 'TU-18S:SS-Concrete-5CP:Temp-Mon'],
            # ['TU-19S:SS-Concrete-5BP:Temp-Mon'],
            20: ['TU-20S:SS-Concrete-5AN:Temp-Mon', 'TU-20S:SS-Concrete-5CN:Temp-Mon'],
            1: ['TU-01S:SS-Concrete-5AP:Temp-Mon', 'TU-01S:SS-Concrete-5BP:Temp-Mon', 'TU-01S:SS-Concrete-5CP:Temp-Mon'],
            2: ['TU-02S:SS-Concrete-5AP:Temp-Mon', 'TU-02S:SS-Concrete-5BP:Temp-Mon', 'TU-02S:SS-Concrete-5CP:Temp-Mon'],
            3: ['TU-03S:SS-Concrete-5AP:Temp-Mon', 'TU-03S:SS-Concrete-5BP:Temp-Mon', 'TU-03S:SS-Concrete-5CP:Temp-Mon'],
            # ['TU-04S:SS-Concrete-5BP:Temp-Mon'],
            5: ['TU-05S:SS-Concrete-5AN:Temp-Mon', 'TU-05S:SS-Concrete-5CN:Temp-Mon'],
            6: ['TU-06S:SS-Concrete-5AP:Temp-Mon', 'TU-06S:SS-Concrete-5BP:Temp-Mon', 'TU-06S:SS-Concrete-5CP:Temp-Mon'],
            7: ['TU-07S:SS-Concrete-5AP:Temp-Mon', 'TU-07S:SS-Concrete-5BP:Temp-Mon', 'TU-07S:SS-Concrete-5CP:Temp-Mon'],
            8: ['TU-08S:SS-Concrete-5AP:Temp-Mon', 'TU-08S:SS-Concrete-5BP:Temp-Mon', 'TU-08S:SS-Concrete-5CP:Temp-Mon'],
            # ['TU-09S:SS-Concrete-5BP:Temp-Mon'],
            10: ['TU-10S:SS-Concrete-5AN:Temp-Mon', 'TU-10S:SS-Concrete-5CN:Temp-Mon'],
            11: ['TU-11S:SS-Concrete-5AP:Temp-Mon', 'TU-11S:SS-Concrete-5BP:Temp-Mon', 'TU-11S:SS-Concrete-5CP:Temp-Mon']
        },
        "comb4": {
            15: ['TU-15S:SS-Concrete-1AN:Temp-Mon', 'TU-15S:SS-Concrete-1CN:Temp-Mon',\
                 'TU-15S:SS-Concrete-3AN:Temp-Mon', 'TU-15S:SS-Concrete-3CN:Temp-Mon',\
                 'TU-15S:SS-Concrete-5AN:Temp-Mon', 'TU-15S:SS-Concrete-5CN:Temp-Mon',\
                 'TU-15S:SS-Concrete-7AN:Temp-Mon', 'TU-15S:SS-Concrete-7CN:Temp-Mon',\
                 'TU-15S:SS-Concrete-9AN:Temp-Mon', 'TU-15S:SS-Concrete-9CN:Temp-Mon'],
            20: ['TU-20S:SS-Concrete-1AN:Temp-Mon', 'TU-20S:SS-Concrete-1CN:Temp-Mon',\
                 'TU-20S:SS-Concrete-3AN:Temp-Mon', 'TU-20S:SS-Concrete-3CN:Temp-Mon',\
                 'TU-20S:SS-Concrete-5AN:Temp-Mon', 'TU-20S:SS-Concrete-5CN:Temp-Mon',\
                 'TU-20S:SS-Concrete-7AN:Temp-Mon', 'TU-20S:SS-Concrete-7CN:Temp-Mon',\
                 'TU-20S:SS-Concrete-9AN:Temp-Mon', 'TU-20S:SS-Concrete-9CN:Temp-Mon'],
            5: ['TU-05S:SS-Concrete-1AN:Temp-Mon', 'TU-05S:SS-Concrete-1CN:Temp-Mon',\
                 'TU-05S:SS-Concrete-3AN:Temp-Mon', 'TU-05S:SS-Concrete-3CN:Temp-Mon',\
                 'TU-05S:SS-Concrete-5AN:Temp-Mon', 'TU-05S:SS-Concrete-5CN:Temp-Mon',\
                 'TU-05S:SS-Concrete-7AN:Temp-Mon', 'TU-05S:SS-Concrete-7CN:Temp-Mon',\
                 'TU-05S:SS-Concrete-9AN:Temp-Mon', 'TU-05S:SS-Concrete-9CN:Temp-Mon'],
            10: ['TU-10S:SS-Concrete-1AN:Temp-Mon', 'TU-10S:SS-Concrete-1CN:Temp-Mon',\
                 'TU-10S:SS-Concrete-3AN:Temp-Mon', 'TU-10S:SS-Concrete-3CN:Temp-Mon',\
                 'TU-10S:SS-Concrete-5AN:Temp-Mon', 'TU-10S:SS-Concrete-5CN:Temp-Mon',\
                 'TU-10S:SS-Concrete-7AN:Temp-Mon', 'TU-10S:SS-Concrete-7CN:Temp-Mon',\
                 'TU-10S:SS-Concrete-9AN:Temp-Mon', 'TU-10S:SS-Concrete-9CN:Temp-Mon'],
        },
        "comb5": {
            12: ['TU-12S:SS-Concrete-5AP:Temp-Mon', 'TU-12S:SS-Concrete-5BP:Temp-Mon', 'TU-12S:SS-Concrete-5CP:Temp-Mon'],
            13: ['TU-13S:SS-Concrete-5AP:Temp-Mon', 'TU-13S:SS-Concrete-5BP:Temp-Mon', 'TU-13S:SS-Concrete-5CP:Temp-Mon'],
            # ['TU-14S:SS-Concrete-5BP:Temp-Mon'],
            15: ['TU-15S:SS-Concrete-1AN:Temp-Mon', 'TU-15S:SS-Concrete-1CN:Temp-Mon',\
                 'TU-15S:SS-Concrete-3AN:Temp-Mon', 'TU-15S:SS-Concrete-3CN:Temp-Mon',\
                 'TU-15S:SS-Concrete-5AN:Temp-Mon', 'TU-15S:SS-Concrete-5CN:Temp-Mon',\
                 'TU-15S:SS-Concrete-7AN:Temp-Mon', 'TU-15S:SS-Concrete-7CN:Temp-Mon',\
                 'TU-15S:SS-Concrete-9AN:Temp-Mon', 'TU-15S:SS-Concrete-9CN:Temp-Mon'],
            16: ['TU-16S:SS-Concrete-5AP:Temp-Mon', 'TU-16S:SS-Concrete-5BP:Temp-Mon', 'TU-16S:SS-Concrete-5CP:Temp-Mon'],
            17: ['TU-17S:SS-Concrete-5AP:Temp-Mon', 'TU-17S:SS-Concrete-5BP:Temp-Mon', 'TU-17S:SS-Concrete-5CP:Temp-Mon'],
            18: ['TU-18S:SS-Concrete-5AP:Temp-Mon', 'TU-18S:SS-Concrete-5BP:Temp-Mon', 'TU-18S:SS-Concrete-5CP:Temp-Mon'],
            # ['TU-19S:SS-Concrete-5BP:Temp-Mon'],
            20: ['TU-20S:SS-Concrete-1AN:Temp-Mon', 'TU-20S:SS-Concrete-1CN:Temp-Mon',\
                 'TU-20S:SS-Concrete-3AN:Temp-Mon', 'TU-20S:SS-Concrete-3CN:Temp-Mon',\
                 'TU-20S:SS-Concrete-5AN:Temp-Mon', 'TU-20S:SS-Concrete-5CN:Temp-Mon',\
                 'TU-20S:SS-Concrete-7AN:Temp-Mon', 'TU-20S:SS-Concrete-7CN:Temp-Mon',\
                 'TU-20S:SS-Concrete-9AN:Temp-Mon', 'TU-20S:SS-Concrete-9CN:Temp-Mon'],
            1: ['TU-01S:SS-Concrete-5AP:Temp-Mon', 'TU-01S:SS-Concrete-5BP:Temp-Mon', 'TU-01S:SS-Concrete-5CP:Temp-Mon'],
            2: ['TU-02S:SS-Concrete-5AP:Temp-Mon', 'TU-02S:SS-Concrete-5BP:Temp-Mon', 'TU-02S:SS-Concrete-5CP:Temp-Mon'],
            3: ['TU-03S:SS-Concrete-5AP:Temp-Mon', 'TU-03S:SS-Concrete-5BP:Temp-Mon', 'TU-03S:SS-Concrete-5CP:Temp-Mon'],
            # ['TU-04S:SS-Concrete-5BP:Temp-Mon'],
            5: ['TU-05S:SS-Concrete-1AN:Temp-Mon', 'TU-05S:SS-Concrete-1CN:Temp-Mon',\
                 'TU-05S:SS-Concrete-3AN:Temp-Mon', 'TU-05S:SS-Concrete-3CN:Temp-Mon',\
                 'TU-05S:SS-Concrete-5AN:Temp-Mon', 'TU-05S:SS-Concrete-5CN:Temp-Mon',\
                 'TU-05S:SS-Concrete-7AN:Temp-Mon', 'TU-05S:SS-Concrete-7CN:Temp-Mon',\
                 'TU-05S:SS-Concrete-9AN:Temp-Mon', 'TU-05S:SS-Concrete-9CN:Temp-Mon'],
            6: ['TU-06S:SS-Concrete-5AP:Temp-Mon', 'TU-06S:SS-Concrete-5BP:Temp-Mon', 'TU-06S:SS-Concrete-5CP:Temp-Mon'],
            7: ['TU-07S:SS-Concrete-5AP:Temp-Mon', 'TU-07S:SS-Concrete-5BP:Temp-Mon', 'TU-07S:SS-Concrete-5CP:Temp-Mon'],
            8: ['TU-08S:SS-Concrete-5AP:Temp-Mon', 'TU-08S:SS-Concrete-5BP:Temp-Mon', 'TU-08S:SS-Concrete-5CP:Temp-Mon'],
            # ['TU-09S:SS-Concrete-5BP:Temp-Mon'],
            10: ['TU-10S:SS-Concrete-1AN:Temp-Mon', 'TU-10S:SS-Concrete-1CN:Temp-Mon',\
                 'TU-10S:SS-Concrete-3AN:Temp-Mon', 'TU-10S:SS-Concrete-3CN:Temp-Mon',\
                 'TU-10S:SS-Concrete-5AN:Temp-Mon', 'TU-10S:SS-Concrete-5CN:Temp-Mon',\
                 'TU-10S:SS-Concrete-7AN:Temp-Mon', 'TU-10S:SS-Concrete-7CN:Temp-Mon',\
                 'TU-10S:SS-Concrete-9AN:Temp-Mon', 'TU-10S:SS-Concrete-9CN:Temp-Mon'],
            11: ['TU-11S:SS-Concrete-5AP:Temp-Mon', 'TU-11S:SS-Concrete-5BP:Temp-Mon', 'TU-11S:SS-Concrete-5CP:Temp-Mon']
        },
        "comb6": {
            12: ['TU-12S:SS-Concrete-5AP:Temp-Mon', 'TU-12S:SS-Concrete-5BP:Temp-Mon', 'TU-12S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-12S:SS-Concrete-7AN:Temp-Mon', 'TU-12S:SS-Concrete-7CN:Temp-Mon',
                 'TU-12S:SS-Concrete-9AN:Temp-Mon', 'TU-12S:SS-Concrete-9CN:Temp-Mon'],
            13: ['TU-13S:SS-Concrete-5AP:Temp-Mon', 'TU-13S:SS-Concrete-5BP:Temp-Mon', 'TU-13S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-13S:SS-Concrete-7AN:Temp-Mon', 'TU-13S:SS-Concrete-7CN:Temp-Mon',
                 'TU-13S:SS-Concrete-9AN:Temp-Mon', 'TU-13S:SS-Concrete-9CN:Temp-Mon'],
            15: ['TU-15S:SS-Concrete-1AN:Temp-Mon', 'TU-15S:SS-Concrete-1CN:Temp-Mon',
                 'TU-15S:SS-Concrete-3AN:Temp-Mon', 'TU-15S:SS-Concrete-3BN:Temp-Mon', 'TU-15S:SS-Concrete-3CN:Temp-Mon', 
                 'TU-15S:SS-Concrete-4AN:Temp-Mon', 'TU-15S:SS-Concrete-4BN:Temp-Mon', 'TU-15S:SS-Concrete-4CN:Temp-Mon',
                 'TU-15S:SS-Concrete-5AN:Temp-Mon', 'TU-15S:SS-Concrete-5CN:Temp-Mon',
                 'TU-15S:SS-Concrete-6AN:Temp-Mon', 'TU-15S:SS-Concrete-6BN:Temp-Mon', 'TU-15S:SS-Concrete-6CN:Temp-Mon',
                 'TU-15S:SS-Concrete-7AN:Temp-Mon', 'TU-15S:SS-Concrete-7BN:Temp-Mon', 'TU-15S:SS-Concrete-7CN:Temp-Mon',
                 'TU-15S:SS-Concrete-9AN:Temp-Mon', 'TU-15S:SS-Concrete-9CN:Temp-Mon'],
            16: ['TU-16S:SS-Concrete-5AP:Temp-Mon', 'TU-16S:SS-Concrete-5BP:Temp-Mon', 'TU-16S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-16S:SS-Concrete-7AN:Temp-Mon', 'TU-16S:SS-Concrete-7CN:Temp-Mon',
                 'TU-16S:SS-Concrete-9AN:Temp-Mon', 'TU-16S:SS-Concrete-9CN:Temp-Mon'],
            17: ['TU-17S:SS-Concrete-5AP:Temp-Mon', 'TU-17S:SS-Concrete-5BP:Temp-Mon', 'TU-17S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-17S:SS-Concrete-7AN:Temp-Mon', 'TU-17S:SS-Concrete-7CN:Temp-Mon',
                 'TU-17S:SS-Concrete-9AN:Temp-Mon', 'TU-17S:SS-Concrete-9CN:Temp-Mon'],
            18: ['TU-18S:SS-Concrete-5AP:Temp-Mon', 'TU-18S:SS-Concrete-5BP:Temp-Mon', 'TU-18S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-18S:SS-Concrete-7AN:Temp-Mon', 'TU-18S:SS-Concrete-7CN:Temp-Mon',
                 'TU-18S:SS-Concrete-9AN:Temp-Mon', 'TU-18S:SS-Concrete-9CN:Temp-Mon'],
            20: ['TU-20S:SS-Concrete-1AN:Temp-Mon', 'TU-20S:SS-Concrete-1CN:Temp-Mon',
                 'TU-20S:SS-Concrete-3AN:Temp-Mon', 'TU-20S:SS-Concrete-3BN:Temp-Mon', 'TU-20S:SS-Concrete-3CN:Temp-Mon', 
                 'TU-20S:SS-Concrete-4AN:Temp-Mon', 'TU-20S:SS-Concrete-4BN:Temp-Mon', 'TU-20S:SS-Concrete-4CN:Temp-Mon',
                 'TU-20S:SS-Concrete-5AN:Temp-Mon', 'TU-20S:SS-Concrete-5CN:Temp-Mon',
                 'TU-20S:SS-Concrete-6AN:Temp-Mon', 'TU-20S:SS-Concrete-6BN:Temp-Mon', 'TU-20S:SS-Concrete-6CN:Temp-Mon',
                 'TU-20S:SS-Concrete-7AN:Temp-Mon', 'TU-20S:SS-Concrete-7BN:Temp-Mon', 'TU-20S:SS-Concrete-7CN:Temp-Mon',
                 'TU-20S:SS-Concrete-9AN:Temp-Mon', 'TU-20S:SS-Concrete-9CN:Temp-Mon'],
            1: ['TU-01S:SS-Concrete-5AP:Temp-Mon', 'TU-01S:SS-Concrete-5BP:Temp-Mon', 'TU-01S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-01S:SS-Concrete-7AN:Temp-Mon', 'TU-01S:SS-Concrete-7CN:Temp-Mon',
                 'TU-01S:SS-Concrete-9AN:Temp-Mon', 'TU-01S:SS-Concrete-9CN:Temp-Mon'],
            2: ['TU-02S:SS-Concrete-5AP:Temp-Mon', 'TU-02S:SS-Concrete-5BP:Temp-Mon', 'TU-02S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-02S:SS-Concrete-7AN:Temp-Mon', 'TU-02S:SS-Concrete-7CN:Temp-Mon',
                 'TU-02S:SS-Concrete-9AN:Temp-Mon', 'TU-02S:SS-Concrete-9CN:Temp-Mon'],
            3: ['TU-03S:SS-Concrete-5AP:Temp-Mon', 'TU-03S:SS-Concrete-5BP:Temp-Mon', 'TU-03S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-03S:SS-Concrete-7AN:Temp-Mon', 'TU-03S:SS-Concrete-7CN:Temp-Mon',
                 'TU-03S:SS-Concrete-9AN:Temp-Mon', 'TU-03S:SS-Concrete-9CN:Temp-Mon'],
            5: ['TU-05S:SS-Concrete-1AN:Temp-Mon', 'TU-05S:SS-Concrete-1CN:Temp-Mon',
                 'TU-05S:SS-Concrete-3AN:Temp-Mon', 'TU-05S:SS-Concrete-3BN:Temp-Mon', 'TU-05S:SS-Concrete-3CN:Temp-Mon', 
                 'TU-05S:SS-Concrete-4AN:Temp-Mon', 'TU-05S:SS-Concrete-4BN:Temp-Mon', 'TU-05S:SS-Concrete-4CN:Temp-Mon',
                 'TU-05S:SS-Concrete-5AN:Temp-Mon', 'TU-05S:SS-Concrete-5CN:Temp-Mon',
                 'TU-05S:SS-Concrete-6AN:Temp-Mon', 'TU-05S:SS-Concrete-6BN:Temp-Mon', 'TU-05S:SS-Concrete-6CN:Temp-Mon',
                 'TU-05S:SS-Concrete-7AN:Temp-Mon', 'TU-05S:SS-Concrete-7BN:Temp-Mon', 'TU-05S:SS-Concrete-7CN:Temp-Mon',
                 'TU-05S:SS-Concrete-9AN:Temp-Mon', 'TU-05S:SS-Concrete-9CN:Temp-Mon'],
            6: ['TU-06S:SS-Concrete-5AP:Temp-Mon', 'TU-06S:SS-Concrete-5BP:Temp-Mon', 'TU-06S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-06S:SS-Concrete-7AN:Temp-Mon', 'TU-06S:SS-Concrete-7CN:Temp-Mon',
                 'TU-06S:SS-Concrete-9AN:Temp-Mon', 'TU-06S:SS-Concrete-9CN:Temp-Mon'],
            7: ['TU-07S:SS-Concrete-5AP:Temp-Mon', 'TU-07S:SS-Concrete-5BP:Temp-Mon', 'TU-07S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-07S:SS-Concrete-7AN:Temp-Mon', 'TU-07S:SS-Concrete-7CN:Temp-Mon',
                 'TU-07S:SS-Concrete-9AN:Temp-Mon', 'TU-07S:SS-Concrete-9CN:Temp-Mon'],
            8: ['TU-08S:SS-Concrete-5AP:Temp-Mon', 'TU-08S:SS-Concrete-5BP:Temp-Mon', 'TU-08S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-08S:SS-Concrete-7AN:Temp-Mon', 'TU-08S:SS-Concrete-7CN:Temp-Mon',
                 'TU-08S:SS-Concrete-9AN:Temp-Mon', 'TU-08S:SS-Concrete-9CN:Temp-Mon'],
            10: ['TU-10S:SS-Concrete-1AN:Temp-Mon', 'TU-10S:SS-Concrete-1CN:Temp-Mon',
                 'TU-10S:SS-Concrete-3AN:Temp-Mon', 'TU-10S:SS-Concrete-3BN:Temp-Mon', 'TU-10S:SS-Concrete-3CN:Temp-Mon', 
                 'TU-10S:SS-Concrete-4AN:Temp-Mon', 'TU-10S:SS-Concrete-4BN:Temp-Mon', 'TU-10S:SS-Concrete-4CN:Temp-Mon',
                 'TU-10S:SS-Concrete-5AN:Temp-Mon', 'TU-10S:SS-Concrete-5CN:Temp-Mon',
                 'TU-10S:SS-Concrete-6AN:Temp-Mon', 'TU-10S:SS-Concrete-6BN:Temp-Mon', 'TU-10S:SS-Concrete-6CN:Temp-Mon',
                 'TU-10S:SS-Concrete-7AN:Temp-Mon', 'TU-10S:SS-Concrete-7BN:Temp-Mon', 'TU-10S:SS-Concrete-7CN:Temp-Mon',
                 'TU-10S:SS-Concrete-9AN:Temp-Mon', 'TU-10S:SS-Concrete-9CN:Temp-Mon'],
            11: ['TU-11S:SS-Concrete-5AP:Temp-Mon', 'TU-11S:SS-Concrete-5BP:Temp-Mon', 'TU-11S:SS-Concrete-5CP:Temp-Mon',\
                 'TU-11S:SS-Concrete-7AN:Temp-Mon', 'TU-11S:SS-Concrete-7CN:Temp-Mon',
                 'TU-11S:SS-Concrete-9AN:Temp-Mon', 'TU-11S:SS-Concrete-9CN:Temp-Mon'],
        },
        # todos sensores, sem os de corda vibrante
        "comb7*": {
            "Q1P1": ['TU-17S:SS-Concrete-5AP:Temp-Mon', 'TU-17S:SS-Concrete-5BP:Temp-Mon', 'TU-17S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P2": ['TU-16S:SS-Concrete-7AN:Temp-Mon', 'TU-16S:SS-Concrete-7CN:Temp-Mon',
                     'TU-16S:SS-Concrete-9AN:Temp-Mon', 'TU-16S:SS-Concrete-9CN:Temp-Mon'],
            "Q1P3": ['TU-16S:SS-Concrete-5AP:Temp-Mon', 'TU-16S:SS-Concrete-5BP:Temp-Mon', 'TU-16S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P4": ['TU-15S:SS-Concrete-7AN:Temp-Mon', 'TU-15S:SS-Concrete-7BN:Temp-Mon', 'TU-15S:SS-Concrete-7CN:Temp-Mon',
                     'TU-15S:SS-Concrete-9AN:Temp-Mon', 'TU-15S:SS-Concrete-9CN:Temp-Mon'],
            "Q1P5": ['TU-15S:SS-Concrete-4AN:Temp-Mon', 'TU-15S:SS-Concrete-4BN:Temp-Mon', 'TU-15S:SS-Concrete-4CN:Temp-Mon',
                     'TU-15S:SS-Concrete-5AN:Temp-Mon', 'TU-15S:SS-Concrete-5CN:Temp-Mon',
                     'TU-15S:SS-Concrete-6AN:Temp-Mon', 'TU-15S:SS-Concrete-6BN:Temp-Mon', 'TU-15S:SS-Concrete-6CN:Temp-Mon'],
            "Q1P6": ['TU-15S:SS-Concrete-1AN:Temp-Mon', 'TU-15S:SS-Concrete-1CN:Temp-Mon',
                     'TU-15S:SS-Concrete-3AN:Temp-Mon', 'TU-15S:SS-Concrete-3BN:Temp-Mon', 'TU-15S:SS-Concrete-3CN:Temp-Mon'],
            "Q1P7": ['TU-14S:SS-Concrete-5AP:Temp-Mon', 'TU-14S:SS-Concrete-5BP:Temp-Mon', 'TU-14S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P8": ['TU-13S:SS-Concrete-7AN:Temp-Mon', 'TU-13S:SS-Concrete-7CN:Temp-Mon',
                     'TU-13S:SS-Concrete-9AN:Temp-Mon', 'TU-13S:SS-Concrete-9CN:Temp-Mon'],
            "Q1P9": ['TU-13S:SS-Concrete-5AP:Temp-Mon', 'TU-13S:SS-Concrete-5BP:Temp-Mon', 'TU-13S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P10": ['TU-12S:SS-Concrete-7AN:Temp-Mon', 'TU-12S:SS-Concrete-7CN:Temp-Mon',
                      'TU-12S:SS-Concrete-9AN:Temp-Mon', 'TU-12S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P1": ['TU-12S:SS-Concrete-5AP:Temp-Mon', 'TU-12S:SS-Concrete-5BP:Temp-Mon', 'TU-12S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P2": ['TU-11S:SS-Concrete-7AN:Temp-Mon', 'TU-11S:SS-Concrete-7CN:Temp-Mon',
                     'TU-11S:SS-Concrete-9AN:Temp-Mon', 'TU-11S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P3": ['TU-11S:SS-Concrete-5AP:Temp-Mon', 'TU-11S:SS-Concrete-5BP:Temp-Mon', 'TU-11S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P4": ['TU-10S:SS-Concrete-7AN:Temp-Mon', 'TU-10S:SS-Concrete-7BN:Temp-Mon', 'TU-10S:SS-Concrete-7CN:Temp-Mon',
                     'TU-10S:SS-Concrete-9AN:Temp-Mon', 'TU-10S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P5": ['TU-10S:SS-Concrete-4AN:Temp-Mon', 'TU-10S:SS-Concrete-4BN:Temp-Mon', 'TU-10S:SS-Concrete-4CN:Temp-Mon',
                     'TU-10S:SS-Concrete-5AN:Temp-Mon', 'TU-10S:SS-Concrete-5CN:Temp-Mon',
                     'TU-10S:SS-Concrete-6AN:Temp-Mon', 'TU-10S:SS-Concrete-6BN:Temp-Mon', 'TU-10S:SS-Concrete-6CN:Temp-Mon'],
            "Q2P6": ['TU-10S:SS-Concrete-1AN:Temp-Mon', 'TU-10S:SS-Concrete-1CN:Temp-Mon',
                     'TU-10S:SS-Concrete-3AN:Temp-Mon', 'TU-10S:SS-Concrete-3BN:Temp-Mon', 'TU-10S:SS-Concrete-3CN:Temp-Mon'],
            "Q2P7": ['TU-09S:SS-Concrete-5AP:Temp-Mon', 'TU-09S:SS-Concrete-5BP:Temp-Mon', 'TU-09S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P8": ['TU-08S:SS-Concrete-7AN:Temp-Mon', 'TU-08S:SS-Concrete-7CN:Temp-Mon',
                     'TU-08S:SS-Concrete-9AN:Temp-Mon', 'TU-08S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P9": ['TU-08S:SS-Concrete-5AP:Temp-Mon', 'TU-08S:SS-Concrete-5BP:Temp-Mon', 'TU-08S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P10": ['TU-07S:SS-Concrete-7AN:Temp-Mon', 'TU-07S:SS-Concrete-7CN:Temp-Mon',
                      'TU-07S:SS-Concrete-9AN:Temp-Mon', 'TU-07S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P1": ['TU-07S:SS-Concrete-5AP:Temp-Mon', 'TU-07S:SS-Concrete-5BP:Temp-Mon', 'TU-07S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P2": ['TU-06S:SS-Concrete-7AN:Temp-Mon', 'TU-06S:SS-Concrete-7CN:Temp-Mon',
                     'TU-06S:SS-Concrete-9AN:Temp-Mon', 'TU-06S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P3": ['TU-06S:SS-Concrete-5AP:Temp-Mon', 'TU-06S:SS-Concrete-5BP:Temp-Mon', 'TU-06S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P4": ['TU-05S:SS-Concrete-7AN:Temp-Mon', 'TU-05S:SS-Concrete-7BN:Temp-Mon', 'TU-05S:SS-Concrete-7CN:Temp-Mon',
                     'TU-05S:SS-Concrete-9AN:Temp-Mon', 'TU-05S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P5": ['TU-05S:SS-Concrete-4AN:Temp-Mon', 'TU-05S:SS-Concrete-4BN:Temp-Mon', 'TU-05S:SS-Concrete-4CN:Temp-Mon',
                     'TU-05S:SS-Concrete-5AN:Temp-Mon', 'TU-05S:SS-Concrete-5CN:Temp-Mon',
                     'TU-05S:SS-Concrete-6AN:Temp-Mon', 'TU-05S:SS-Concrete-6BN:Temp-Mon', 'TU-05S:SS-Concrete-6CN:Temp-Mon'],
            "Q3P6": ['TU-05S:SS-Concrete-1AN:Temp-Mon', 'TU-05S:SS-Concrete-1CN:Temp-Mon',
                     'TU-05S:SS-Concrete-3AN:Temp-Mon', 'TU-05S:SS-Concrete-3BN:Temp-Mon', 'TU-05S:SS-Concrete-3CN:Temp-Mon'],
            "Q3P7": ['TU-04S:SS-Concrete-5AP:Temp-Mon', 'TU-04S:SS-Concrete-5BP:Temp-Mon', 'TU-04S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P8": ['TU-03S:SS-Concrete-7AN:Temp-Mon', 'TU-03S:SS-Concrete-7CN:Temp-Mon',
                     'TU-03S:SS-Concrete-9AN:Temp-Mon', 'TU-03S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P9": ['TU-03S:SS-Concrete-5AP:Temp-Mon', 'TU-03S:SS-Concrete-5BP:Temp-Mon', 'TU-03S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P10": ['TU-02S:SS-Concrete-7AN:Temp-Mon', 'TU-02S:SS-Concrete-7CN:Temp-Mon',
                      'TU-02S:SS-Concrete-9AN:Temp-Mon', 'TU-02S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P1": ['TU-02S:SS-Concrete-5AP:Temp-Mon', 'TU-02S:SS-Concrete-5BP:Temp-Mon', 'TU-02S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P2": ['TU-01S:SS-Concrete-7AN:Temp-Mon', 'TU-01S:SS-Concrete-7CN:Temp-Mon',
                     'TU-01S:SS-Concrete-9AN:Temp-Mon', 'TU-01S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P3": ['TU-01S:SS-Concrete-5AP:Temp-Mon', 'TU-01S:SS-Concrete-5BP:Temp-Mon', 'TU-01S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P4": ['TU-20S:SS-Concrete-7AN:Temp-Mon', 'TU-20S:SS-Concrete-7BN:Temp-Mon', 'TU-20S:SS-Concrete-7CN:Temp-Mon',
                     'TU-20S:SS-Concrete-9AN:Temp-Mon', 'TU-20S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P5": ['TU-20S:SS-Concrete-4AN:Temp-Mon', 'TU-20S:SS-Concrete-4BN:Temp-Mon', 'TU-20S:SS-Concrete-4CN:Temp-Mon',
                     'TU-20S:SS-Concrete-5AN:Temp-Mon', 'TU-20S:SS-Concrete-5CN:Temp-Mon',
                     'TU-20S:SS-Concrete-6AN:Temp-Mon', 'TU-20S:SS-Concrete-6BN:Temp-Mon', 'TU-20S:SS-Concrete-6CN:Temp-Mon'],
            "Q4P6": ['TU-20S:SS-Concrete-1AN:Temp-Mon', 'TU-20S:SS-Concrete-1CN:Temp-Mon',
                     'TU-20S:SS-Concrete-3AN:Temp-Mon', 'TU-20S:SS-Concrete-3BN:Temp-Mon', 'TU-20S:SS-Concrete-3CN:Temp-Mon'],
            "Q4P7": ['TU-19S:SS-Concrete-5AP:Temp-Mon', 'TU-19S:SS-Concrete-5BP:Temp-Mon', 'TU-19S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P8": ['TU-18S:SS-Concrete-7AN:Temp-Mon', 'TU-18S:SS-Concrete-7CN:Temp-Mon',
                     'TU-18S:SS-Concrete-9AN:Temp-Mon', 'TU-18S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P9": ['TU-18S:SS-Concrete-5AP:Temp-Mon', 'TU-18S:SS-Concrete-5BP:Temp-Mon', 'TU-18S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P10": ['TU-17S:SS-Concrete-7AN:Temp-Mon', 'TU-17S:SS-Concrete-7CN:Temp-Mon',
                      'TU-17S:SS-Concrete-9AN:Temp-Mon', 'TU-17S:SS-Concrete-9CN:Temp-Mon'],
        },
        # todos os sensores
        "all_sensors": {
            "Q1P1": ['TU-17S:SS-Concrete-5AP:Temp-Mon', 'TU-17S:SS-Concrete-5BP:Temp-Mon', 'TU-17S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P2": ['TU-16S:SS-Concrete-7AN:Temp-Mon', 'TU-16S:SS-Concrete-7CN:Temp-Mon',
                     'TU-16S:SS-Concrete-9AN:Temp-Mon', 'TU-16S:SS-Concrete-9CN:Temp-Mon'],
            "Q1P3": ['TU-16S:SS-Concrete-5AP:Temp-Mon', 'TU-16S:SS-Concrete-5BP:Temp-Mon', 'TU-16S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P4": ['TU-15S:SS-Concrete-7AN:Temp-Mon', 'TU-15S:SS-Concrete-7BN:Temp-Mon', 'TU-15S:SS-Concrete-7CN:Temp-Mon',
                     'TU-15S:SS-Concrete-7AV:Temp-Mon', 'TU-15S:SS-Concrete-7BV:Temp-Mon', 'TU-15S:SS-Concrete-7CV:Temp-Mon',
                     'TU-15S:SS-Concrete-9AN:Temp-Mon', 'TU-15S:SS-Concrete-9CN:Temp-Mon'],
            "Q1P5": ['TU-15S:SS-Concrete-4AN:Temp-Mon', 'TU-15S:SS-Concrete-4BN:Temp-Mon', 'TU-15S:SS-Concrete-4CN:Temp-Mon',
                     'TU-15S:SS-Concrete-4AV:Temp-Mon', 'TU-15S:SS-Concrete-4BV:Temp-Mon', 'TU-15S:SS-Concrete-4CV:Temp-Mon',
                     'TU-15S:SS-Concrete-5AN:Temp-Mon', 'TU-15S:SS-Concrete-5CN:Temp-Mon',
                     'TU-15S:SS-Concrete-5AV:Temp-Mon', 'TU-15S:SS-Concrete-5CV:Temp-Mon',
                     'TU-15S:SS-Concrete-6AN:Temp-Mon', 'TU-15S:SS-Concrete-6BN:Temp-Mon', 'TU-15S:SS-Concrete-6CN:Temp-Mon',
                     'TU-15S:SS-Concrete-6AV:Temp-Mon', 'TU-15S:SS-Concrete-6BV:Temp-Mon', 'TU-15S:SS-Concrete-6CV:Temp-Mon'],
            "Q1P6": ['TU-15S:SS-Concrete-1AN:Temp-Mon', 'TU-15S:SS-Concrete-1CN:Temp-Mon',
                     'TU-15S:SS-Concrete-3AN:Temp-Mon', 'TU-15S:SS-Concrete-3BN:Temp-Mon', 'TU-15S:SS-Concrete-3CN:Temp-Mon',
                     'TU-15S:SS-Concrete-3AV:Temp-Mon', 'TU-15S:SS-Concrete-3BV:Temp-Mon', 'TU-15S:SS-Concrete-3CV:Temp-Mon'],
            "Q1P7": ['TU-14S:SS-Concrete-5AP:Temp-Mon', 'TU-14S:SS-Concrete-5BP:Temp-Mon', 'TU-14S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P8": ['TU-13S:SS-Concrete-7AN:Temp-Mon', 'TU-13S:SS-Concrete-7CN:Temp-Mon',
                     'TU-13S:SS-Concrete-9AN:Temp-Mon', 'TU-13S:SS-Concrete-9CN:Temp-Mon'],
            "Q1P9": ['TU-13S:SS-Concrete-5AP:Temp-Mon', 'TU-13S:SS-Concrete-5BP:Temp-Mon', 'TU-13S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P10": ['TU-12S:SS-Concrete-7AN:Temp-Mon', 'TU-12S:SS-Concrete-7CN:Temp-Mon',
                      'TU-12S:SS-Concrete-9AN:Temp-Mon', 'TU-12S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P1": ['TU-12S:SS-Concrete-5AP:Temp-Mon', 'TU-12S:SS-Concrete-5BP:Temp-Mon', 'TU-12S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P2": ['TU-11S:SS-Concrete-7AN:Temp-Mon', 'TU-11S:SS-Concrete-7CN:Temp-Mon',
                     'TU-11S:SS-Concrete-9AN:Temp-Mon', 'TU-11S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P3": ['TU-11S:SS-Concrete-5AP:Temp-Mon', 'TU-11S:SS-Concrete-5BP:Temp-Mon', 'TU-11S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P4": ['TU-10S:SS-Concrete-7AN:Temp-Mon', 'TU-10S:SS-Concrete-7BN:Temp-Mon', 'TU-10S:SS-Concrete-7CN:Temp-Mon',
                     'TU-10S:SS-Concrete-7AV:Temp-Mon', 'TU-10S:SS-Concrete-7BV:Temp-Mon', 'TU-10S:SS-Concrete-7CV:Temp-Mon',
                     'TU-10S:SS-Concrete-9AN:Temp-Mon', 'TU-10S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P5": ['TU-10S:SS-Concrete-4AN:Temp-Mon', 'TU-10S:SS-Concrete-4BN:Temp-Mon', 'TU-10S:SS-Concrete-4CN:Temp-Mon',
                     'TU-10S:SS-Concrete-4AV:Temp-Mon', 'TU-10S:SS-Concrete-4BV:Temp-Mon', 'TU-10S:SS-Concrete-4CV:Temp-Mon',
                     'TU-10S:SS-Concrete-5AN:Temp-Mon', 'TU-10S:SS-Concrete-5CN:Temp-Mon',
                     'TU-10S:SS-Concrete-5AV:Temp-Mon', 'TU-10S:SS-Concrete-5CV:Temp-Mon',
                     'TU-10S:SS-Concrete-6AN:Temp-Mon', 'TU-10S:SS-Concrete-6BN:Temp-Mon', 'TU-10S:SS-Concrete-6CN:Temp-Mon',
                     'TU-10S:SS-Concrete-6AV:Temp-Mon', 'TU-10S:SS-Concrete-6BV:Temp-Mon', 'TU-10S:SS-Concrete-6CV:Temp-Mon'],
            "Q2P6": ['TU-10S:SS-Concrete-1AN:Temp-Mon', 'TU-10S:SS-Concrete-1CN:Temp-Mon',
                     'TU-10S:SS-Concrete-3AN:Temp-Mon', 'TU-10S:SS-Concrete-3BN:Temp-Mon', 'TU-10S:SS-Concrete-3CN:Temp-Mon',
                     'TU-10S:SS-Concrete-3AV:Temp-Mon', 'TU-10S:SS-Concrete-3BV:Temp-Mon', 'TU-10S:SS-Concrete-3CV:Temp-Mon'],
            "Q2P7": ['TU-09S:SS-Concrete-5AP:Temp-Mon', 'TU-09S:SS-Concrete-5BP:Temp-Mon', 'TU-09S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P8": ['TU-08S:SS-Concrete-7AN:Temp-Mon', 'TU-08S:SS-Concrete-7CN:Temp-Mon',
                     'TU-08S:SS-Concrete-9AN:Temp-Mon', 'TU-08S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P9": ['TU-08S:SS-Concrete-5AP:Temp-Mon', 'TU-08S:SS-Concrete-5BP:Temp-Mon', 'TU-08S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P10": ['TU-07S:SS-Concrete-7AN:Temp-Mon', 'TU-07S:SS-Concrete-7CN:Temp-Mon',
                      'TU-07S:SS-Concrete-9AN:Temp-Mon', 'TU-07S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P1": ['TU-07S:SS-Concrete-5AP:Temp-Mon', 'TU-07S:SS-Concrete-5BP:Temp-Mon', 'TU-07S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P2": ['TU-06S:SS-Concrete-7AN:Temp-Mon', 'TU-06S:SS-Concrete-7CN:Temp-Mon',
                     'TU-06S:SS-Concrete-9AN:Temp-Mon', 'TU-06S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P3": ['TU-06S:SS-Concrete-5AP:Temp-Mon', 'TU-06S:SS-Concrete-5BP:Temp-Mon', 'TU-06S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P4": ['TU-05S:SS-Concrete-7AN:Temp-Mon', 'TU-05S:SS-Concrete-7BN:Temp-Mon', 'TU-05S:SS-Concrete-7CN:Temp-Mon',
                     'TU-05S:SS-Concrete-7AV:Temp-Mon', 'TU-05S:SS-Concrete-7BV:Temp-Mon', 'TU-05S:SS-Concrete-7CV:Temp-Mon',
                     'TU-05S:SS-Concrete-9AN:Temp-Mon', 'TU-05S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P5": ['TU-05S:SS-Concrete-4AN:Temp-Mon', 'TU-05S:SS-Concrete-4BN:Temp-Mon', 'TU-05S:SS-Concrete-4CN:Temp-Mon',
                     'TU-05S:SS-Concrete-4AV:Temp-Mon', 'TU-05S:SS-Concrete-4BV:Temp-Mon', 'TU-05S:SS-Concrete-4CV:Temp-Mon',
                     'TU-05S:SS-Concrete-5AN:Temp-Mon', 'TU-05S:SS-Concrete-5CN:Temp-Mon',
                     'TU-05S:SS-Concrete-5AV:Temp-Mon', 'TU-05S:SS-Concrete-5CV:Temp-Mon',
                     'TU-05S:SS-Concrete-6AN:Temp-Mon', 'TU-05S:SS-Concrete-6BN:Temp-Mon', 'TU-05S:SS-Concrete-6CN:Temp-Mon',
                     'TU-05S:SS-Concrete-6AV:Temp-Mon', 'TU-05S:SS-Concrete-6BV:Temp-Mon', 'TU-05S:SS-Concrete-6CV:Temp-Mon'],
            "Q3P6": ['TU-05S:SS-Concrete-1AN:Temp-Mon', 'TU-05S:SS-Concrete-1CN:Temp-Mon',
                     'TU-05S:SS-Concrete-3AN:Temp-Mon', 'TU-05S:SS-Concrete-3BN:Temp-Mon', 'TU-05S:SS-Concrete-3CN:Temp-Mon',
                     'TU-05S:SS-Concrete-3AV:Temp-Mon', 'TU-05S:SS-Concrete-3BV:Temp-Mon', 'TU-05S:SS-Concrete-3CV:Temp-Mon'],
            "Q3P7": ['TU-04S:SS-Concrete-5AP:Temp-Mon', 'TU-04S:SS-Concrete-5BP:Temp-Mon', 'TU-04S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P8": ['TU-03S:SS-Concrete-7AN:Temp-Mon', 'TU-03S:SS-Concrete-7CN:Temp-Mon',
                     'TU-03S:SS-Concrete-9AN:Temp-Mon', 'TU-03S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P9": ['TU-03S:SS-Concrete-5AP:Temp-Mon', 'TU-03S:SS-Concrete-5BP:Temp-Mon', 'TU-03S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P10": ['TU-02S:SS-Concrete-7AN:Temp-Mon', 'TU-02S:SS-Concrete-7CN:Temp-Mon',
                      'TU-02S:SS-Concrete-9AN:Temp-Mon', 'TU-02S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P1": ['TU-02S:SS-Concrete-5AP:Temp-Mon', 'TU-02S:SS-Concrete-5BP:Temp-Mon', 'TU-02S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P2": ['TU-01S:SS-Concrete-7AN:Temp-Mon', 'TU-01S:SS-Concrete-7CN:Temp-Mon',
                     'TU-01S:SS-Concrete-9AN:Temp-Mon', 'TU-01S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P3": ['TU-01S:SS-Concrete-5AP:Temp-Mon', 'TU-01S:SS-Concrete-5BP:Temp-Mon', 'TU-01S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P4": ['TU-20S:SS-Concrete-7AN:Temp-Mon', 'TU-20S:SS-Concrete-7BN:Temp-Mon', 'TU-20S:SS-Concrete-7CN:Temp-Mon',
                     'TU-20S:SS-Concrete-7AV:Temp-Mon', 'TU-20S:SS-Concrete-7BV:Temp-Mon', 'TU-20S:SS-Concrete-7CV:Temp-Mon',
                     'TU-20S:SS-Concrete-9AN:Temp-Mon', 'TU-20S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P5": ['TU-20S:SS-Concrete-4AN:Temp-Mon', 'TU-20S:SS-Concrete-4BN:Temp-Mon', 'TU-20S:SS-Concrete-4CN:Temp-Mon',
                     'TU-20S:SS-Concrete-4AV:Temp-Mon', 'TU-20S:SS-Concrete-4BV:Temp-Mon', 'TU-20S:SS-Concrete-4CV:Temp-Mon',
                     'TU-20S:SS-Concrete-5AN:Temp-Mon', 'TU-20S:SS-Concrete-5CN:Temp-Mon',
                     'TU-20S:SS-Concrete-5AV:Temp-Mon', 'TU-20S:SS-Concrete-5CV:Temp-Mon',
                     'TU-20S:SS-Concrete-6AN:Temp-Mon', 'TU-20S:SS-Concrete-6BN:Temp-Mon', 'TU-20S:SS-Concrete-6CN:Temp-Mon',
                     'TU-20S:SS-Concrete-6AV:Temp-Mon', 'TU-20S:SS-Concrete-6BV:Temp-Mon', 'TU-20S:SS-Concrete-6CV:Temp-Mon'],
            "Q4P6": ['TU-20S:SS-Concrete-1AN:Temp-Mon', 'TU-20S:SS-Concrete-1CN:Temp-Mon',
                     'TU-20S:SS-Concrete-3AN:Temp-Mon', 'TU-20S:SS-Concrete-3BN:Temp-Mon', 'TU-20S:SS-Concrete-3CN:Temp-Mon',
                     'TU-20S:SS-Concrete-3AV:Temp-Mon', 'TU-20S:SS-Concrete-3BV:Temp-Mon', 'TU-20S:SS-Concrete-3CV:Temp-Mon'],
            "Q4P7": ['TU-19S:SS-Concrete-5AP:Temp-Mon', 'TU-19S:SS-Concrete-5BP:Temp-Mon', 'TU-19S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P8": ['TU-18S:SS-Concrete-7AN:Temp-Mon', 'TU-18S:SS-Concrete-7CN:Temp-Mon',
                     'TU-18S:SS-Concrete-9AN:Temp-Mon', 'TU-18S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P9": ['TU-18S:SS-Concrete-5AP:Temp-Mon', 'TU-18S:SS-Concrete-5BP:Temp-Mon', 'TU-18S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P10": ['TU-17S:SS-Concrete-7AN:Temp-Mon', 'TU-17S:SS-Concrete-7CN:Temp-Mon',
                      'TU-17S:SS-Concrete-9AN:Temp-Mon', 'TU-17S:SS-Concrete-9CN:Temp-Mon'],
        },
        # todos sensores - vw, só nível C
        "comb9*": {
            "Q1P1": ['TU-17S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P2": ['TU-16S:SS-Concrete-7CN:Temp-Mon',
                     'TU-16S:SS-Concrete-9CN:Temp-Mon'],
            "Q1P3": ['TU-16S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P4": ['TU-15S:SS-Concrete-7CN:Temp-Mon',
                     'TU-15S:SS-Concrete-7CV:Temp-Mon',
                     'TU-15S:SS-Concrete-9CN:Temp-Mon'],
            "Q1P5": ['TU-15S:SS-Concrete-4CN:Temp-Mon',
                     'TU-15S:SS-Concrete-4CV:Temp-Mon',
                     'TU-15S:SS-Concrete-5CN:Temp-Mon',
                     'TU-15S:SS-Concrete-5CV:Temp-Mon',
                     'TU-15S:SS-Concrete-6CN:Temp-Mon',
                     'TU-15S:SS-Concrete-6CV:Temp-Mon'],
            "Q1P6": ['TU-15S:SS-Concrete-1CN:Temp-Mon',
                     'TU-15S:SS-Concrete-3CN:Temp-Mon',
                     'TU-15S:SS-Concrete-3CV:Temp-Mon'],
            "Q1P7": ['TU-14S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P8": ['TU-13S:SS-Concrete-7CN:Temp-Mon',
                     'TU-13S:SS-Concrete-9CN:Temp-Mon'],
            "Q1P9": ['TU-13S:SS-Concrete-5CP:Temp-Mon'],
            "Q1P10": ['TU-12S:SS-Concrete-7CN:Temp-Mon',
                      'TU-12S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P1": ['TU-12S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P2": ['TU-11S:SS-Concrete-7CN:Temp-Mon',
                     'TU-11S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P3": ['TU-11S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P4": ['TU-10S:SS-Concrete-7CN:Temp-Mon',
                     'TU-10S:SS-Concrete-7CV:Temp-Mon',
                     'TU-10S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P5": ['TU-10S:SS-Concrete-4CN:Temp-Mon',
                     'TU-10S:SS-Concrete-4CV:Temp-Mon',
                     'TU-10S:SS-Concrete-5CN:Temp-Mon',
                     'TU-10S:SS-Concrete-5CV:Temp-Mon',
                     'TU-10S:SS-Concrete-6CN:Temp-Mon',
                     'TU-10S:SS-Concrete-6CV:Temp-Mon'],
            "Q2P6": ['TU-10S:SS-Concrete-1CN:Temp-Mon',
                     'TU-10S:SS-Concrete-3CN:Temp-Mon',
                     'TU-10S:SS-Concrete-3CV:Temp-Mon'],
            "Q2P7": ['TU-09S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P8": ['TU-08S:SS-Concrete-7CN:Temp-Mon',
                     'TU-08S:SS-Concrete-9CN:Temp-Mon'],
            "Q2P9": ['TU-08S:SS-Concrete-5CP:Temp-Mon'],
            "Q2P10": ['TU-07S:SS-Concrete-7CN:Temp-Mon',
                      'TU-07S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P1": ['TU-07S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P2": ['TU-06S:SS-Concrete-7CN:Temp-Mon',
                     'TU-06S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P3": ['TU-06S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P4": ['TU-05S:SS-Concrete-7CN:Temp-Mon',
                     'TU-05S:SS-Concrete-7CV:Temp-Mon',
                     'TU-05S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P5": ['TU-05S:SS-Concrete-4CN:Temp-Mon',
                     'TU-05S:SS-Concrete-4CV:Temp-Mon',
                     'TU-05S:SS-Concrete-5CN:Temp-Mon',
                     'TU-05S:SS-Concrete-5CV:Temp-Mon',
                     'TU-05S:SS-Concrete-6CN:Temp-Mon',
                     'TU-05S:SS-Concrete-6CV:Temp-Mon'],
            "Q3P6": ['TU-05S:SS-Concrete-1CN:Temp-Mon',
                     'TU-05S:SS-Concrete-3CN:Temp-Mon',
                     'TU-05S:SS-Concrete-3CV:Temp-Mon'],
            "Q3P7": ['TU-04S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P8": ['TU-03S:SS-Concrete-7CN:Temp-Mon',
                     'TU-03S:SS-Concrete-9CN:Temp-Mon'],
            "Q3P9": ['TU-03S:SS-Concrete-5CP:Temp-Mon'],
            "Q3P10": ['TU-02S:SS-Concrete-7CN:Temp-Mon',
                      'TU-02S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P1": ['TU-02S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P2": ['TU-01S:SS-Concrete-7CN:Temp-Mon',
                     'TU-01S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P3": ['TU-01S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P4": ['TU-20S:SS-Concrete-7CN:Temp-Mon',
                     'TU-20S:SS-Concrete-7CV:Temp-Mon',
                     'TU-20S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P5": ['TU-20S:SS-Concrete-4CN:Temp-Mon',
                     'TU-20S:SS-Concrete-4CV:Temp-Mon',
                     'TU-20S:SS-Concrete-5CN:Temp-Mon',
                     'TU-20S:SS-Concrete-5CV:Temp-Mon',
                     'TU-20S:SS-Concrete-6CN:Temp-Mon',
                     'TU-20S:SS-Concrete-6CV:Temp-Mon'],
            "Q4P6": ['TU-20S:SS-Concrete-1CN:Temp-Mon',
                     'TU-20S:SS-Concrete-3CN:Temp-Mon',
                     'TU-20S:SS-Concrete-3CV:Temp-Mon'],
            "Q4P7": ['TU-19S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P8": ['TU-18S:SS-Concrete-7CN:Temp-Mon',
                     'TU-18S:SS-Concrete-9CN:Temp-Mon'],
            "Q4P9": ['TU-18S:SS-Concrete-5CP:Temp-Mon'],
            "Q4P10": ['TU-17S:SS-Concrete-7CN:Temp-Mon',
                      'TU-17S:SS-Concrete-9CN:Temp-Mon'],
        }
    },
    "fancoil": [
        'FCPLC03:Blindagem:02-04',
        'FCPLC03:Blindagem:04-06',
        'FCPLC03:Blindagem:06-08',
        'FCPLC01:Blindagem:08-10',
        'FCPLC01:Blindagem:10-12',
        'FCPLC01:Blindagem:12-14',
        'FCPLC02:Blindagem:14-16',
        'FCPLC02:Blindagem:16-18',
        'FCPLC02:Blindagem:18-20',
        'FCPLC02:Blindagem:20-22'
    ],
    "hls": [
        'TU-17C:SS-HLS-Ax04NE5:Temp-Mon',
        'TU-16C:SS-HLS-Ax01NE4:Temp-Mon',
        'TU-15C:SS-HLS-Ax59NE3:Temp-Mon',
        # 'TU-14C:SS-HLS-Ax57NE2:Temp-Mon',
        'TU-13C:SS-HLS-Ax54NE1:Temp-Mon',
        'TU-01C:SS-HLS-Ax18SE5:Temp-Mon',
        # 'TU-01C:SS-HLS-Ax16SE4:Temp-Mon',
        'TU-20C:SS-HLS-Ax14SE3:Temp-Mon',
        # 'TU-19C:SS-HLS-Ax12SE2:Temp-Mon',
        'TU-18C:SS-HLS-Ax09SE1:Temp-Mon',
        'TU-06C:SS-HLS-Ax33SW5:Temp-Mon',
        # 'TU-06C:SS-HLS-Ax31SW4:Temp-Mon',
        'TU-05C:SS-HLS-Ax29SW3:Temp-Mon',
        # 'TU-04C:SS-HLS-Ax27SW2:Temp-Mon',
        'TU-03C:SS-HLS-Ax24SW1:Temp-Mon',
        'TU-11C:SS-HLS-Ax48NW5:Temp-Mon',
        # 'TU-11C:SS-HLS-Ax46NW4:Temp-Mon',
        'TU-10C:SS-HLS-Ax44NW3:Temp-Mon',
        # 'TU-09C:SS-HLS-Ax42NW2:Temp-Mon',
        'TU-08C:SS-HLS-Ax39NW1:Temp-Mon'
    ]
}

class TemperatureDeformation:
    def __init__(self, data_source: str, which_temp: str, timespam: dict = None, concrete_pvs_combination: str = None, filepath: str = None, combination_params: list = None ) -> None:
        self.filepath = filepath
        self.which_temp = which_temp
        self.data_source = data_source
        self.timespam = timespam
        self.concrete_pvs_combination = concrete_pvs_combination
        self.combination_params = combination_params

    def get_local_data(self) -> pd.DataFrame:
        temp_data = pd.read_excel(self.filepath)
        # temp_data.index = pd.DatetimeIndex([datetime.strptime(ts_str, "%d.%m.%y %H:%M") for ts_str in temp_data['datetime']])
        # temp_data.index = pd.DatetimeIndex([datetime.strptime(ts_str, "%Y-%m-%d %H:%M:$S") for ts_str in temp_data['datetime']])
        temp_data.index = pd.DatetimeIndex(temp_data['datetime'])
        temp_data.drop(columns=['datetime'], inplace=True)
        return temp_data
    
    def resolve_pvs(self) -> list:
        # simple case of fancoil or hls temp data
        if self.which_temp != 'concrete':
            return PVS[self.which_temp]
        # more complex case of concrete temp data
        pvs_list = []
        if self.combination_params is None:
            sector_pvs_relations = PVS[self.which_temp][self.concrete_pvs_combination]
            for sector in sector_pvs_relations:
                [pvs_list.append(pv) for pv in sector_pvs_relations[sector]]
        else:
            for position in self.custom_comb:
                [pvs_list.append(pv) for pv in self.custom_comb[position]]

        return pvs_list

    def get_data_from_archiver(self) -> pd.DataFrame:
        # pvs = PVS[self.which_temp] if self.concrete_pvs_combination is None else PVS[self.which_temp][self.concrete_pvs_combination]
        pvs = self.resolve_pvs()
        temp_data = asyncio.run(Archiver.request_data(pvs, self.timespam, 1))
        return temp_data
    
    
    def load_temp_data(self) -> None:
        # creating custom combination if it is the case
        if (not self.combination_params is None):
            self.generate_custom_pvs_combination()

        # extracting data from specified source
        if (self.data_source == 'local'):
            self.temp_data = self.get_local_data()
        elif (self.data_source == 'archiver'):
            self.temp_data = self.get_data_from_archiver()
        # referencing the first value
        self.temp_data = self.temp_data - self.temp_data.iloc[0,:]
        
        # calling general data treatment procedures
        self.treat_data()

    def treat_data(self):
        mapping = MAPPING_CARDINAL_SECTOR[self.which_temp]
        # simple mapping between sector and cardinals is needed
        if self.which_temp != 'concrete':
            for i, col in enumerate(self.temp_data.columns):
                if self.which_temp == 'fancoil':
                    sector_ref = col[-5:]
                elif self.which_temp == 'hls':
                    sector_ref = int(col[3:5])
                self.temp_data.columns.values[i] = mapping[sector_ref]
        # applies a mean between specific columns and map sector to cardinal
        else:
            treated_data = self.temp_data.copy()
            treated_data.drop(columns=self.temp_data.columns.values, inplace=True)

            sector_pvs_relations = PVS[self.which_temp][self.concrete_pvs_combination] if self.combination_params is None else self.custom_comb
            for sector in sector_pvs_relations:
                pvs = sector_pvs_relations[sector]
                try:
                    column_name = mapping[sector]
                except KeyError:
                    column_name = sector
                treated_data[column_name] = self.temp_data.loc[:, pvs].mean(axis='columns')
            # droping columns that refers to positions without loaded pvs
            treated_data.dropna(axis='columns', how='all', inplace=True)
            self.temp_data = treated_data
        

    def map_sector_to_cardinal(self):
        mapping = MAPPING_CARDINAL_SECTOR[self.which_temp]
        for i, col in enumerate(self.temp_data.columns):
            if (self.which_temp == 'concrete'):
                sector_ref = int(col[3:5])
            else:
                sector_ref = col[-5:]
            self.temp_data.columns.values[i] = mapping[sector_ref]
        
    def generate_custom_pvs_combination(self):
        base_comb = PVS['concrete']['all_sensors']
        custom_comb = {}
        level, sensor_type = self.combination_params
        for position in base_comb:
            # filtering level and sensor type
            custom_comb[position] = list(filter(lambda i: (level in i.split(':')[1].split('-')[2]) and (sensor_type in i.split(':')[1].split('-')[2]), base_comb[position]))
        self.custom_comb = custom_comb


    def calculate_deformation(self) -> pd.DataFrame:
        alpha = 12e-6 # concrete's thermal coeficient
        r = Perimeter.REAL_PERIMETER/(2 * np.pi) # considering a linear section of the slab in the radial direction
        self.def_data = self.temp_data.copy() * alpha * r
        return self.def_data

    def plot_temp(self, mode=None) -> None:
        """plots selected temp variables or the overall mean"""
        if mode == 'mean':
            _, ax = plt.subplots()
            mean_temp = self.temp_data.mean(axis=1).values
            print(f'delta temp. média = {mean_temp[-1]}')
            ax.plot(self.temp_data.index, mean_temp)
            plt.show()
            return

        plot = LegendPickablePlot()
        fig, ax = plot.get_plot_props()

        y = self.temp_data.iloc[:,:].values
        x = self.temp_data.index

        lines, legends = [], []
        line = ax.plot(x, y)
        [legends.append(var) for var in self.temp_data.columns.values]
        [lines.append(l) for l in line] 
        
        leg = ax.legend(lines, legends)        
        plot.define_legend_items(leg, lines)
        plot.change_legend_alpha(leg)

        # generic plot parameters and call
        ax.yaxis.labelpad = 10
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both')        
        ax.grid()
        fig.canvas.mpl_connect('pick_event', partial(LegendPickablePlot.on_pick, fig=fig, lined=plot.get_lined()))
        fig.tight_layout()
        plt.show()
