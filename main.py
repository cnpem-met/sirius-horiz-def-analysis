import numpy as np
import pandas as pd
from pandas.core.arrays import boolean
from pandas.core.frame import DataFrame


from utils import DataUtils
from temp import TemperatureDeformation
from tides import Tides
from perimeter import Perimeter
from rf import RF
from plot import plot_rf


def create_temperature_deformation_data(temperature: TemperatureDeformation) -> pd.DataFrame:
    temperature.load_temp_data()
    return temperature.calculate_deformation()


def generate_node_temp_directions() -> list:
    # quadrant definitions
    inter_distances = [14.8, 11.1, 14.8, 11.1, 11.1, 14.8, 14.8, 11.1, 14.8, 11.1]
    arc_length = Perimeter.REAL_PERIMETER/4
    # calculating distances along the cirfunference
    distances_one_quad = [np.sum(inter_distances[0:i]) for i in range(len(inter_distances))]
    distances = distances_one_quad +\
                [(dist + arc_length) for dist in distances_one_quad] +\
                [(dist + 2*arc_length) for dist in distances_one_quad] +\
                [(dist + 3*arc_length) for dist in distances_one_quad]
    # mapping to angular position
    pos_ang = [distance/Perimeter.REAL_PERIMETER * 360 for distance in distances]
    return pos_ang

def generate_node_tides_directions() -> list:
    # mapped angles for each cardinal position, starting from East
    return [10.27, 28.26, 43.68, 64.23, 82.22, 118.26, 133.68, 154.23, 190.27, 208.26, 223.68, 244.23, 280.27, 298.26, 313.68, 334.23]

def main(temp_options: list, use_tides: boolean, use_temp: boolean, timespam: dict):
    # defining node/point names
    point_names_temp = ['Q1P1', 'Q1P2', 'Q1P3', 'Q1P4', 'Q1P5', 'Q1P6', 'Q1P7', 'Q1P8', 'Q1P9', 'Q1P10',
                        'Q2P1', 'Q2P2', 'Q2P3', 'Q2P4', 'Q2P5', 'Q2P6', 'Q2P7', 'Q2P8', 'Q2P9', 'Q2P10',
                        'Q3P1', 'Q3P2', 'Q3P3', 'Q3P4', 'Q3P5', 'Q3P6', 'Q3P7', 'Q3P8', 'Q3P9', 'Q3P10',
                        'Q4P1', 'Q4P2', 'Q4P3', 'Q4P4', 'Q4P5', 'Q4P6', 'Q4P7', 'Q4P8', 'Q4P9', 'Q4P10']
    point_names_tides = ["Q1P2","Q1P4","Q1P6","Q1P8","Q1P10","Q2P4","Q2P6","Q2P8","Q3P2","Q3P4","Q3P6","Q3P8","Q4P2","Q4P4","Q4P6","Q4P8"]

    if use_tides:
        # creating tide signals
        tides = Tides(point_names_tides, mapping_needed=True)
        tides.generate_tides(timespam)
        tides_data = tides.get_timeseries()

    if use_temp:
        # calculating local deformation based on simulated temperature fluctuations
        real_temp = TemperatureDeformation(timespam=timespam, **temp_options)
        temp_data = create_temperature_deformation_data(real_temp)


    # creating Perimeter object and calculating relative changes
    # based on temperatura and tides influence, individually

    # set/calculate the angular direction of each node/point
    directions = generate_node_temp_directions()

    # instantiate Perimeter class that will define the discretized circle scheme and calculate the perimeter evolution
    perimeter = Perimeter(point_names_temp, directions)
    # calculating perimeter evolution based on deformations caused by the temperature
    delta_perimeter_temp = perimeter.calculate_delta_perimeter('temperature', temp_data)

    # same process to tidal effects
    directions = generate_node_tides_directions()
    perimeter = Perimeter(point_names_tides, directions)
    delta_perimeter_tide = perimeter.calculate_delta_perimeter('tides', tides_data)

    # after calculating the contribution of both temperature and tides, compose the signals
    delta_perimeter = delta_perimeter_temp - delta_perimeter_tide
    # transforming to microns
    delta_perimeter *= 1e6
    # fixing length incompability with datetime index (don't know why)
    delta_perimeter = delta_perimeter[:-1]


    # load RF data
    rf = RF('archiver', timespam)
    rf_df = rf.get_data()
    rf_time = rf_df.index
    rf_data = rf_df.iloc[:,0]

    # # excluding outliers
    # rf_df = DataUtils.filter_and_save_dataframe(rf_df, plot_output=True)
    # rf_df, perim_filt = DataUtils.filter_dataframes_mutually(rf_df, pd.DataFrame(delta_perimeter, index=rf_time))
    # delta_perimeter = perim_filt.iloc[:,0]

    # shifting -3h (1 minute period -> 180 records)
    perim_temp = np.roll(delta_perimeter, -180)
    perim_temp -= perim_temp[0] # necessary to reference again

    # converting to Hz
    freq_temp = -perim_temp/1.04

    # extracting the contribution of the well
    well_contrib = rf_data - freq_temp

    # plotting (the last 180 records are ignored because of the 3h shift)
    plot_rf({'rf': [rf_time[:-180], rf_data[:-180]],\
             'temp': [temp_data.index[:-180], freq_temp[:-180]],\
             'poço': [temp_data.index[:-180], well_contrib[:-180]]})


if __name__ == "__main__":
    # user definitions
    use_tides = True
    use_temperature = True

    temp_options = {
        'data_source': 'archiver',
        'which_temp': 'concrete',
        'combination_params': ['A', 'N']
    }

    timespam = {
        'init': {'day': 13,'month': 11, 'year': 2021,'hour': 0,'minute': 0,'second': 0},
        'end': {'day': 14,'month': 11, 'year': 2021,'hour': 10,'minute': 0,'second': 0}
    }

    main(temp_options, use_tides, use_temperature, timespam)



