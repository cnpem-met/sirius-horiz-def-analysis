

from typing import Dict, List
import numpy as np
import pandas as pd


class Perimeter:    
    REAL_PERIMETER = 518.4
    
    def __init__(self, point_names, directions: list = None) -> None:
        self.point_names = point_names
        self.initial_coordinates = {}
        self.point_pairs = []
        self.directions = {}

        self.form_point_pairs()
        self.calculate_directions(directions)
        self.compute_initial_coordinates()
    
    def form_point_pairs(self) -> None:
        for i in range(len(self.point_names)-1):
            self.point_pairs.append((self.point_names[i], self.point_names[i+1]))
        self.point_pairs.append((self.point_names[-1], self.point_names[0]))

    def calculate_directions(self, directions: list = None) -> None:
        # calculating angle for each point considering an evenly space distribution
        # and that East direction is at 0 degrees
        for i, point_name in enumerate(self.point_names):
            self.directions[point_name] = i/len(self.point_names) * 360 if not directions else directions[i]
            

    def compute_initial_coordinates(self) -> None:
        # defining circunference size
        circum = self.REAL_PERIMETER
        radius = circum / (2*np.pi)
        # calculating initial coordinates for the points in the discrete circle
        # in relation to its center
        for dir in self.directions:
            theta = self.directions[dir] * (np.pi / 180)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = 0
            self.initial_coordinates[dir] = (x, y, z)

    @staticmethod
    def print_progress_bar (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 50, fill = '█', print_end = "\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = print_end)
        # Print New Line on Complete
        if iteration == total:
            print()

    @staticmethod
    def calc_2point_dist(pt1: List, pt2: List, mode: str ='2D') -> float:
        if (mode == '2D'):
            dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
        elif (mode == '3D'):
            dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2)**0.5
        return dist

    
    def calc_perimeter(self, points: Dict) -> float:
        """calculates perimeter by iterating over each pair of adjacent points and summing its current distances"""
        perim = 0
        for pairs in self.point_pairs:
            perim += Perimeter.calc_2point_dist(points[pairs[0]], points[pairs[1]], mode='3D')

        return perim


    def calculate_delta_perimeter(self, deformation_type: str, deformation: pd.DataFrame or Dict[pd.DataFrame]) -> np.ndarray:
        # setting first perimeter value as the initial/unaltered one
        perimeter_value = [self.calc_perimeter(self.initial_coordinates)]

        # defining the index to iterate over
        datetime_index = deformation.index if deformation_type == 'temperature' else list(deformation.values())[0].index

        # definining and calling the progress bar
        len_data = len(datetime_index) * len(self.point_names)
        Perimeter.print_progress_bar(0, len_data)

        # starting iteration for perimeter calculation
        for i, timestamp in enumerate(datetime_index):
            curr_pos = {}
            for j, coord in enumerate(self.point_names):
                # setting default coordinates
                x_pos = self.initial_coordinates[coord][0]
                y_pos = self.initial_coordinates[coord][1]
                z_pos = self.initial_coordinates[coord][2]

                # adding tidal effects
                if (deformation_type == 'tides'):
                    if (coord in deformation):
                        x_pos += deformation[coord].loc[timestamp,'East']
                        y_pos += deformation[coord].loc[timestamp,'North']
                        z_pos += deformation[coord].loc[timestamp,'Up']
                # adding thermal effects
                elif (deformation_type == 'temperature'):
                    if (coord in deformation.columns):
                        theta = self.directions[coord] * (np.pi / 180)
                        x_pos += np.cos(theta) * deformation.loc[timestamp, coord]
                        y_pos += np.sin(theta) * deformation.loc[timestamp, coord]

                curr_pos[coord] = (x_pos, y_pos, z_pos)
                
                # printing progress
                count = i * len(self.point_names) + j
                if (count % int(len_data/10) == 0):
                    Perimeter.print_progress_bar(count, len_data)

            perimeter_value.append(self.calc_perimeter(curr_pos))
        # setting first record as reference
        delta_perimeter = np.array(perimeter_value) - perimeter_value[0]
        return delta_perimeter