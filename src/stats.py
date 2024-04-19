'''Statistics class for pulling information from the 
   Missile strike data set 
@date: 3/28/2024
@author: Steven Hoodikoff
'''

import matplotlib.pyplot as plt
import pandas as pd
from util import Util
import numpy as np

class Stats:
    def __init__(self):
        self.util = Util()
        self.util.read_data()
        self.data = self.util.get_data()

    def get_all_strikes_by_day(self):
        ''' Return a dictionary of coordinate lists for each day
        format: {"3-24-2":[[45.523, 34.912], [50.353, 28.538]]}
        :return: dict
        '''
        strikes_hash_table = {}
        for sample in self.data:
            strike_data = sample.split(",")
            key = "-".join(strike_data[:3])
            if (strikes_hash_table.get(key) is None):
                strikes_hash_table[key] = [[float(coord) for coord in strike_data[3:]]]
            else:
                strikes_hash_table[key].append([float(coord) for coord in strike_data[3:]])
        
        return strikes_hash_table

    def graph_strikes(self, number_of_days):
        strike_table = self.get_all_strikes_by_day()
        iterator = 0
        # while (strike_table[iterator])
            

        plt.plot(number_of_days)
        plt.ylabel("Number of Strikes")
        plt.xlabel("Days from Start of War")
        plt.yticks(np.arange(0, 30, step=4), minor=True)
        plt.title("Number of Strikes Per Day")
        plt.show()

    def get_strike_count(self, coordinate_range):
        pass


    def convert_to_day_of_year(self, date):
        #split the date string into day and month
        day, month = map(int, date.split('-'))
        
        #define a list of days in each month
        days_in_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        
        #calculate the day of the year
        day_of_year = days_in_month[month - 1] + day
        
        return day_of_year

