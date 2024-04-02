'''Statistics class for pulling information from the 
   Missile strike data set 
@date: 3/28/2024
@author: Steven Hoodikoff
'''

import matplotlib.pyplot as plt
import pandas as pd
from util import Util

class Stats:
    def __init__(self):
        self.util = Util()
        self.util.read_data()
        self.data = self.util.get_data()

    def graph_strikes(self, number_of_days):
        strikes_hash_table = {}
        for sample in self.data[:number_of_days]:
            strikes_hash_table[]

