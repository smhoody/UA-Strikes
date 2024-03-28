import csv
import pandas as pd

class Util:
    ''' Utility class helper '''
    def __init__(self):
        self.input_file = 'ukr-civharm-2024-02-26.csv'
        self.data = [] #holds all sample data
        self.training_data = [] #holds data for training
        self.testing_data = [] #holds data for testing

    def read_data(self):
        # Read CSV and save necessary data (date & coordinates)
        with open(self.input_file, 'r', encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip header
            df = {} #temporary date-format storage

            for row in reader:
                #convert row[1] (date) to dayofweek,dayofmonth,month
                df = self.get_formatted_date(row[1])
                row[1] = f"{df['day_of_week']},{df['day_of_month']},{df['month']}"
                formatted_row = ','.join(row[1:4])  # Extract columns 2, 3, 4 and join with comma
                #finished formatting is: dayofweek,dayofmonth,month,latitude,longitude
                self.data.append(formatted_row)
        
        self.partition_data() #separate into training and test data


    def partition_data(self):
        ''' Separate sample data into batches
        return :list, list: list of training and test samples  
        '''

        #every 3rd sample goes to testing
        for i, sample in enumerate(self.data):
            if (i%3==0): self.testing_data.append(sample)
            else: self.training_data.append(sample)
        
        # convert all data to floats
        for i, sample in enumerate(self.training_data):
            converted_sample = list(map(float, sample.split(",")))
            self.training_data[i] = converted_sample
        for i, sample in enumerate(self.testing_data):
            converted_sample = list(map(float, sample.split(",")))
            self.testing_data[i] = converted_sample

    
    def get_formatted_date(self, date):
        ''' Create a dictionary from a date that contains the 
            date, day of week, day of month, and month of year.
        return :dictionary:
        e.g.: 02/24/2022 => {"date":"02/24/2022", "day_of_week":3, "day_of_month":24, "month":2}
        '''
        df = {}
        df["date"] = pd.to_datetime(date)
        df["day_of_week"] = int(df["date"].dayofweek)
        df["day_of_month"] = int(df["date"].day)
        df["month"] = int(df["date"].month)

        return df