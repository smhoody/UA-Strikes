import csv
import pandas as pd

# Input and output file paths
input_file = 'ukr-civharm-2024-02-26.csv'
output_file = 'output.txt'

# Read CSV and write formatted data to text file
with open(input_file, 'r') as csv_file, open(output_file, 'w') as txt_file:
    reader = csv.reader(csv_file)
    next(reader)  # Skip header if exists
    writer = csv.writer(txt_file)
    writer.writerow(['date', 'latitude', 'longitude'])  # Write header to text file
    df = {}
    for row in reader:
        #convert row[1] to dayofweek,dayofmonth,month
        df["date"] = pd.to_datetime(row[1])
        df["day_of_week"] = df["date"].dayofweek
        df["day_of_month"] = df["date"].day
        df["month"] = df["date"].month
        row[1] = f"{df['day_of_week']},{df['day_of_month']},{df['month']}"
        formatted_row = ','.join(row[1:4])  # Extract columns 2, 3, 4 and join with comma
        txt_file.write(formatted_row + '\n')  # Write formatted row to text file