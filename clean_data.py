import pandas as pd
import numpy as np

# STUDENT CODE: Cleaning the data
# We want to take the big file and make it smaller (daily data)

def clean_data():
    print("Reading the big file... this might take a moment!")
    # The file has ';' as a separator. 
    # 'low_memory=False' helps read big files without warning.
    # We replace '?' with NaN (Not a Number) because some data is missing.
    df = pd.read_csv('../model/household_power_consumption.txt', sep=';', na_values=['?'], low_memory=False)

    print("Fixing the dates...")
    # Combine Date and Time into one 'datetime' column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    
    # Set the new datetime as the index (like the row label)
    df.set_index('datetime', inplace=True)

    # We only need 'Global_active_power' for this project
    # It is a string object initially because of '?', so we make sure it is float
    df['Global_active_power'] = df['Global_active_power'].astype(float)

    print("Handling missing values...")
    # forward fill: fill missing value with the previous day's value
    df['Global_active_power'].fillna(method='ffill', inplace=True)

    print("Calculating daily data...")
    # Resample to 'D' (Daily). 
    # The data is in 'active power' (kilowatts averaged over one minute).
    # To get energy (kWh), we sum it up and divide by 60? 
    # Actually, Global_active_power is (10/60)*min from the description usually, 
    # but let's just take the SUM for the day to represent total 'load' or 'usage' score.
    daily_data = df['Global_active_power'].resample('D').sum()

    print("Saving to CSV...")
    daily_data.to_csv('daily_electricity_usage.csv')
    print("Done! Created 'daily_electricity_usage.csv'")

if __name__ == "__main__":
    clean_data()
