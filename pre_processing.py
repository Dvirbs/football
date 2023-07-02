import pandas as pd
import os
from datetime import datetime, timedelta


def validate_inputs(CSV_FNAME, resample_time, stats_column):
    """
    This function validates the inputs for the 'load_and_preprocess_data' function.
    """
    assert isinstance(CSV_FNAME, str), "CSV_FNAME should be a string"
    assert isinstance(resample_time, str), "resample_time should be a string"
    assert isinstance(stats_column, str), "stats_column should be a string"
    assert CSV_FNAME.endswith('.csv'), "CSV_FNAME should be a .csv file"


def load_and_preprocess_data(filename):
    # Load data
    df = pd.read_csv(f'./daq/{filename}')

    # Preprocess Data

    # Fill missing data if any, you can replace this with any other preprocessing step as required.
    df.fillna(method='ffill', inplace=True)

    # Convert time columns to datetime object, if they are not already
    time_columns = ['Drill Start Time', 'Drill End Time', 'Session Start Time', 'Session End Time']
    for column in time_columns:
        if df[column].dtype == 'object':
            df[column] = pd.to_datetime(df[column])

    return df


def aggregate_total_distance(column_to_aggregate):
    # The directory where the CSV files are stored
    directory = './daq/'

    # Initialize an empty DataFrame to store the aggregated data
    aggregated_df = pd.DataFrame()

    # Iterate over all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Extract the date from the filename
            date_str = filename.split('-Players-Full-report-export')[0]

            # Check if the hour is '24'
            if date_str[-2:] == '24':
                date_str = date_str[:-2] + '00'
                date = datetime.strptime(date_str, '%Y-%m-%d-%H').date() + timedelta(days=1)
            else:
                date = datetime.strptime(date_str, '%Y-%m-%d-%H').date()

            # Load the data
            df = pd.read_csv(directory + filename)

            # Sum the column of interest for all players
            total = df[column_to_aggregate].sum()

            # Append the data to the aggregated DataFrame
            aggregated_df = aggregated_df.append({'Date': date, column_to_aggregate: total}, ignore_index=True)

    # Set 'Date' as the index
    aggregated_df.set_index('Date', inplace=True)

    # Sort the DataFrame by date
    aggregated_df.sort_index(inplace=True)

    return aggregated_df


def check_data():
    directory = './daq/'

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(directory + filename)

            print(df.head())

            break  # Only process one file


def calculate_statistics(col_name: str, df, resample_df):
    """
    This function calculates the count, sum, and mean for the raw data and the resampled data.
    """
    # Calculating the statistics
    count = df[col_name].count()
    res_count = resample_df[col_name].count()
    sum_val = df[col_name].sum()
    res_sum = resample_df[col_name].sum()
    mean = df[col_name].mean()
    res_mean = resample_df[col_name].mean()

    # Preparing the data for the DataFrame
    stats_data = {
        f'{col_name} counts': [count, res_count],
        f'{col_name} sum': [sum_val, res_sum],
        f'{col_name} mean': [mean, res_mean]
    }

    # Creating the DataFrame
    stats_df = pd.DataFrame(stats_data, index=["raw data", "resampled data"])


def merge_and_calculate_derivatives(df, df2, stat):
    """
    This function combines the data from two DataFrames and calculates their first and second derivatives.
    """
    # Merging the data
    combined_data = pd.concat([df, df2], axis=1)

    # Calculating the first and second derivatives
    combined_data['first_derivative'] = combined_data[stat].diff()
    combined_data['second_derivative'] = combined_data[stat].diff().diff()

    return combined_data


def repair_missing_values(df, start_date, end_date):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # This generates a DatetimeIndex that includes every minute from 'start' to 'end'
    date_range = pd.date_range(start, end, freq='T').tz_localize('Asia/Jerusalem')
    df = df.reindex(date_range)
    mask = (df.index.hour >= 0) & (df.index.hour < 6)
    df[mask] = df[mask].bfill()
    return df
