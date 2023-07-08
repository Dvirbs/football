import pandas as pd
import os
from datetime import datetime, timedelta


def load_and_preprocess_data(filename):
    # Load data
    df = pd.read_csv(f'{filename}')

    # Preprocess Data

    # Fill missing data if any, you can replace this with any other preprocessing step as required.
    #df.fillna(method='ffill', inplace=True)

    # Convert time columns to datetime object, if they are not already
    time_columns = ['Drill Start Time', 'Drill End Time', 'Session Start Time', 'Session End Time']
    for column in time_columns:
        if df[column].dtype == 'object':
            df[column] = pd.to_datetime(df[column])

    return df


def reorder_dataframe(df):
    """
    Reorder the DataFrame based on the 'Day' index.

    :param df: DataFrame with 'Day' as index and 'Summed Total Distance', 'Confidence Interval (Low)', and 'Confidence Interval (High)'
    :return: DataFrame ordered based on the 'Day' index.
    """
    # Specify the order of 'Day' values
    day_order = ['MD+1', 'MD+2', 'MD+3', 'MD-4', 'MD-3', 'MD-2', 'MD-1']

    # Reorder the DataFrame
    ordered_df = df.loc[day_order]

    return ordered_df















#
#
# def aggregate_total_distance(column_to_aggregate):
#     # The directory where the CSV files are stored
#     directory = './daq/'
#
#     # Initialize an empty DataFrame to store the aggregated data
#     aggregated_df = pd.DataFrame()
#
#     # Iterate over all CSV files in the directory
#     for filename in os.listdir(directory):
#         if filename.endswith('.csv'):
#             # Extract the date from the filename
#             date_str = filename.split('-Players-Full-report-export')[0]
#
#             # Check if the hour is '24'
#             if date_str[-2:] == '24':
#                 date_str = date_str[:-2] + '00'
#                 date = datetime.strptime(date_str, '%Y-%m-%d-%H').date() + timedelta(days=1)
#             else:
#                 date = datetime.strptime(date_str, '%Y-%m-%d-%H').date()
#
#             # Load the data
#             df = pd.read_csv(directory + filename)
#
#             # Sum the column of interest for all players
#             total = df[column_to_aggregate].sum()
#
#             # Append the data to the aggregated DataFrame
#             aggregated_df = aggregated_df.append({'Date': date, column_to_aggregate: total}, ignore_index=True)
#
#     # Set 'Date' as the index
#     aggregated_df.set_index('Date', inplace=True)
#
#     # Sort the DataFrame by date
#     aggregated_df.sort_index(inplace=True)
#
#     return aggregated_df
#
#
# def check_data():
#     directory = './daq/'
#
#     for filename in os.listdir(directory):
#         if filename.endswith('.csv'):
#             df = pd.read_csv(directory + filename)
#
#             print(df.head())
#
#             break  # Only process one file
#
#
# def calculate_statistics(col_name: str, df, resample_df):
#     """
#     This function calculates the count, sum, and mean for the raw data and the resampled data.
#     """
#     # Calculating the statistics
#     count = df[col_name].count()
#     res_count = resample_df[col_name].count()
#     sum_val = df[col_name].sum()
#     res_sum = resample_df[col_name].sum()
#     mean = df[col_name].mean()
#     res_mean = resample_df[col_name].mean()
#
#     # Preparing the data for the DataFrame
#     stats_data = {
#         f'{col_name} counts': [count, res_count],
#         f'{col_name} sum': [sum_val, res_sum],
#         f'{col_name} mean': [mean, res_mean]
#     }
#
#     # Creating the DataFrame
#     stats_df = pd.DataFrame(stats_data, index=["raw data", "resampled data"])


#
# def sum_total_distance_by_session(data):
#     # Group by 'Session' and sum the 'Total Distance' values
#     data.rename(columns={'Session ': 'Session'}, inplace=True)
#     summed_data = data.groupby('Session')['Total Distance'].sum().reset_index()
#
#     # Create a new DataFrame with the summed values
#     new_df = pd.DataFrame({
#         'Session': summed_data['Session'],
#         'Total Distance': summed_data['Total Distance']
#     })
#     new_df.set_index('Session', inplace=True)
#     return new_df
#
#
# def sum_total_distance_by_match_day(data):
#     # Group by 'Session' and sum the 'Total Distance' values
#     summed_data = data.groupby('Day')['Total Distance'].sum().reset_index()
#
#     # Create a new DataFrame with the summed values
#     new_df = pd.DataFrame({
#         'Day': summed_data['Day'],
#         'Total Distance': summed_data['Total Distance']
#     })
#     new_df.set_index('Day', inplace=True)
#     return new_df
