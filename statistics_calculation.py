import pandas as pd
import numpy as np
import plotly.graph_objects
import scipy.stats as stats
from scipy import stats
from scipy.optimize import minimize

import pre_processing


def sum_total_distance_by_match_day(df):
    # Group by 'Day' and 'Session' and sum the 'Total Distance' values
    summed_data = df.groupby(['Day', 'Session, '])['Total Distance'].sum().reset_index()

    # Calculate the mean and standard deviation for 'Total Distance' grouped by 'Day'
    mean_by_day = summed_data.groupby('Day')['Total Distance'].mean()
    std_by_day = summed_data.groupby('Day')['Total Distance'].std()
    # test = summed_data.groupby('Day')['Total Distance'].agg({'mean','std','median','count'})

    # Calculate the confidence intervals for each day
    alpha = 0.05  # Significance level (95% confidence level)
    n = summed_data['Day'].nunique()
    t_critical = np.abs(stats.t.ppf(alpha / 2, n - 1))
    margin_of_error = t_critical * std_by_day / np.sqrt(n)
    confidence_interval_low = mean_by_day - margin_of_error
    confidence_interval_high = mean_by_day + margin_of_error

    # Create a new DataFrame with the summed values and confidence intervals
    new_df = pd.DataFrame({
        'Day': mean_by_day.index,
        'Summed Total Distance': mean_by_day.values,
        'Confidence Interval (Low)': confidence_interval_low.values,
        'Confidence Interval (High)': confidence_interval_high.values
    })
    new_df.set_index('Day', inplace=True)
    return new_df, summed_data


def calculate_average_distance_per_average_player_by_day(df):
    """
    Calculates the average distance per player for each day based on the provided DataFrame.

    Parameters:
        - df (DataFrame): Input DataFrame containing the data.

    Returns:
        - new_df (DataFrame): DataFrame with the average distance per player and confidence intervals, indexed by day.
        - summed_data (DataFrame): DataFrame with the summed total distance and number of unique players, grouped by day and session.

    Raises:
        - ValueError: If any session has zero players.

    """
    # Group by 'Day' and 'Session' and sum the 'Total Distance' values
    summed_data = df.groupby(['Day', 'Session '])['Total Distance'].sum().reset_index()
    num_players = df.groupby(['Day', 'Session '])['Player Last Name'].nunique().reset_index()
    summed_data = summed_data.merge(num_players, on=['Day', 'Session '], how='left')

    # Calculate the average total distance per player for each combination of 'Day' and 'Session'
    summed_data['Average Distance per Player'] = summed_data['Total Distance'] / summed_data['Player Last Name']

    # Calculate the mean and standard deviation for 'Average Distance per Player' grouped by 'Day'
    mean_by_day = summed_data.groupby('Day')['Average Distance per Player'].mean()
    std_by_day = summed_data.groupby('Day')['Average Distance per Player'].std()

    # Calculate the confidence intervals for each day
    alpha = 0.05  # Significance level (95% confidence level)
    new_df = pd.DataFrame()
    for day, n in summed_data.groupby('Day').size().items():
        t_critical = np.abs(stats.t.ppf(1 - alpha, n - 1))
        standard_error = std_by_day[day] / np.sqrt(n)
        margin_of_error = t_critical * standard_error
        confidence_interval_low = mean_by_day[day] - margin_of_error
        confidence_interval_high = mean_by_day[day] + margin_of_error

        new_df = new_df.append({
            'Day': day,
            'Total Distance': mean_by_day[day],
            'Confidence Interval (Low)': confidence_interval_low,
            'Confidence Interval (High)': confidence_interval_high
        }, ignore_index=True)

    new_df.set_index('Day', inplace=True)

    return new_df, summed_data


def calculate_average_distance_per_player(df):
    """
    Calculates the average distance per player for each day based on the provided DataFrame.

    Parameters:
        - df (DataFrame): Input DataFrame containing the data.

    Returns:
        - new_df (DataFrame): DataFrame with the average distance per player and confidence intervals, indexed by day.
        - summed_data (DataFrame): DataFrame with the summed total distance and number of unique players, grouped by day and session.

    Raises:
        - ValueError: If any session has zero players.

    """
    # Group by 'Day' and 'Session' and sum the 'Total Distance' values
    summed_data = df.groupby(['Day', 'Session ', 'Player Last Name'])['Total Distance'].sum().reset_index()

    # Calculate the mean and standard deviation for 'Total Distance' grouped by 'Day'
    mean_by_day = summed_data.groupby('Day')['Total Distance'].mean()
    std_by_day = summed_data.groupby('Day')['Total Distance'].std()

    # Calculate the confidence intervals for each day
    alpha = 0.05  # Significance level (95% confidence level)
    new_df = pd.DataFrame()
    for day, n in summed_data.groupby('Day').size().items():
        t_critical = np.abs(stats.t.ppf(1 - alpha, n - 1))
        standard_error = std_by_day[day] / np.sqrt(n)
        margin_of_error = t_critical * standard_error
        confidence_interval_low = mean_by_day[day] - margin_of_error
        confidence_interval_high = mean_by_day[day] + margin_of_error

        new_df = new_df.append({
            'Day': day,
            'Total Distance': mean_by_day[day],
            'Confidence Interval (Low)': confidence_interval_low,
            'Confidence Interval (High)': confidence_interval_high
        }, ignore_index=True)

    new_df.set_index('Day', inplace=True)

    return new_df, summed_data


def calculate_next_intensity(last_practice, historical_stats, next_day):
    # Calculate the total distance of the last practice
    total_distance_last_practice = last_practice['Total Distance'].sum()

    # Get the day of the last practice
    last_practice_day = last_practice['Day'].values[0]

    # Get the historical stats for this day
    hist_stats_for_day = historical_stats.loc[last_practice_day]

    # Calculate the percentile of the last practice
    percentile_last_practice = (total_distance_last_practice - hist_stats_for_day['Confidence Interval (Low)']) / (
            hist_stats_for_day['Confidence Interval (High)'] - hist_stats_for_day['Confidence Interval (Low)'])

    # Determine desired percentile for the next practice
    if percentile_last_practice > 0.75:
        desired_percentile_next_practice = 0.25
    elif percentile_last_practice < 0.25:
        desired_percentile_next_practice = 0.75
    else:
        desired_percentile_next_practice = percentile_last_practice

    # Get the historical stats for the next day
    hist_stats_for_next_day = historical_stats.loc[next_day]

    # Calculate the desired intensity for the next practice
    desired_intensity_next_practice = hist_stats_for_next_day[
                                          'Confidence Interval (Low)'] + desired_percentile_next_practice * (
                                              hist_stats_for_next_day['Confidence Interval (High)'] -
                                              hist_stats_for_next_day['Confidence Interval (Low)'])

    return desired_intensity_next_practice, percentile_last_practice, desired_percentile_next_practice


#
#
# def calculate_statistics_all_days(df):
#     # Use the describe function to calculate the statistics
#     stats = df.describe()
#
#     # Return the statistics
#     return stats
#
#
# def adjust_intensity(last_practice_df, historical_stats, intensity_col='Total Distance'):
#     """
#     Adjust today's workout intensity based on the intensity of the last practice and historical stats.
#
#     :param last_practice_df: DataFrame of the last practice
#     :param historical_stats: DataFrame of historical statistics obtained from the calculate_statistics function
#     :param intensity_col: column name of the intensity measure
#     :return: adjusted intensity
#     """
#
#     # Calculate the mean intensity of the last practice
#     last_practice_intensity = last_practice_df[intensity_col].mean()
#
#     # Calculate the average historical intensity
#     average_historical_intensity = historical_stats.loc['mean', intensity_col]
#
#     # Calculate the proportion of the last practice intensity to the average historical intensity
#     proportion = last_practice_intensity / average_historical_intensity
#
#     if proportion > 1.2:
#         # If the last practice intensity was more than 120% of the average historical intensity,
#         # set today's intensity to 80% of the average historical intensity
#         today_intensity = 0.8 * average_historical_intensity
#     elif proportion < 0.8:
#         # If the last practice intensity was less than 80% of the average historical intensity,
#         # set today's intensity to 110% of the average historical intensity
#         today_intensity = 1.1 * average_historical_intensity
#     else:
#         # If the last practice intensity was between 80% and 120% of the average historical intensity,
#         # keep today's intensity the same as the average historical intensity
#         today_intensity = average_historical_intensity
#
#     return today_intensity
#
#


def normal_log_likelihood(params, data):
    """
    Calculate the negative log-likelihood of a normal distribution.

    :param params: A tuple containing (mu, sigma).
    :param data: The data array for a specific day.
    :return: The negative log-likelihood value.
    """
    mu, sigma = params
    log_likelihood = -np.sum(np.log(1 / (np.sqrt(2 * np.pi * sigma ** 2))) - ((data - mu) ** 2) / (2 * sigma ** 2))
    return -log_likelihood


def estimate_normal_parameters(data):
    """
    Estimate the best parameters (mu, sigma) for a normal distribution that fits the data.

    :param data: The data array for a specific day.
    :return: A tuple containing (mu, sigma) - the estimated parameters.
    """
    # Initial guess for mu and sigma
    initial_params = (np.mean(data), np.std(data))

    # Use the minimize function from scipy.optimize to find the best parameters
    result = minimize(normal_log_likelihood, initial_params, args=(data,))

    # Extract the estimated parameters
    mu_estimated, sigma_estimated = result.x

    return mu_estimated, sigma_estimated
