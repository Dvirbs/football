from typing import Tuple, List
import pandas as pd
import numpy as np
# from sklearn.metrics import roc_curve, auc, accuracy_score
import plotly.graph_objects


def calculate_daily_statistics(df, start, end, stat):
    """
    Function to compute and organize daily statistics
    :param df: DataFrame containing data to be analyzed
    :param start: Start date for the analysis
    :param end: End date for the analysis
    :param stat: String indicating the statistic to be computed
    :return: DataFrame containing the computed statistics
    """
    # Input checks
    assert isinstance(df, pd.DataFrame), "df should be a pandas DataFrame"
    assert isinstance(start, str) and isinstance(end, str), "start and end should be strings representing dates"
    assert isinstance(stat, str), "stat should be a string"

    try:
        pd.to_datetime(start)
        pd.to_datetime(end)
    except ValueError:
        raise ValueError("start and end should be convertible to datetime")

    # List to collect the statistics DataFrames for each day
    statistics_dfs = []

    for date, day_df in df[start:end].groupby(df[start:end].index.date):
        # Calculate the statistics for this day
        statistics_df = calculate_statistics(day_df, stat)

        # Add the date as an index
        statistics_df['date'] = pd.to_datetime(date)
        statistics_df.set_index('date', append=True, inplace=True)

        # Append the statistics DataFrame to the list
        statistics_dfs.append(statistics_df)

    # Combine all statistics DataFrames into one DataFrame
    all_statistics_df = pd.concat(statistics_dfs)

    return all_statistics_df


def compute_absolute_sum(df, stat):
    """
    Function to compute the absolute sum of a column
    :param df: DataFrame containing data to be analyzed
    :param stat: Column on which to compute the absolute sum
    :return: Dictionary with the absolute sum of the column
    """
    # Input checks
    assert isinstance(df, pd.DataFrame), "df should be a pandas DataFrame"
    assert isinstance(stat, str), "stat should be a string"
    assert stat in df.columns, "stat should be a column in df"

    return df[[stat]].abs().sum().to_dict()


def calculate_before_after_timestamps(change_index):
    """
    Calculate the timestamps before and after a given timestamp
    """
    before_change_index = change_index - pd.Timedelta(minutes=20)
    after_20_min_change_index = change_index + pd.Timedelta(minutes=20)
    after_40_min_change_index = change_index + pd.Timedelta(minutes=40)

    return before_change_index, after_20_min_change_index, after_40_min_change_index


def get_statistics(day_df, start_index, end_index, measures):
    """Get statistics for a given time period"""
    return day_df[start_index:end_index][measures].describe().to_dict()


def calculate_integration_2(day_df, start_index, end_index, stat):
    """Calculate integration for a given time period"""
    return compute_absolute_sum(day_df[start_index:end_index], stat)


def calculate_statistics(day_df, stat):
    """Calculate command change related statistics"""
    assert isinstance(day_df, pd.DataFrame), "day_df should be a pandas DataFrame"
    assert isinstance(stat, str), "stat should be a string"
    assert stat in day_df.columns, "stat should be a column in day_df"

    change_indices = day_df[(day_df['command'].shift() == 0) & (day_df['command'] != 0)].index
    data = []

    for change_index in change_indices:
        apply_zero = change_index.hour == 6 and change_index.minute == 0

        before_change_index, after_20_min_change_index, after_40_min_change_index = calculate_before_after_timestamps(
            change_index)

        if apply_zero:
            empty_stats = {'value': {stat: np.nan for stat in ['mean', 'std', 'min', 'max']},
                           'first_derivative': {stat: np.nan for stat in ['mean', 'std', 'min', 'max']},
                           'second_derivative': {stat: np.nan for stat in ['mean', 'std', 'min', 'max']}}
            command_execution_period = empty_stats
            before_change_stats = empty_stats
            non_zero_integration = {'value': np.nan, 'first_derivative': np.nan, 'second_derivative': np.nan}
            before_change_integration = non_zero_integration
        else:
            command_execution_period_value = get_statistics(day_df, after_20_min_change_index,
                                                            after_40_min_change_index, [stat])
            command_execution_period_derivative = get_statistics(day_df, change_index, after_20_min_change_index,
                                                                 ['first_derivative', 'second_derivative'])
            command_execution_period = {**command_execution_period_value, **command_execution_period_derivative}

            before_change_stats = get_statistics(day_df, before_change_index, change_index,
                                                 [stat, 'first_derivative', 'second_derivative'])

            non_zero_integration = calculate_integration_2(day_df, after_20_min_change_index, after_40_min_change_index,
                                                           stat)
            before_change_integration = calculate_integration_2(day_df, before_change_index, change_index, stat)

        data_row = {
            'timestamp': change_index,
            **{f'command_execution_period_{k}_{stat}': value for k, stats in command_execution_period.items() for
               stat, value in stats.items()},
            **{f'before_command_sent_{k}_{stat}': value for k, stats in before_change_stats.items() for stat, value in
               stats.items()},
        }

        # Add integration related columns for only 'value'
        if stat == 'value':
            data_row.update({
                **{f'command_execution_period_{k}_integration': value for k, value in non_zero_integration.items() if
                   k == 'value'},
                **{f'before_command_sent_{k}_integration': value for k, value in before_change_integration.items() if
                   k == 'value'}
            })

        data.append(data_row)

    stats_df = pd.DataFrame(data)

    return stats_df


def validate_input_data(df: pd.DataFrame) -> None:
    """
    Validate the input data.

    Args:
    df (pd.DataFrame): DataFrame to be validated.
    """
    assert 'timestamp' in df.columns, "Input DataFrame must contain a 'timestamp' column."
    assert df['timestamp'].dtype == np.dtype('<M8[ns]'), "'timestamp' column must be of datetime64 type."


def analyze_roc_find_threshold(df):
    # List of measures to process
    measures = ['value', 'first_derivative', 'second_derivative']

    # List of statistics that you have calculated
    statistics = ['mean', 'std', 'min', 'max', '25%', '50%', '75%']
    # Define your date ranges
    start_date = pd.to_datetime('2022-02-08').tz_localize('Asia/Jerusalem')
    end_date = pd.to_datetime('2023-01-23').tz_localize('Asia/Jerusalem')
    start_date_2 = pd.to_datetime('2023-05-22').tz_localize('Asia/Jerusalem')
    end_date_2 = pd.to_datetime('2023-06-06').tz_localize('Asia/Jerusalem')
    # Create a condition for both date ranges
    condition = ((df['timestamp'] >= start_date) & (
            df['timestamp'] <= end_date)) | \
                ((df['timestamp'] >= start_date_2) & (
                        df['timestamp'] <= end_date_2))

    roc_df = pd.DataFrame()
    fig = plotly.graph_objects.Figure()

    # Use the condition to assign 0 to the dates that fulfill the condition, and 1 to the others
    roc_df['y_true'] = (~condition).astype(int)

    # Initialize a dictionary before the loop
    top_threshold_dict = {}

    maxi = 0
    top_threshold_data = [0, 'col_name']

    for i, measure in enumerate(measures):
        # Loop over each statistic
        for stat in statistics:
            if (measure == 'first_derivative' or measure == 'second_derivative') and stat == 'integration':
                continue
            # Define the command execution period column name
            command_col = f'command_execution_period_{measure}_{stat}'

            # Define the before command sent column name
            before_col = f'before_command_sent_{measure}_{stat}'

            roc_df[f'y_score_{measure}_{stat}'] = df[command_col] - df[before_col]
            roc_df = roc_df.dropna()

            fpr, tpr, thresholds = roc_curve(roc_df['y_true'], roc_df[f'y_score_{measure}_{stat}'])
            roc_auc = auc(fpr, tpr)
            # Add the ROC curve to the figure
            fig.add_trace(plotly.graph_objects.Scatter(x=fpr, y=tpr, mode='lines',
                                                       name=f'ROC curve (area = {roc_auc:.2f}) for {measure}_{stat}'))
            accuracy_ls = []
            for thres in thresholds:
                y_pred = np.where(roc_df[f'y_score_{measure}_{stat}'] > thres, 1, 0)
                accuracy_ls.append(accuracy_score(roc_df['y_true'], y_pred, normalize=True))

            accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)], axis=1)
            accuracy_ls.columns = ['thresholds', 'accuracy']
            accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
            accuracy_ls.head()
            top_threshold = accuracy_ls['thresholds'].iloc[0]

            # Update dictionary with the top_threshold for this measure and stat
            top_threshold_dict[f'y_score_{measure}_{stat}'] = top_threshold

            if maxi < roc_auc:
                maxi = roc_auc
                top_threshold_data = [top_threshold, f'y_score_{measure}_{stat}']

            # analyze_difference(df, [command_col, before_col], top_threshold)

    # After the loop, add 'y_true' key to the dictionary and set its value
    top_threshold_dict['y_true'] = np.nan  # or any suitable default value

    # Reset index to preserve existing data and add new row to the DataFrame
    roc_df.reset_index(drop=True, inplace=True)
    roc_df.loc[0] = top_threshold_dict
    roc_df.sort_index(inplace=True)

    roc_df['label'] = np.nan

    # Set the value for the first row in 'label' column as 'thresholds'
    roc_df.loc[0, 'label'] = 'thresholds'

    # Get a list of all the columns
    cols = list(roc_df.columns)

    # Remove 'label' from the list
    cols.remove('label')

    # Insert 'label' at the desired position. In this case, the second position corresponds to index 1.
    cols.insert(1, 'label')

    # Reorder the dataframe columns
    roc_df = roc_df[cols]

    fig.update_layout(title="ROC Curve - Days AHU NOT Listen")

    # fig.update_layout(title_text="Receiver operating characteristic example")
    fig.show(renderer='browser')
    # fig.write_html(f'html/ROC_Days_AHU_NOT_Listen.html')

    return roc_df, top_threshold_data[0], top_threshold_data[1]


def calculate_statistics_all_days(df):
    # Use the describe function to calculate the statistics
    stats = df.describe()

    # Return the statistics
    return stats


def adjust_intensity(last_practice_df, historical_stats, intensity_col='Total Distance'):
    """
    Adjust today's workout intensity based on the intensity of the last practice and historical stats.

    :param last_practice_df: DataFrame of the last practice
    :param historical_stats: DataFrame of historical statistics obtained from the calculate_statistics function
    :param intensity_col: column name of the intensity measure
    :return: adjusted intensity
    """

    # Calculate the mean intensity of the last practice
    last_practice_intensity = last_practice_df[intensity_col].mean()

    # Calculate the average historical intensity
    average_historical_intensity = historical_stats.loc['mean', intensity_col]

    # Calculate the proportion of the last practice intensity to the average historical intensity
    proportion = last_practice_intensity / average_historical_intensity

    if proportion > 1.2:
        # If the last practice intensity was more than 120% of the average historical intensity,
        # set today's intensity to 80% of the average historical intensity
        today_intensity = 0.8 * average_historical_intensity
    elif proportion < 0.8:
        # If the last practice intensity was less than 80% of the average historical intensity,
        # set today's intensity to 110% of the average historical intensity
        today_intensity = 1.1 * average_historical_intensity
    else:
        # If the last practice intensity was between 80% and 120% of the average historical intensity,
        # keep today's intensity the same as the average historical intensity
        today_intensity = average_historical_intensity

    return today_intensity

