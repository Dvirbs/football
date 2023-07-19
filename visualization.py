import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import scipy.stats as stats


def visualize_total_distance_distribution(new_df, summed_data):
    """
    Visualize the distribution of 'Total Distance' values for each day with confidence intervals.

    :param new_df: DataFrame with 'Day', 'Summed Total Distance', 'Confidence Interval (Low)', and 'Confidence Interval (High)'
    :param summed_data: DataFrame with 'Day', 'Session', and 'Total Distance'
    """
    # Create a plotly figure
    fig = go.Figure()

    # Iterate over each day and add scatter plot trace with confidence intervals
    for _, row in new_df.iterrows():
        day = row.name
        # Access the total distance value using the correct index value
        total_distance_values = summed_data.loc[summed_data['Day'] == day, 'Total Distance']

        # Calculate the confidence intervals
        confidence_interval_low = row['Confidence Interval (Low)']
        confidence_interval_high = row['Confidence Interval (High)']

        # Add scatter plot trace
        fig.add_trace(go.Scatter(x=[day] * len(total_distance_values), y=total_distance_values,
                                 mode='markers', marker=dict(symbol='circle', size=12),
                                 name=f"Day {day} - Total Distance per Player"))

        # Add confidence interval lines
        fig.add_shape(type='line', x0=day, y0=confidence_interval_low,
                      x1=day, y1=confidence_interval_high,
                      line=dict(color='red', dash='dash', width=4),
                      name=f"Day {day} - Confidence Interval")

        # Add sample mean marker
        fig.add_trace(go.Scatter(x=[day], y=[row['Total Distance']],
                                 mode='markers', marker=dict(symbol='diamond', size=16, color='black'),
                                 name=f"Day {day} - Sample Mean"))

    # Set x-axis label
    fig.update_xaxes(title_text="Match Day", title_font=dict(size=24), tickfont=dict(size=18))

    # Set y-axis label
    fig.update_yaxes(title_text="Total Distance Covered (meter)", title_font=dict(size=24), tickfont=dict(size=18))

    # Set title
    fig.update_layout(title_text="Total Distance Covered in Practice for Each Match Day", title_font=dict(size=28))

    # Show legend
    fig.update_layout(showlegend=True, legend=dict(font=dict(size=18)))

    # Show the plot
    fig.show(renderer='browser')


def visualize_distribution(data_df, summed_data, parameter_name):
    """
    Visualize the distribution of a specified parameter for each day with confidence intervals.

    :param data_df: DataFrame with 'Day', 'Summed {parameter_name}', 'Confidence Interval (Low)', and 'Confidence Interval (High)'
    :param summed_data: DataFrame with 'Day', 'Session', and the specified parameter
    :param parameter_name: Name of the parameter to visualize
    """
    # Create a plotly figure
    fig = go.Figure()

    # Iterate over each day and add scatter plot trace with confidence intervals
    for _, row in data_df.iterrows():
        day = row.name
        # Access the parameter values using the correct index value
        parameter_values = summed_data.loc[summed_data['Day'] == day, parameter_name]

        # Calculate the confidence intervals
        confidence_interval_low = row['Confidence Interval (Low)']
        confidence_interval_high = row['Confidence Interval (High)']

        # Add scatter plot trace
        fig.add_trace(go.Scatter(x=[day] * len(parameter_values), y=parameter_values,
                                 mode='markers', marker=dict(symbol='circle', size=12),
                                 name=f"Day {day} - {parameter_name} per Player"))

        # Add confidence interval lines
        fig.add_shape(type='line', x0=day, y0=confidence_interval_low,
                      x1=day, y1=confidence_interval_high,
                      line=dict(color='red', dash='dash', width=4),
                      name=f"Day {day} - Confidence Interval")

        # Add sample mean marker
        fig.add_trace(go.Scatter(x=[day], y=[row[parameter_name]],
                                 mode='markers', marker=dict(symbol='diamond', size=16, color='black'),
                                 name=f"Day {day} - Sample Mean"))

    # Set x-axis label
    fig.update_xaxes(title_text="Match Day", title_font=dict(size=24), tickfont=dict(size=18))

    # Set y-axis label
    fig.update_yaxes(title_text=f"{parameter_name} [meter]", title_font=dict(size=24), tickfont=dict(size=18))

    # Set title
    fig.update_layout(title_text=f"{parameter_name} in Practice for Each Match Day", title_font=dict(size=28))

    # Show legend
    fig.update_layout(showlegend=True, legend=dict(font=dict(size=18)))

    # fig.write_html('test_123_123.html')

    # Show the plot
    fig.show(renderer='browser')


def visualize_stats_go(last_practice, historical_stats, next_day, desired_intensity_next_practice):
    # Create the base figure
    fig = go.Figure()

    # Add bar chart for historical average total distances
    fig.add_trace(go.Bar(
        x=historical_stats.index,
        y=historical_stats['Summed Total Distance'],
        error_y=dict(
            type='data',
            symmetric=False,
            array=(historical_stats['Confidence Interval (High)'] - historical_stats['Summed Total Distance']),
            arrayminus=(historical_stats['Summed Total Distance'] - historical_stats['Confidence Interval (Low)']),
            visible=True
        ),
        name='Historical Avg. Distance'
    ))

    # Get the total distance of the last practice
    last_practice_day = last_practice['Day'].values[0]
    last_practice_distance = last_practice['Total Distance'].sum()

    # Add point for the last practice
    fig.add_trace(go.Scatter(
        x=[last_practice_day],
        y=[last_practice_distance],
        mode='markers',
        marker=dict(
            color='Red',
            size=20
        ),
        name='Last Practice'
    ))

    # Add point for the desired intensity of the next practice
    fig.add_trace(go.Scatter(
        x=[next_day],
        y=[desired_intensity_next_practice],
        mode='markers',
        marker=dict(
            color='Green',
            size=20
        ),
        name='Next Practice Target'
    ))

    # Set the layout
    fig.update_layout(
        title='Historical Distances with Confidence Intervals, Last Practice and Next Practice Target',
        xaxis_title='Day',
        yaxis_title='Total Distance',
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )

    fig.show(renderer='browser')


#
#
#
#
#
#
#
# def visualize_results(aggregate_df, adjusted_intensity, latest_date):
#     """
#     Visualize the Total Distance trend over time and mark the intensity of the latest practice and adjusted intensity of the next practice.
#
#     :param aggregate_df: DataFrame with total distances for each day
#     :param adjusted_intensity: adjusted intensity for the next practice
#     :param latest_date: date of the latest practice
#     """
#
#     # Create a plot
#     plt.figure(figsize=(12, 6))
#     aggregate_df.mean(axis=1).plot()
#
#     # Mark the intensity of the latest practice
#     plt.scatter(latest_date, aggregate_df.loc[latest_date].mean(), color='red')
#     plt.text(latest_date, aggregate_df.loc[latest_date].mean(), 'Latest Practice', verticalalignment='bottom',
#              horizontalalignment='right')
#
#     # Mark the adjusted intensity of the next practice
#     plt.scatter(latest_date, adjusted_intensity, color='green')
#     plt.text(latest_date, adjusted_intensity, 'Next Practice', verticalalignment='top', horizontalalignment='left')
#
#     # Label the plot
#     plt.title('Total Distance over time')
#     plt.xlabel('Date')
#     plt.ylabel('Total Distance')
#
#     # Save the plot to a file
#     filename = "plot.html"
#     plt.savefig(filename)
#     plt.close()
#
#     # Open the plot in a web browser
#     webbrowser.open('file://' + os.path.realpath(filename))
#
#
# def visualize_results_go(aggregate_df, adjusted_intensity, latest_date):
#     """
#     Visualize the Total Distance trend over time and mark the intensity of the latest practice and adjusted intensity of the next practice.
#
#     :param aggregate_df: DataFrame with total distances for each day
#     :param adjusted_intensity: adjusted intensity for the next practice
#     :param latest_date: date of the latest practice
#     """
#
#     # Create a plot
#     fig = go.Figure()
#
#     # Add line for average Total Distance over time
#     fig.add_trace(
#         go.Bar(x=aggregate_df.index, y=aggregate_df.mean(axis=1), name='Average Total Distance'))
#
#     # Add markers for the latest practice and the next practice
#     fig.add_trace(go.Scatter(x=[latest_date, latest_date], y=[aggregate_df.loc[latest_date].mean(), adjusted_intensity],
#                              mode='markers', name='Practices',
#                              marker=dict(size=[10, 10], color=['red', 'green']),
#                              text=['Latest Practice', 'Next Practice'], textposition='bottom center'))
#
#     # Label the plot
#     fig.update_layout(title='Total Distance over time', xaxis_title='Date', yaxis_title='Total Distance')
#
#     # Open the plot in a web browser
#     fig.show(renderer='browser')
#
#
# def visualize_results_go_session_data_structre(aggregate_df, stats, adjusted_intensity,
#                                                intensity_col='Total Distance'):
#     """
#     Visualize the Total Distance trend over time and mark the intensity of the last practice session and the adjusted
#     intensity for the next session.
#
#     :param adjusted_intensity: adjusted_intensity
#     :param aggregate_df: DataFrame with total distances for each session
#     :param stats: DataFrame of statistics obtained from the calculate_statistics_all_days function
#     :param intensity_col: column name of the intensity measure
#     """
#
#     # Create a plot
#     fig = go.Figure()
#
#     # Add line for average Total Distance over time
#     fig.add_trace(
#         go.Bar(x=aggregate_df.index, y=aggregate_df.mean(axis=1), name='Average Total Distance'))
#
#     # Add marker for the adjusted intensity of the next session
#     fig.add_trace(go.Scatter(x=[aggregate_df.index[-1] + 1], y=[adjusted_intensity],
#                              mode='markers', name='Next Session',
#                              marker=dict(size=[10], color=['green']),
#                              text=['Next Session'], textposition='bottom center'))
#
#     # Add vertical line for the overall mean Total Distance
#     # overall_mean = stats.loc['mean', intensity_col]
#     # fig.add_shape(type='line', x0=aggregate_df.index[0], y0=overall_mean,
#     #               x1=aggregate_df.index[-1], y1=overall_mean,
#     #               line=dict(color='black', width=2, dash='dash'))
#
#     # Calculate session_std
#     session_std = stats.loc['std', intensity_col]
#
#     # Convert session_std to a list
#     session_std_list = [session_std] * len(aggregate_df.index)
#
#     # # Add error bars to represent the standard deviation
#     # fig.add_trace(go.Bar(x=aggregate_df.index, y=aggregate_df, name='Average Total Distance',
#     #                      error_y=dict(type='data', array=session_std_list)))
#
#     # Label the plot
#     fig.update_layout(title='Total Distance over time', xaxis_title='Session', yaxis_title='Total Distance')
#
#     # Open the plot in a web browser
#     fig.show(renderer='browser')
#
#
# def visualize_statistics(stats):
#     # Extract statistical measures
#     mean = stats['Mean'][0]
#     confidence_interval_low = stats['Confidence Interval (Low)'][0]
#     confidence_interval_high = stats['Confidence Interval (High)'][0]
#
#     # Create a plot
#     fig = go.Figure()
#
#     # Add error bars for confidence interval
#     fig.add_trace(
#         go.Scatter(x=[1], y=[mean], mode='markers', name='Sample Mean', marker=dict(symbol='diamond', size=10)))
#     fig.add_trace(go.Scatter(x=[1], y=[confidence_interval_low, confidence_interval_high], mode='lines',
#                              name='Confidence Interval', line=dict(color='blue')))
#     fig.add_trace(
#         go.Scatter(x=[1], y=[mean], mode='markers', name='Population Mean', marker=dict(symbol='diamond', size=10),
#                    line=dict(color='red')))
#
#     # Label the plot
#     fig.update_layout(title='Statistical Description',
#                       xaxis=dict(title='Sample'),
#                       yaxis=dict(title='Total Distance'),
#                       showlegend=False)
#
#     # Open the plot in a web browser
#     fig.show(renderer='browser')
#
#
#
# def visualize_stats(last_practice, historical_stats, next_day, desired_intensity_next_practice):
#     # Set seaborn style
#     sns.set_theme()
#
#     # Create a barplot
#     plt.figure(figsize=(10, 6))
#     bar = sns.barplot(x=historical_stats.index, y="Summed Total Distance", data=historical_stats, ci="sd", capsize=.2,
#                       color='lightblue')
#
#     # Add error bars for the confidence intervals
#     bar.errorbar(x=historical_stats.index, y=historical_stats['Summed Total Distance'],
#                  yerr=[(top - bot) / 2 for top, bot in zip(historical_stats['Confidence Interval (High)'],
#                                                            historical_stats['Confidence Interval (Low)'])],
#                  fmt='o', color='black', capsize=5)
#
#     # Plot the total distance of the last practice
#     last_practice_day = last_practice['Day'].values[0]
#     last_practice_distance = last_practice['Total Distance'].sum()
#     plt.plot([last_practice_day], [last_practice_distance], marker='o', markersize=5, color="red",
#              label="Last Practice")
#
#     # Plot the desired intensity for the next practice
#     plt.plot([next_day], [desired_intensity_next_practice], marker='o', markersize=5, color="green",
#              label="Next Practice Target")
#
#     plt.legend()
#     plt.title('Historical Distances with Confidence Intervals, Last Practice and Next Practice Target')
#     plt.xlabel('Day')
#     plt.ylabel('Total Distance')
#     plt.show()
#


def plot_1d_histogram_per_day(data_df, specific_day=None):
    """
    Plot a 1D histogram for each day in the DataFrame.

    :param data_df: DataFrame with columns 'Day' and 'Total Distance'
    :param specific_day: Optional parameter to specify a specific day for plotting (e.g., 'MD+1')
                         If None, all days will be plotted.
    """
    # Filter data for the specific day or keep all days
    if specific_day:
        filtered_data = data_df[data_df['Day'] == specific_day]
    else:
        filtered_data = data_df

    # Get the unique days from the filtered DataFrame
    unique_days = filtered_data['Day'].unique()

    # Set the number of bins for the histogram
    num_bins = 40

    # Define subplot arrangement
    num_days = len(unique_days)
    if num_days <= 6:
        rows = 2
        cols = 3
    else:
        rows = 3
        cols = 3

    # Create subplots
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Day {day}" for day in unique_days])

    # Plot histograms for each day
    for i, day in enumerate(unique_days, start=1):
        # Get the data for the current day
        day_data = filtered_data[filtered_data['Day'] == day]

        # Plot the histogram for 'Total Distance' values for the current day
        row = (i - 1) // cols + 1
        col = (i - 1) % cols + 1
        fig.add_trace(go.Histogram(x=day_data['Total Distance'], nbinsx=num_bins, opacity=0.7,
                                   name=f"Day {day}"), row=row, col=col)

        # Set subplot titles
        fig.update_xaxes(title_text="Total Distance", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    # Set main title for the entire plot
    if specific_day:
        fig.update_layout(title_text=f"1D Histogram for Day {specific_day}")
    else:
        fig.update_layout(title_text="1D Histograms for All Days", height=1200)

    # Show legend
    fig.update_layout(showlegend=True)

    # Write the figure to an HTML file
    html_filename = "1d_histograms.html"
    #pio.write_html(fig, file=html_filename)

    # Show the plot in the browser
    fig.show(renderer='browser')


def visualize_data_and_estimation(data, mu_estimate, sigma_estimate):
    """
    Visualize the original data and the estimated normal distribution parameters.

    :param data: The original data array for a specific day.
    :param mu_estimate: The estimated mean (mu) of the normal distribution.
    :param sigma_estimate: The estimated standard deviation (sigma) of the normal distribution.
    """
    # Create a histogram of the original data
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Original Data')

    # Create an array of x values for the PDF of the estimated normal distribution
    x = np.linspace(np.min(data), np.max(data), 100)

    # Calculate the PDF of the estimated normal distribution using scipy.stats.norm
    pdf_estimate = stats.norm.pdf(x, loc=mu_estimate, scale=sigma_estimate)

    # Plot the estimated normal distribution PDF
    plt.plot(x, pdf_estimate, 'r', label='Estimated Normal Distribution')

    # Plot a vertical line at the estimated mean (mu)
    plt.axvline(x=mu_estimate, color='orange', linestyle='--', linewidth=2,
                label=f'Estimated Mean (mu): {mu_estimate:.2f}')

    # Set plot labels and title
    plt.xlabel('Total Distance')
    plt.ylabel('Probability Density')
    plt.title('Histogram and Estimated Normal Distribution')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()
