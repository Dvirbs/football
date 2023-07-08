import plotly.graph_objects as go
import matplotlib.pyplot as plt
import webbrowser
import os
import seaborn as sns


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
        total_distance_values = summed_data.loc[day, 'Total Distance']

        # Calculate the confidence intervals
        confidence_interval_low = row['Confidence Interval (Low)']
        confidence_interval_high = row['Confidence Interval (High)']

        # Add scatter plot trace
        fig.add_trace(go.Scatter(x=[day] * len(total_distance_values), y=total_distance_values,
                                 mode='markers', marker=dict(symbol='circle', size=12),
                                 name=f"Day {day} - Total Distance"))

        # Add confidence interval lines
        fig.add_shape(type='line', x0=day, y0=confidence_interval_low,
                      x1=day, y1=confidence_interval_high,
                      line=dict(color='red', dash='dash', width=4),
                      name=f"Day {day} - Confidence Interval")

        # Add sample mean marker
        fig.add_trace(go.Scatter(x=[day], y=[row['Summed Total Distance']],
                                 mode='markers', marker=dict(symbol='diamond', size=16, color='black'),
                                 name=f"Day {day} - Sample Mean"))

    # Set x-axis label
    fig.update_xaxes(title_text="Match Day", title_font=dict(size=24), tickfont=dict(size=18))

    # Set y-axis label
    fig.update_yaxes(title_text="Total Distance Covered (units)", title_font=dict(size=24), tickfont=dict(size=18))

    # Set title
    fig.update_layout(title_text="Distance Covered in Practice for Each Match Day", title_font=dict(size=28))

    # Show legend
    fig.update_layout(showlegend=True, legend=dict(font=dict(size=18)))

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
