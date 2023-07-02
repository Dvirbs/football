import plotly.graph_objects as go
import matplotlib.pyplot as plt
import webbrowser
import os


def compute_and_plot_difference_by_threshold(df, columns_to_compare, difference_threshold):
    """
    This function computes the difference between two columns, marks
    where the absolute difference is less than a threshold, and generates a plot.

    Args:
        df (pd.DataFrame): The input dataframe.
        columns_to_compare (list): A list containing two column names to compare.
        difference_threshold (float): The threshold for marking the difference.

    Returns:
        None
    """

    # Select the columns to compare
    selected_columns_df = df[columns_to_compare].copy()

    # Drop NA values
    selected_columns_df = selected_columns_df.dropna()

    # Reset index to allow dropping duplicates
    selected_columns_df = selected_columns_df.reset_index()

    # Drop duplicates based on the date
    selected_columns_df = selected_columns_df.drop_duplicates(subset='date')

    # Set the date back as the index
    selected_columns_df = selected_columns_df.set_index('date')

    # Compute the difference
    selected_columns_df['difference_prediction'] = ((selected_columns_df[columns_to_compare[0]] - selected_columns_df[
        columns_to_compare[1]]) > difference_threshold).astype(int)

    # Create two separate dataframes based on 'difference_prediction'
    df0 = selected_columns_df[selected_columns_df['difference_prediction'] == 0]
    df1 = selected_columns_df[selected_columns_df['difference_prediction'] == 1]
    # Create the figure
    fig = plotly.graph_objects.Figure()
    # Add the traces
    fig.add_trace(plotly.graph_objects.Scatter(x=df0.index, y=df0['difference_prediction'], mode='markers',
                                               name='Urecsys Algorithm Control AHU',
                                               marker=dict(color='green', size=10)))
    fig.add_trace(plotly.graph_objects.Scatter(x=df1.index, y=df1['difference_prediction'], mode='markers',
                                               name="Urecsys Algorithm Don't Control AHU",
                                               marker=dict(color='red', size=10)))
    # # Add title and labels
    fig.update_layout(title=dict(
        text='Analysis of AHU Control based on Relative Humidity Measurements',
        font=dict(
            size=24,  # this line sets the font size of the title
        )
    ),
        xaxis=dict(
            title='Date',
            titlefont=dict(
                size=18,  # this line sets the font size of the x-axis title
            ),
            tickfont=dict(
                size=18,  # this line sets the font size of the x-axis tick labels
            ),
        ),
        yaxis=dict(
            title='Deviation in Predicted and Actual Humidity Levels',
            titlefont=dict(
                size=18,  # this line sets the font size of the y-axis title
            ),
            tickfont=dict(
                size=18,  # this line sets the font size of the y-axis tick labels
            ),
        ),
        legend=dict(
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0.7,
            font=dict(
                size=25,  # this line sets the font size
            )
        ),
        xaxis_title='Date',
        yaxis_title='Deviation in Predicted and Actual Humidity Levels', template='plotly_white')
    # Show the figure
    fig.show(renderer='browser')


def visualize_results(aggregate_df, adjusted_intensity, latest_date):
    """
    Visualize the Total Distance trend over time and mark the intensity of the latest practice and adjusted intensity of the next practice.

    :param aggregate_df: DataFrame with total distances for each day
    :param adjusted_intensity: adjusted intensity for the next practice
    :param latest_date: date of the latest practice
    """

    # Create a plot
    plt.figure(figsize=(12, 6))
    aggregate_df.mean(axis=1).plot()

    # Mark the intensity of the latest practice
    plt.scatter(latest_date, aggregate_df.loc[latest_date].mean(), color='red')
    plt.text(latest_date, aggregate_df.loc[latest_date].mean(), 'Latest Practice', verticalalignment='bottom',
             horizontalalignment='right')

    # Mark the adjusted intensity of the next practice
    plt.scatter(latest_date, adjusted_intensity, color='green')
    plt.text(latest_date, adjusted_intensity, 'Next Practice', verticalalignment='top', horizontalalignment='left')

    # Label the plot
    plt.title('Total Distance over time')
    plt.xlabel('Date')
    plt.ylabel('Total Distance')

    # Save the plot to a file
    filename = "plot.html"
    plt.savefig(filename)
    plt.close()

    # Open the plot in a web browser
    webbrowser.open('file://' + os.path.realpath(filename))


def visualize_results_go(aggregate_df, adjusted_intensity, latest_date):
    """
    Visualize the Total Distance trend over time and mark the intensity of the latest practice and adjusted intensity of the next practice.

    :param aggregate_df: DataFrame with total distances for each day
    :param adjusted_intensity: adjusted intensity for the next practice
    :param latest_date: date of the latest practice
    """

    # Create a plot
    fig = go.Figure()

    # Add line for average Total Distance over time
    fig.add_trace(
        go.Bar(x=aggregate_df.index, y=aggregate_df.mean(axis=1), name='Average Total Distance'))

    # Add markers for the latest practice and the next practice
    fig.add_trace(go.Scatter(x=[latest_date, latest_date], y=[aggregate_df.loc[latest_date].mean(), adjusted_intensity],
                             mode='markers', name='Practices',
                             marker=dict(size=[10, 10], color=['red', 'green']),
                             text=['Latest Practice', 'Next Practice'], textposition='bottom center'))

    # Label the plot
    fig.update_layout(title='Total Distance over time', xaxis_title='Date', yaxis_title='Total Distance')

    # Open the plot in a web browser
    fig.show(renderer='browser')
