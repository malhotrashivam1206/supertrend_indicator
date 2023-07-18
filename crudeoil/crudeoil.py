import json
from datetime import datetime, timedelta
from pytz import timezone
from time import sleep
import pandas as pd
import dash
import os
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import plotly.graph_objects as go
from pya3 import *

# Define your AliceBlue user ID and API key
user_id = 'AB093838'
api_key = 'cy5uYssgegMaUOoyWy0VGLBA6FsmbxYd0jNkajvBVJuEV9McAM3o0o2yG6Z4fEFYUGtTggJYGu5lgK89HumH3nBLbxsLjgplbodFHDLYeXX0jGQ5CUuGtDvYKSEzWSMk'

# Initialize AliceBlue connection
alice = Aliceblue(user_id=user_id, api_key=api_key)

# Print AliceBlue session ID
print(alice.get_session_id())

lp = 0
socket_opened = False
subscribe_flag = False
subscribe_list = []
unsubscribe_list = []
data_list = []  # List to store the received data
df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data

# File paths for saving data and graph
data_file_path = "crudeoil.csv"
graph_file_path = "crudeoil.html"

# Check if the data file exists
if os.path.exists(data_file_path):
    # Load existing data from the CSV file
    df = pd.read_csv(data_file_path, index_col="timestamp", parse_dates=True)
else:
    df = pd.DataFrame(columns=["timestamp", "lp"])  # Initialize an empty DataFrame for storing the data


def socket_open():
    print("Connected")
    global socket_opened
    socket_opened = True
    if subscribe_flag:
        alice.subscribe(subscribe_list)


def socket_close():
    global socket_opened, lp
    socket_opened = False
    lp = 0
    print("Closed")


def socket_error(message):
    global lp
    lp = 0
    print("Error:", message)


def feed_data(message):
    global lp, subscribe_flag, data_list
    feed_message = json.loads(message)
    if feed_message["t"] == "ck":
        print("Connection Acknowledgement status: %s (Websocket Connected)" % feed_message["s"])
        subscribe_flag = True
        print("subscribe_flag:", subscribe_flag)
        print("-------------------------------------------------------------------------------")
        pass
    elif feed_message["t"] == "tk":
        print("Token Acknowledgement status: %s" % feed_message)
        print("-------------------------------------------------------------------------------")
        pass
    else:
        print("Feed:", feed_message)
        if 'lp' in feed_message:
            timestamp = datetime.now(timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S.%f')
            feed_message['timestamp'] = timestamp
            lp = feed_message['lp']
            data_list.append(feed_message)  # Append the received data to the list
        else:
            print("'lp' key not found in feed message.")


# Connect to AliceBlue

# Socket Connection Request
alice.start_websocket(socket_open_callback=socket_open, socket_close_callback=socket_close,
                      socket_error_callback=socket_error, subscription_callback=feed_data, run_in_background=True,
                      market_depth=False)

while not socket_opened:
    pass

# Subscribe to Tata Motors
subscribe_list = [alice.get_instrument_by_token('MCX', 252453)]
alice.subscribe(subscribe_list)
print(datetime.now())
sleep(10)
print(datetime.now())

# Create an empty figure for the animated candlestick graph
fig = go.Figure()

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout of the Dash app
app.layout = html.Div([
    html.H1("Live Candlestick Graph", style={'textAlign': 'center'}),
    dcc.Graph(id='live-candlestick-graph', config={'displayModeBar': False, 'scrollZoom': False}),
    dcc.Dropdown(
        id='chart-type-dropdown',
        options=[
            {'label': 'Normal', 'value': 'normal'},
            {'label': 'Heikin Ashi', 'value': 'heikin_ashi'},
        ],
        value='normal',
        clearable=False,
        style={'width': '150px'}
    ),
    dcc.Dropdown(
        id='interval-dropdown',
        options=[
            {'label': '5 Sec', 'value': 5},
            {'label': '30 Sec', 'value': 30},
            {'label': '1 Min', 'value': 1},
            {'label': '3 Min', 'value': 3},
            {'label': '5 Min', 'value': 5},
            {'label': '10 Min', 'value': 10},
            {'label': '30 Min', 'value': 30},
            {'label': '60 Min', 'value': 60},
            {'label': '1 Day', 'value': 1440}
        ],
        value=1,
        clearable=False,
        style={'width': '150px'}
    ),
    dcc.Interval(id='graph-update-interval', interval=200, n_intervals=0)
], style={'height': '100vh', 'width': '100vw'})


def calculate_supertrend(data, atr_period=1, factor=2.0):
    data = data.copy()  # Create a copy of the data DataFrame

    if 'ha_open' in data.columns:
        ha_open = data['ha_open']
        ha_high = data['ha_high']
        ha_low = data['ha_low']
        ha_close = data['ha_close']
    else:
        ha_open = (data['open'].shift(1) + data['close'].shift(1)) / 2
        ha_high = data[['high', 'open', 'close']].max(axis=1)
        ha_low = data[['low', 'open', 'close']].min(axis=1)
        ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4

    tr = pd.DataFrame()
    tr['h-l'] = ha_high - ha_low
    tr['h-pc'] = abs(ha_high - ha_close.shift())
    tr['l-pc'] = abs(ha_low - ha_close.shift())
    tr['tr'] = tr.max(axis=1)

    atr = tr['tr'].rolling(atr_period).mean()

    data['upper_band'] = ha_open + factor * atr
    data['lower_band'] = ha_open - factor * atr

    supertrend = pd.Series(index=data.index)
    direction = pd.Series(index=data.index)

    supertrend.iloc[0] = ha_open.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(data)):
        if ha_close.iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = max(data['lower_band'].iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = min(data['upper_band'].iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = -1

    data['supertrend'] = supertrend  # Add the 'supertrend' column to the data DataFrame
    data['direction'] = direction  # Add the 'direction' column to the data DataFrame

    return data


def update_graph(n, interval, chart_type):
    global df, data_list

    # Check if there is new data
    if len(data_list) > 0:
        # Convert the received data list to a DataFrame
        new_df = pd.DataFrame(data_list)

        # Convert the 'lp' column to numeric format
        new_df['lp'] = pd.to_numeric(new_df['lp'], errors='coerce')

        # Drop rows with missing 'lp' values
        new_df = new_df.dropna(subset=['lp'])

        # Extract the relevant columns (timestamp, lp)
        new_df = new_df[["timestamp", "lp"]]

        # Convert the timestamp column to datetime format
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], format='%Y-%m-%d %H:%M:%S.%f')

        # Set the timestamp column as the DataFrame index
        new_df.set_index("timestamp", inplace=True)

        # Append the new data to the existing DataFrame
        df = df.append(new_df)

        # Save the updated data to the CSV file
        df.to_csv(data_file_path)

        data_list = []  # Clear the data list

    # Resample the data into OHLC format using the selected interval
    resampled_data = df["lp"].resample(f'{interval}T').ohlc()

    # Check if there is any data from the previous day overlapping with the current day's data
    if len(resampled_data) > 0:
        today_start = resampled_data.index[0].floor('D')  # Get the start of the current day
        previous_day_data = df["lp"].loc[df.index.floor('D') < today_start]  # Get the data from the previous day

        if len(previous_day_data) > 0:
            # Find the index of the last data point from the previous day
            previous_day_last_index = previous_day_data.index[-1]

            # Remove the duplicate data from the current day's data
            resampled_data = resampled_data.loc[resampled_data.index > previous_day_last_index]

    if chart_type == 'normal':
        # Calculate the uptrend and downtrend lines based on normal candles
        supertrend_data = calculate_supertrend(resampled_data)
        fig = plot_candlestick(resampled_data)
    elif chart_type == 'heikin_ashi':
        # Calculate the uptrend and downtrend lines based on Heikin Ashi candles
        ha_open = (resampled_data['open'].shift(1) + resampled_data['close'].shift(1)) / 2
        ha_high = resampled_data[['high', 'open', 'close']].max(axis=1)
        ha_low = resampled_data[['low', 'open', 'close']].min(axis=1)
        ha_close = (resampled_data['open'] + resampled_data['high'] + resampled_data['low'] + resampled_data['close']) / 4
        ha_data = pd.DataFrame({'open': ha_open, 'high': ha_high, 'low': ha_low, 'close': ha_close})
        supertrend_data = calculate_supertrend(ha_data)
        fig = plot_candlestick(ha_data)
    else:
        raise ValueError("Invalid chart type: {}".format(chart_type))

    # Initialize variables for tracking the trend
    current_trend = None
    trend_start = None
    trend_lines = []  # List to store trend lines

    # Iterate over the supertrend data to identify trend changes and add trend lines
    for i in range(len(supertrend_data)):
        current_signal = supertrend_data.iloc[i]

        if current_trend is None:
            # If no trend is active, start a new trend
            current_trend = current_signal['direction']
            trend_start = current_signal.name
        elif current_trend != current_signal['direction']:
            # If the trend changes, add the completed trend line to the list
            trend_data = resampled_data.loc[trend_start:current_signal.name]
            if len(trend_data) > 1:
                trend_lines.append((current_trend, trend_data[:-1]))

            # Update the trend variables
            current_trend = current_signal['direction']
            trend_start = current_signal.name

    # Add the last completed trend line to the list
    last_trend_data = resampled_data.loc[trend_start:]
    if len(last_trend_data) > 1:
        trend_lines.append((current_trend, last_trend_data[:-1]))

    # Add trend lines to the figure
    for trend, trend_data in trend_lines:
        if trend == 1:
            fig.add_trace(go.Scatter(x=trend_data.index,
                                     y=supertrend_data.loc[trend_data.index, 'supertrend'],
                                     mode='lines',
                                     name='Up Trend',
                                     line=dict(color='green')))
        else:
            fig.add_trace(go.Scatter(x=trend_data.index,
                                     y=supertrend_data.loc[trend_data.index, 'supertrend'],
                                     mode='lines',
                                     name='Down Trend',
                                     line=dict(color='red')))

    # Add markers for buy and sell signals
    buy_signals = supertrend_data[supertrend_data['direction'].diff() > 0]
    sell_signals = supertrend_data[supertrend_data['direction'].diff() < 0]

    fig.add_trace(go.Scatter(x=buy_signals.index,
                             y=buy_signals['supertrend'],
                             mode='markers',
                             name='Buy Signal',
                             marker=dict(color='blue', symbol='triangle-up', size=10)))

    fig.add_trace(go.Scatter(x=sell_signals.index,
                             y=sell_signals['supertrend'],
                             mode='markers',
                             name='Sell Signal',
                             marker=dict(color='orange', symbol='triangle-down', size=10)))

    # Save the graph to HTML file
    fig.write_html(graph_file_path)

    return fig


def plot_candlestick(data):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['open'],
                                         high=data['high'],
                                         low=data['low'],
                                         close=data['close'])])
    return fig


# Define the callback function to update the graph
@app.callback(Output('live-candlestick-graph', 'figure'),
              Input('graph-update-interval', 'n_intervals'),
              Input('interval-dropdown', 'value'),
              Input('chart-type-dropdown', 'value'))
def update_graph_callback(n, interval, chart_type):
    return update_graph(n, interval, chart_type)


# Run the Dash app
if __name__ == '__main__':
    df = pd.read_csv(data_file_path, index_col="timestamp", parse_dates=True)
    app.run_server(debug=True)
