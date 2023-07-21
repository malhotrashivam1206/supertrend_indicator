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
import numpy as np

# Define your AliceBlue user ID and API key
user_id = 'AB093838'
api_key = 'cy5uYssgegMaUOoyWy0VGLBA6FsmbxYd0jNkajvBVJuEV9McAM3o0o2yG6Z4fEFYUGtTggJYGu5lgK89HumH3nBLbxsLjgplbodFHDLYeXX0jGQ5CUuGtDvYKSEzWSMk'

# Initialize AliceBlue connection
alice = Aliceblue(user_id=user_id, api_key=api_key)

# Print AliceBlue session ID
print(alice.get_session_id())

# Initialize variables for WebSocket communication
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


# Callback functions for WebSocket connection
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


# Callback function for receiving data from WebSocket
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
subscribe_list = [alice.get_instrument_by_token('MCX', 253460)]
alice.subscribe(subscribe_list)
print(datetime.now())
sleep(10)
print(datetime.now())

# Function to calculate supertrend based on ATR
def calculate_supertrend(data, atr_period=1, multiplier=1.5, heikin_ashi=False):
    high = data['high']
    low = data['low']
    close = data['close']

    if heikin_ashi:
        # Calculate True Range (TR) for Heikin Ashi candles
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
    else:
        # Calculate True Range (TR) for normal candles
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Average True Range (ATR)
    atr = tr.rolling(atr_period).mean()

    # Calculate basic upper and lower bands
    basic_upper_band = (high + low) / 2 + multiplier * atr
    basic_lower_band = (high + low) / 2 - multiplier * atr

    # Initialize supertrend and direction columns
    supertrend = pd.Series(data=np.nan, index=data.index)
    direction = pd.Series(data=np.nan, index=data.index)

    for i in range(atr_period, len(data)):
        if close.iloc[i] > basic_upper_band.iloc[i - 1]:
            supertrend.iloc[i] = basic_lower_band.iloc[i]
            direction.iloc[i] = -1
        elif close.iloc[i] < basic_lower_band.iloc[i - 1]:
            supertrend.iloc[i] = basic_upper_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if direction.iloc[i - 1] == 1:
                supertrend.iloc[i] = basic_upper_band.iloc[i]
            else:
                supertrend.iloc[i] = basic_lower_band.iloc[i]
            direction.iloc[i] = direction.iloc[i - 1]

    # Forward-fill NaN values in the supertrend and direction columns
    supertrend.ffill(inplace=True)
    direction.ffill(inplace=True)

    # Combine supertrend and direction into a DataFrame
    supertrend_data = pd.DataFrame({'supertrend': supertrend, 'direction': direction})

    return supertrend_data


# Function to update the graph
def update_graph(n, interval, chart_type):
    global df, data_list

    # Check if there is new data
    if len(data_list) > 0:
        new_df = pd.DataFrame(data_list)
        new_df['lp'] = pd.to_numeric(new_df['lp'], errors='coerce')
        new_df = new_df.dropna(subset=['lp'])
        new_df = new_df[["timestamp", "lp"]]
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], format='%Y-%m-%d %H:%M:%S.%f')
        new_df.set_index("timestamp", inplace=True)
        df = df.append(new_df)
        df.to_csv(data_file_path)
        data_list = []

    resampled_data = df["lp"].resample(f'{interval}T').ohlc()

    if len(resampled_data) > 0:
        today_start = resampled_data.index[0].floor('D')
        previous_day_data = df["lp"].loc[df.index.floor('D') < today_start]

        if len(previous_day_data) > 0:
            previous_day_last_index = previous_day_data.index[-1]
            resampled_data = resampled_data.loc[resampled_data.index > previous_day_last_index]

    supertrend_data = calculate_supertrend(resampled_data, heikin_ashi=(chart_type == 'heikin_ashi'))

    fig = plot_candlestick(resampled_data)

    current_trend = None
    trend_start = None
    trend_lines = []
    buy_signals = pd.DataFrame(columns=['supertrend'])
    sell_signals = pd.DataFrame(columns=['supertrend'])

    for i in range(len(supertrend_data)):
        current_signal = supertrend_data.iloc[i]

        if current_trend is None:
            current_trend = current_signal['direction']
            trend_start = current_signal.name
        elif current_trend != current_signal['direction']:
            trend_data = resampled_data.loc[trend_start:current_signal.name]
            if len(trend_data) > 1:
                trend_lines.append((current_trend, trend_data))

            current_trend = current_signal['direction']
            trend_start = current_signal.name

            if current_signal['direction'] == 1:
                buy_signals = buy_signals.append(current_signal)
            else:
                sell_signals = sell_signals.append(current_signal)

    last_trend_data = resampled_data.loc[trend_start:]
    if len(last_trend_data) > 1:
        trend_lines.append((current_trend, last_trend_data))

    for trend, trend_data in trend_lines:
        color = 'red' if trend == 1 else 'green'  # Change colors here (green for uptrend, red for downtrend)
        fig.add_trace(go.Scatter(x=trend_data.index,
                                 y=supertrend_data.loc[trend_data.index, 'supertrend'],
                                 mode='lines',
                                 name='Trend Line',
                                 line=dict(color=color, width=2)))

    fig.add_trace(go.Scatter(x=buy_signals.index,
                             y=buy_signals['supertrend'],
                             mode='markers',
                             name='sell Signal',
                             marker=dict(color='red', symbol='triangle-down', size=10)))  # Change color to green

    fig.add_trace(go.Scatter(x=sell_signals.index,
                             y=sell_signals['supertrend'],
                             mode='markers',
                             name='buy Signal',
                             marker=dict(color='green', symbol='triangle-up', size=10)))  # Change color to red

    fig.write_html(graph_file_path)

    return fig


# Function to plot candlestick graph
def plot_candlestick(data):
    fig = go.Figure(data=[
        go.Candlestick(x=data.index,
                       open=data['open'],
                       high=data['high'],
                       low=data['low'],
                       close=data['close'])
    ])

    return fig


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
