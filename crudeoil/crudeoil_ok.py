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

def calculate_heikin_ashi(data):
    ha_open = (data['open'].shift() + data['close'].shift()) / 2
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    ha_high = data[['high', 'open', 'close']].max(axis=1)
    ha_low = data[['low', 'open', 'close']].min(axis=1)

    ha_data = pd.DataFrame({'open': ha_open, 'high': ha_high, 'low': ha_low, 'close': ha_close})
    ha_data['open'] = ha_data['open'].combine_first(data['open'].shift())
    ha_data['high'] = ha_data['high'].combine_first(data['high'].shift())
    ha_data['low'] = ha_data['low'].combine_first(data['low'].shift())
    ha_data['close'] = ha_data['close'].combine_first(data['close'].shift())

    return ha_data

# Function to calculate supertrend based on ATR
def calculate_supertrend(data, atr_period=1, factor=2.0):
    data = data.copy()  # Create a copy of the data DataFrame

    close = data['close']
    high = data['high']
    low = data['low']

    tr = pd.DataFrame()
    tr['h-l'] = high - low
    tr['h-pc'] = abs(high - close.shift())
    tr['l-pc'] = abs(low - close.shift())
    tr['tr'] = tr.max(axis=1)

    atr = tr['tr'].rolling(atr_period).mean()

    data['upper_band'] = (high + low) / 2 + factor * atr
    data['lower_band'] = (high + low) / 2 - factor * atr

    supertrend = pd.Series(index=data.index)
    direction = pd.Series(index=data.index)

    supertrend.iloc[0] = data['upper_band'].iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(data)):
        if close.iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = max(data['lower_band'].iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = min(data['upper_band'].iloc[i], supertrend.iloc[i - 1])
            direction.iloc[i] = -1

    data['supertrend'] = supertrend  # Add the 'supertrend' column to the data DataFrame
    data['direction'] = direction  # Add the 'direction' column to the data DataFrame

    return data


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

    if chart_type == 'heikin_ashi':
        resampled_data = calculate_heikin_ashi(resampled_data)

    supertrend_data = calculate_supertrend(resampled_data, factor=2.0)  # Use the new factor parameter

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
                sell_signals = sell_signals.append(current_signal)
            else:
                buy_signals = buy_signals.append(current_signal)

    last_trend_data = resampled_data.loc[trend_start:]
    if len(last_trend_data) > 1:
        trend_lines.append((current_trend, last_trend_data))

    for trend, trend_data in trend_lines:
        color = 'green' if trend == 1 else 'red'  # Change colors here (green for uptrend, red for downtrend)
        fig.add_trace(go.Scatter(x=trend_data.index,
                                 y=supertrend_data.loc[trend_data.index, 'supertrend'],
                                 mode='lines',
                                 name='Trend Line',
                                 line=dict(color=color, width=2)))

    fig.add_trace(go.Scatter(x=sell_signals.index,
                             y=sell_signals['supertrend'],
                             mode='markers',
                             name='sell Signal',
                             marker=dict(color='red', symbol='triangle-down', size=10)))  # Change color to green

    fig.add_trace(go.Scatter(x=buy_signals.index,
                             y=buy_signals['supertrend'],
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
    dcc.Interval(id='graph-update-interval', interval=800, n_intervals=0)
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
