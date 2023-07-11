import json
from datetime import datetime
from pytz import timezone
from time import sleep
import pandas as pd
import dash
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
data_file_path = "ohlc_datasup11.csv"
graph_file_path = "candlest_graphsup11.html"


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
subscribe_list = [alice.get_instrument_by_token('NSE', 3456)]
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
    dcc.Graph(id='live-candlestick-graph', config={'displayModeBar': False}),
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
], style={'height': '100vh'})


def calculate_supertrend(data, atr_period=10, factor=3.0):
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
        if data['close'].iloc[i - 1] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = data['lower_band'].iloc[i]
            if data['close'].iloc[i] < supertrend.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = data['upper_band'].iloc[i]
            if data['close'].iloc[i] > supertrend.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1

    return pd.DataFrame({'supertrend': supertrend, 'direction': direction})


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

        # Save the new data to CSV file
        df.to_csv(data_file_path)

        data_list = []  # Clear the data list

    # Resample the data into OHLC format using the selected interval
    resampled_data = df["lp"].resample(f'{interval}T').ohlc()

    # Calculate the supertrend values
    supertrend_data = calculate_supertrend(resampled_data)

    # Update the candlestick graph based on the chart type
    if chart_type == 'normal':
        fig = plot_candlestick(resampled_data)
    elif chart_type == 'heikin_ashi':
        fig = plot_heikin_ashi(resampled_data)

    # Plot the Supertrend
    fig = plot_supertrend(fig, supertrend_data)

    # Save the graph to HTML file
    fig.write_html(graph_file_path)

    return fig


def plot_candlestick(data):
    fig = go.Figure(data=[
        go.Candlestick(x=data.index,
                       open=data['open'],
                       high=data['high'],
                       low=data['low'],
                       close=data['close'])
    ])

    return fig


def plot_heikin_ashi(data):
    ha_open = (data['open'].shift(1) + data['close'].shift(1)) / 2
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    ha_high = data[['high', 'open', 'close']].max(axis=1)
    ha_low = data[['low', 'open', 'close']].min(axis=1)

    fig = go.Figure(data=[
        go.Candlestick(x=data.index,
                       open=ha_open,
                       high=ha_high,
                       low=ha_low,
                       close=ha_close)
    ])

    return fig


def plot_supertrend(fig, supertrend_data):
    up_trend = supertrend_data[supertrend_data['direction'] > 0]['supertrend']
    down_trend = supertrend_data[supertrend_data['direction'] < 0]['supertrend']

    fig.add_trace(go.Scatter(x=up_trend.index,
                             y=up_trend,
                             mode='lines',
                             name='Up Trend',
                             line=dict(color='green')))

    fig.add_trace(go.Scatter(x=down_trend.index,
                             y=down_trend,
                             mode='lines',
                             name='Down Trend',
                             line=dict(color='red')))

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
    app.run_server(debug=True)
