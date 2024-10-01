import requests
import json
import os
import websocket
import threading
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time
from datetime import datetime, timedelta


url_prefix = 'https://api-capital' # For demo account use demo.api-capital
commodity = 'OIL_CRUDE'
#commodity = 'GOLD'
# For markets that are going sideways use strategy_one = True, For Bullish markets use False
strategy_one = False


# Connection setup
url = url_prefix + '.backend-capital.com/api/v1/session'


headers = {
    'X-CAP-API-KEY': 'YOUR_API_KEY', #Your API Key from Capital.com
    'Content-Type': 'application/json',
}

login_data = {
    "encryptedPassword": "false",
    "identifier": "USERNAME", #Your Username
    "password": "PASSWORD" #Password
}

# Login and retrieve session tokens
response = requests.post(url, headers=headers, json=login_data)
print("Login response status code:", response.status_code)
print("Login response text:", response.text)

if response.status_code != 200:
    raise Exception("Failed to login. Status code: {}".format(response.status_code))

cst = response.headers.get('CST')
security_token = response.headers.get('X-SECURITY-TOKEN')

# Data structures for trading logic
data_bid = deque(maxlen=100)
data_ask = deque(maxlen=100)
ema_short = deque(maxlen=100)
buy_signal = deque(maxlen=100)
sma_short = deque(maxlen=100)
sell_signal = deque(maxlen=100)
atr_values = deque(maxlen=100)
in_position = False
deal_reference = ''
buy_next = False


def calculate_ema(prices, period):
    """ Calculate exponential moving average """
    multiplier = 2 / (period + 1)
    ema = []
    for price in prices:
        if ema:
            ema.append(ema[-1] + multiplier * (price - ema[-1]))
        else:
            ema.append(price)
    return ema

def calculate_sma(prices, period):
    """ Calculate simple moving average """
    if len(prices) >= period:
        return sum(prices[-period:]) / period
    else:
        return None

def calculate_atr(data, period=14):
    """ Calculate the Average True Range (ATR) """
    if len(data) < period:
        return None

    tr_values = []
    for i in range(1, len(data)):
        high = data[i]
        low = data[i]
        prev_close = data[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)

    atr = sum(tr_values[-period:]) / period
    return atr

# Function to plot indicators including predictions
def plot_indicators(sma20, ema20, prices, predicted_diff=None):
    """ Plot SMA20, EMA20, and their predictions along with prices """
    global future_diff,buy_signal, sell_signal
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Calculate the difference between EMA20 and SMA20
    diff = [ema - sma for ema, sma in zip(ema20, sma20)]

    # Plot prices, SMA20, and EMA20 on the primary y-axis
    ax1.plot(prices, label='Price', color='black')
    ax1.plot(sma20, label='SMA 20', color='green')
    ax1.plot(ema20, label='EMA 20', color='red')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')

    # Create a secondary y-axis for the difference
    ax2 = ax1.twinx()
    ax2.plot(diff, label='Difference (EMA20 - SMA20)', color='blue')
    ax2.plot(buy_signal, label='Buy Signal', color='yellow')
    ax2.plot(sell_signal, label='Sell Signal', color='purple')

    # Plot predicted values on the secondary y-axis
    if future_diff is not None:
        extended_x = range(len(diff), len(diff) + len(future_diff))
        ax2.plot(extended_x, future_diff, label='Predicted Difference', linestyle='--', color='orange')

    ax2.set_ylabel('Difference')
    ax2.legend(loc='upper right')

    plt.title('Price, SMA20, EMA20, and Difference with Predictions')
    plt.tight_layout()
    plt.show()


def get_min_stop_distance(epic, security_token, cst_token):
    global currentprice
    url = f"https://api-capital.backend-capital.com/api/v1/markets/{epic}"
    headers = {
        'X-SECURITY-TOKEN': security_token,
        'CST': cst_token
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(data)
        min_stop_distance = data['dealingRules']['minStopOrProfitDistance']['value']
        currentprice = data['snapshot']['bid']  # Fixed the assignment issue here

        return min_stop_distance
    else:
        return None

def trade(action, size=100):
    """ Execute trade via HTTP API """
    global deal_reference

    if action.lower() == "buy":
        buy( size )
    else:  # action is "sell"
        if deal_reference:
            deal_id = get_deal_id(deal_reference)
            if deal_id:
                close_trade(deal_id)
            else:
                print("Failed to retrieve deal ID.")
        else:
            print("Trade execution failed, no deal reference returned.")

# Function to execute trade default trade size is $100
def buy(size=100):
    global currentprice, deal_reference, strategy_one

    min_stop_distance = get_min_stop_distance(commodity, security_token, cst)

    if min_stop_distance is None:
        print("Failed to retrieve min stop distance and Current Price. Exiting.")
        return

    # Leverage ratio
    leverage_ratio = 100  # 1:100 leverage

    # Calculate effective units traded
    effective_units_traded = (size / currentprice) * leverage_ratio

    # Calculate stop loss based on $20 or $10 desired loss
    if strategy_one:
        desired_loss = 10
    else:
        desired_loss = 20

    # Calculate take profit based on $7 or $30 these are based on an investement of $100
    if strategy_one:
        profit_amount = 7
    else:
        profit_amount = 30

    # Calculate the price movement equivalent for the stop loss value
    stop_distance_points = desired_loss / effective_units_traded

    # Ensure stop distance is at least the minimum stop distance
    if stop_distance_points < min_stop_distance:
        stop_distance_points = min_stop_distance

    print(f'Effective Units Traded: {effective_units_traded}')
    print(f'Stop Distance Points: {stop_distance_points}')
    print(f'Current Price: {currentprice}')
    print(f'Min Stop Distance: {min_stop_distance}')

    # Calculate stop level from the current price
    stop_level = currentprice - stop_distance_points

    """ Execute trade via HTTP API """
    trading_url = url_prefix + '.backend-capital.com/api/v1/positions'
    payload = json.dumps({
        "epic": commodity,
        "direction": "BUY",
        "size": effective_units_traded,
        "profitAmount": profit_amount,
        "stopLevel": stop_level  # Adjust stop level from current price
    })
    headers = {
        'X-SECURITY-TOKEN': security_token,
        'CST': cst,
        'Content-Type': 'application/json'
    }

    res = requests.post(trading_url, headers=headers, data=payload)
    print("Trade response status code:", res.status_code)
    print("Trade response text:", res.text)

    if res.status_code == 200:
        data = res.json()
        deal_reference = data.get('dealReference')
    else:
        print(f"Failed to execute trade. Status code: {res.json()}")

    # Plot indicators
    plot_indicators(list(sma_short), list(ema_short), list(data_bid))




def get_deal_id(dealReference):
    """ Retrieve deal ID using the epic, direction, and size """
    positions_url = url_prefix + f'.backend-capital.com/api/v1/confirms/{dealReference}'

    headers = {
        'X-SECURITY-TOKEN': security_token,
        'CST': cst,
        'Content-Type': 'application/json'
    }

    res = requests.get(positions_url, headers=headers)

    print( res.json())

    if res.status_code == 200:
        affectedDeals = res.json().get('affectedDeals', [])
        for deal in affectedDeals:
            return deal['dealId']
    else:
        print("Failed to retrieve positions. Status code: {}", res.status_code)


    return None

def close_trade(deal_id):
    """ Close trade via HTTP API """
    trading_url = url_prefix + f'.backend-capital.com/api/v1/positions/{deal_id}'

    headers = {
        'X-SECURITY-TOKEN': security_token,
        'CST': cst,
        'Content-Type': 'application/json'
    }

    print(trading_url)
    res = requests.delete(trading_url, headers=headers)
    print("Close trade response status code:", res.status_code)
    print("Close trade response text:", res.text)


# Define a function to predict future values using polynomial regression
def predict_future(values, steps=5, degree=2):
    if len(values) < 2:
        return None  # Not enough data to make a prediction

    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)

    # Transform the input data to include polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit the polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Prepare future data points
    future_X = np.arange(len(values), len(values) + steps).reshape(-1, 1)
    future_X_poly = poly.transform(future_X)

    # Predict future values
    future_y = model.predict(future_X_poly)

    return future_y

# Function to log messages to a file this is if your running this from Colab and you have your Google Drive Mounted
def log_message_to_file(message, filename="/content/drive/MyDrive/messages.log"):
    with open(filename, "a") as f:
        f.write(json.dumps(message) + "\n")

# Function to load messages from a file
def load_messages_from_file(filename="/content/drive/MyDrive/messages.log"):
    if not os.path.exists(filename):
        return []
    with open(filename, "r") as f:
        return [json.loads(line.strip()) for line in f]

# Modified on_message function
def on_message(ws, message):
    global data_bid, data_ask, in_position, ema_short, sma_short, last_price_update, future_diff, buy_next, strategy_one

    message = json.loads(message)

    # Log the message to a file
    # log_message_to_file(message)

    if message.get('destination') == 'ohlc.event':
        print(message)
        # Update the time of the last price update
        last_price_update = datetime.now()
        price_type = message['payload']['priceType']
        close_price = message['payload']['c']

        if price_type == 'bid':
            data_bid.append(close_price)
        elif price_type == 'ask':
            data_ask.append(close_price)

        # Calculate EMAs and SMAs based on bid prices
        if len(data_bid) >= 20:  # Make sure we have enough data
            ema20 = calculate_ema(list(data_bid), 20)
            sma20 = calculate_sma(list(data_bid), 20)

            if ema20:
                ema_short.append(ema20[-1])

            if sma20 is not None:
                sma_short.append(sma20)

            if len(ema_short) > 1 and len(sma_short) > 1:

                if strategy_one:
                  if buy_next:
                      trade("buy")
                      in_position = True
                      buy_next = False
                      buy_signal.append(0.005)
                      return

                if ema_short[-1] <= sma_short[-1] and ema_short[-2] >= sma_short[-2]:
                    if strategy_one and not in_position:
                        buy_next = True
                    elif not strategy_one and in_position:
                        trade("sell")
                        in_position = False
                    sell_signal.append(0.005)
                    return

                # Calculate the difference between EMA20 and SMA20
                diff = [ema - sma for ema, sma in zip(ema_short, sma_short)]

                # Use only the last 10 values of the difference
                last_diff = diff[-20:]

                future_diff = predict_future(last_diff)

                #print(f'Last 10 difference values: {last_diff}')
                #print(f'Difference Predicted: {future_diff}')

                # Plot indicators
                # plot_indicators(list(sma_short), list(ema_short), list(data_bid))

                if future_diff is not None:
                    # Predict a crossover in the next 5 steps using the difference
                    if future_diff[4] >= 0 and future_diff[0] < 0 and not in_position:
                        if not strategy_one:
                            trade("buy")
                            in_position = True
                        buy_signal.append(0.005)
                    elif future_diff[4] <= 0 and future_diff[0] > 0 and in_position:
                        if not strategy_one:
                            print("---------------- > Predicted to Sell")
                            #trade("sell")
                            #in_position = False
                        sell_signal.append(0.005)


                buy_signal.append(0)
                sell_signal.append(0)
        else:
            ema_short.append(close_price)
            sma_short.append(close_price)

# Function to test on_message using saved messages
def test_on_message(filename="/content/drive/MyDrive/messages.log"):
    messages = load_messages_from_file(filename)
    for message in messages:
        on_message(None, json.dumps(message))

# Variable to keep track of the ping thread
ping_thread = None
ping_thread_stop = threading.Event()

def on_open(ws):
    print("--------- Sending Subscription 'OHLCMarketData.subscribe' Message ---------------")
    ws.send(json.dumps({
        "destination": "OHLCMarketData.subscribe",
        "correlationId": 2,
        "cst": cst,
        "securityToken": security_token,
        "payload": {
            "epics": [commodity],
            "resolutions": ["MINUTE"],
            "type": "classic"
        }
    }))
    global ping_thread, ping_thread_stop
    if ping_thread is None or not ping_thread.is_alive():
        ping_thread_stop.clear()
        ping_thread = threading.Thread(target=ping, args=(ws, ping_thread_stop))
        ping_thread.start()

def ping(ws, stop_event):
    global in_position
    while not stop_event.is_set():
        time.sleep(60)

        if in_position:
            deal_id = get_deal_id(deal_reference)
            if deal_id:
                print(f"##### Trade Still Active ##### {deal_id}")
            else:
                in_position = False

        try:
            if ws.sock and ws.sock.connected:
                ws.send(json.dumps({
                    "destination": "ping",
                    "correlationId": 1,
                    "cst": cst,
                    "securityToken": security_token,
                }))
            else:
                print("Ping Cannot be done ws  is closed")
        except websocket.WebSocketConnectionClosedException as e:
            print("Ping Failed connection . Reconnecting...", e)
#            ws.close()
#            start_websocket()
            break

def start_websocket():
    ws = websocket.WebSocketApp(
        url="wss://api-streaming-capital.backend-capital.com/connect",
        header={"CST": cst, "X-SECURITY-TOKEN": security_token},
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()

def on_close(ws, status_code, status_message):
    global ping_thread_stop
    print(f"### WebSocket closed ### Status code: {status_code} Status message: {status_message}")
    print("Attempting to reconnect...")
    ping_thread_stop.set()  # Signal the ping thread to stop
    time.sleep(10)  # Wait for 10 seconds before reconnecting
    start_websocket()  # Attempt to restart the websocket connection

def on_error(ws, error):
    global ping_thread_stop, data_bid, data_ask, sma_short, ema_short, atr_values
    print('ERROR socket\n', error)
    ws.on_close = None
    ws.close()  # Close the existing websocket to clean up any faulty state
    print("Attempting to reconnect due to error...")
    ping_thread_stop.set()  # Signal the ping thread to stop
    # Check if the time since the last price update is more than 3 minutes
    if last_price_update and datetime.now() - last_price_update > timedelta(minutes=3):
        print("No price update for more than 3 minutes. Clearing SMA and EMA lists. And closing open trades")
        data_bid.clear()
        data_ask.clear()
        sma_short.clear()
        ema_short.clear()
        atr_values.clear()
        trade('sell')
    time.sleep(10)  # Wait for 10 seconds before reconnecting
    start_websocket()  # Attempt to restart the websocket connection

start_websocket()  # This function call starts the websocket initially
#test_on_message()  # This function call tests the on_message function with saved messages you need to save the message logs by uncommenting the lines first
