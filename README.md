# Capital.com Trading Bot with Indicators and Polynomial Regression Predictions

This repository contains a Python script designed to run a trading bot on Capital.com using the WebSocket API for live price data and the HTTP API for trade execution. The bot implements trading strategies using simple and exponential moving averages (SMA and EMA) along with polynomial regression for price predictions. This version of the script is optimized to be run on Google Colab, with optional logging capabilities to Google Drive.

## Features
- **Moving Average Crossovers**: Uses SMA20 and EMA20 for crossover trading signals.
- **Buy/Sell Logic**: Implements trading logic based on crossovers and predicted trends.
- **ATR Calculation**: Includes Average True Range (ATR) to help with stop-loss placement.
- **Polynomial Regression**: Predicts future price trends using polynomial regression.
- **Trade Execution**: Buys and sells commodities on Capital.com.
- **WebSocket**: Streams live market data in real-time.

## Prerequisites

### 1. Sign Up on Capital.com
You need to sign up for a [Capital.com](https://www.capital.com) account and generate an API key to access their API services.

### 2. Obtain Your API Key
- Log in to your Capital.com account.
- Go to **Settings > API** to generate your API key.

### 3. Google Colab Setup
This script can be run on **Google Colab**. Colab provides an easy way to run Python code in a cloud environment without any setup on your local machine.

### 4. Google Drive (Optional)
If you'd like to log WebSocket messages to your Google Drive, you'll need to mount your Google Drive on Colab.

### Required Libraries
To run this script, you will need the following Python libraries:
- `requests`
- `json`
- `os`
- `websocket-client`
- `threading`
- `matplotlib`
- `numpy`
- `scikit-learn`

### Colab-Specific Setup
Colab may not have some of these libraries pre-installed, so the following installation commands can be added at the top of your Colab notebook:

```python
!pip install websocket-client
!pip install scikit-learn
```

## How to Run the Script on Google Colab

### Step 1: Create a Colab Notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.

### Step 2: Install Required Libraries

Add the following cells at the beginning of your Colab notebook to install necessary libraries:

```python
!pip install websocket-client
!pip install scikit-learn
````
### Step 3: Mount Google Drive (Optional)

If you want to log WebSocket messages to your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
````
### Step 4: Paste the Script Code

Copy and paste the entire code provided in this repository into a Colab cell.

### Step 5: Replace the Credentials

Replace the following placeholders with your actual values in the script:
- `YOUR_API_KEY`: Your API key from Capital.com.
- `USERNAME`: Your Capital.com username.
- `PASSWORD`: Your Capital.com password.

## Trading Strategies

The bot supports two trading strategies:

1. **Strategy One** (`strategy_one = True`): 
   - Optimized for sideways markets.
   - Uses smaller stop losses and take profits.
   
2. **Strategy Two** (`strategy_one = False`): 
   - Optimized for bullish markets.
   - Uses larger stop losses and take profits.

You can switch between these strategies by toggling the `strategy_one` flag in the script.

## License

This project is for educational purposes and comes with no guarantees of profitability. Use it at your own risk. Make sure to thoroughly test with a demo account before attempting live trading.


