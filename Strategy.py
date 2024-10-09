import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

# Function to fetch bar data from the API
def fetch_bar_data(token, symbol, from_date, to_date):
    url = "https://api.benzinga.com/api/v2/bars"
    querystring = {
        "token": token,
        "symbols": symbol,
        "from": from_date,
        "to": to_date,
        "interval": "1D"
    }
    headers = {"accept": "application/json"}
    
    # Fetch data from API
    response = requests.get(url, headers=headers, params=querystring)
    
    # Check response status
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return None

    # Parse JSON data
    data = response.json()
    bars_list = next((item for item in data if item["symbol"] == symbol), None)
    
    if bars_list is None or "candles" not in bars_list:
        print(f"No data available for symbol: {symbol}")
        return None
    
    bars_df = pd.DataFrame(bars_list["candles"])
    bars_df['dateTime'] = pd.to_datetime(bars_df['dateTime'], utc=True)
    return bars_df

# Function to read data from CSV
def read_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    return df

# Calculate Average True Range (ATR)
def calculate_atr(df, period=10):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

# Calculate Super Trend
def calculate_super_trend(df, period=7, multiplier=2):
    df['ATR'] = calculate_atr(df, period)
    df['Upper Band'] = df['close'].shift(1) + (multiplier * df['ATR'])
    df['Lower Band'] = df['close'].shift(1) - (multiplier * df['ATR'])

    df['Super Trend'] = np.nan  # Initialize Super Trend with NaN
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['Upper Band'].iloc[i - 1]:
            df['Super Trend'].iloc[i] = df['Upper Band'].iloc[i]
        elif df['close'].iloc[i] < df['Lower Band'].iloc[i - 1]:
            df['Super Trend'].iloc[i] = df['Lower Band'].iloc[i]
        else:
            df['Super Trend'].iloc[i] = df['Super Trend'].iloc[i - 1] if not np.isnan(df['Super Trend'].iloc[i - 1]) else df['Upper Band'].iloc[i]

    return df

# Calculate Renko
def calculate_renko(df, brick_size=5):
    renko_data = []
    last_price = df['close'].iloc[0]

    for index, row in df.iterrows():
        current_price = row['close']
        while current_price >= last_price + brick_size:
            last_price += brick_size
            renko_data.append({'dateTime': row['dateTime'], 'close': last_price, 'color': 'green'})
        while current_price <= last_price - brick_size:
            last_price -= brick_size
            renko_data.append({'dateTime': row['dateTime'], 'close': last_price, 'color': 'red'})

    return pd.DataFrame(renko_data)

# Simulate trades based on Super Trend and Renko indicators
def simulate_trades(df, initial_capital=10000):
    capital = initial_capital
    position = 0  # Current position in shares
    trades = []  # Store trade records

    for i in range(1, len(df)):
        # Determine the last Renko brick color
        last_renko_color = 'green' if df['Renko'].iloc[i] > df['Renko'].iloc[i-1] else 'red'

        # Buy signal
        if df['close'].iloc[i] > df['Super Trend'].iloc[i] and last_renko_color == 'green' and position == 0:
            position = capital / df['close'].iloc[i]  # Buy as many shares as possible
            capital = 0  # Invest all capital
            trades.append({'dateTime': df['dateTime'].iloc[i], 'action': 'BUY', 'price': df['close'].iloc[i], 'shares': position})

        # Sell signal
        elif df['close'].iloc[i] < df['Super Trend'].iloc[i] and last_renko_color == 'red' and position > 0:
            capital = position * df['close'].iloc[i]  # Sell all shares
            trades.append({'dateTime': df['dateTime'].iloc[i], 'action': 'SELL', 'price': df['close'].iloc[i], 'shares': position})
            position = 0  # Clear position

    # If still holding shares at the end, sell at the last price
    if position > 0:
        capital = position * df['close'].iloc[-1]  # Sell at last price
        trades.append({'dateTime': df['dateTime'].iloc[-1], 'action': 'SELL', 'price': df['close'].iloc[-1], 'shares': position})

    total_return = capital - initial_capital
    return trades, total_return

# Main execution
if __name__ == "__main__":
    # Assuming the API token, symbol, and date range for fetching data
    token = "[Your api token here]"  # Replace with your actual token
    symbol = "MSFT"
    from_date = "2023-01-01"
    to_date = "2023-12-31"
    
    # Fetch bar data from the API
    bars_df = fetch_bar_data(token, symbol, from_date, to_date)

    # If no data is fetched, exit
    if bars_df is None:
        exit()

    # Calculate indicators
    bars_df = calculate_super_trend(bars_df)
    renko_df = calculate_renko(bars_df)

    # Merge Renko information into the main DataFrame
    bars_df['Renko'] = renko_df['close']  # Assuming 'close' contains the Renko prices

    # Simulate trades
    trades, total_return = simulate_trades(bars_df)

    # Create a DataFrame for trades and save to CSV
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('trading_results.csv', index=False)

    # Output results
    print(f"Total Return: ${total_return:.2f}")
    print("Trades:")
    print(trades_df)

    # Create a figure and axis for plotting
    plt.figure(figsize=(14, 7))

    # Plot the Super Trend
    plt.plot(bars_df['dateTime'], bars_df['Super Trend'], label='Super Trend', color='blue', linewidth=2)

    # Plot the close price
    plt.plot(bars_df['dateTime'], bars_df['close'], label='Close Price', color='orange')

    # Plot the Renko bricks
    for _, row in renko_df.iterrows():
        color = 'green' if row['color'] == 'green' else 'red'
        plt.bar(row['dateTime'], height=row['close'], width=0.5, color=color)

    # Formatting the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.gcf().autofmt_xdate()

    # Adding labels and legend
    plt.title('Trading Strategy Visualization with Renko and Super Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()
