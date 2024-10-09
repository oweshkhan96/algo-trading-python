import requests
import pandas as pd
import numpy as np
import mplfinance as mpf

# API parameters
token = "[Your API token here]"
symbol = "MSFT"
from_date = "2023-01-01"
to_date = "2024-09-28"

# Function to fetch bar data
def fetch_bar_data(symbol, from_date, to_date):
    url = "https://api.benzinga.com/api/v2/bars"
    querystring = {"token": token, "symbols": symbol, "from": from_date, "to": to_date, "interval": "1D"}
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    
    # Extract bar data for the specified symbol
    bars_list = next((item for item in data if item["symbol"] == symbol), None)
    
    if bars_list is None or "candles" not in bars_list:
        print(f"No data available for symbol: {symbol}")
        return None
    
    bars_df = pd.DataFrame(bars_list["candles"])
    bars_df['dateTime'] = pd.to_datetime(bars_df['dateTime'], utc=True)
    
    return bars_df

# SuperTrend calculation
def calculate_supertrend(df, period=10, multiplier=3):
    # Calculate the True Range (TR) for ATR calculation
    df['TR'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=period).mean()  # Use a proper ATR calculation
    
    # Calculate the Super Trend Bands
    df['Upper Band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['ATR'])
    df['Lower Band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['ATR'])
    
    # Initialize Super Trend
    df['Super Trend'] = np.nan
    
    # Calculate Super Trend
    for i in range(1, len(df)):
        if df['close'].iloc[i] <= df['Upper Band'].iloc[i - 1]:
            df.at[df.index[i], 'Super Trend'] = df['Upper Band'].iloc[i]
        else:
            df.at[df.index[i], 'Super Trend'] = df['Lower Band'].iloc[i]

    # Replace NaN values with the last valid observation
    df['Super Trend'] = df['Super Trend'].ffill()  # Forward fill to replace NaN

    return df

# Main execution
bars_df = fetch_bar_data(symbol, from_date, to_date)

if bars_df is not None and not bars_df.empty:
    print("Data available for plotting after calculations.")
    
    # Set 'dateTime' as the DataFrame index
    bars_df.set_index('dateTime', inplace=True)
    
    # Calculate Super Trend
    supertrend_df = calculate_supertrend(bars_df)
    
    # Save bar data and Super Trend to CSV
    csv_filename = f"{symbol}_bar_data.csv"
    supertrend_df.to_csv(csv_filename, index=True)
    print(f"Data (including Super Trend) saved to {csv_filename}")

    # Plot SuperTrend
    mpf.plot(
        supertrend_df[['open', 'high', 'low', 'close']],
        type='candle',
        style='yahoo',
        title=f"SuperTrend for {symbol}",
        volume=False,
        addplot=mpf.make_addplot(supertrend_df['Super Trend'], color='green')
    )
else:
    print("No data available for plotting.")
