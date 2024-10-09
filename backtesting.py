import pandas as pd
import numpy as np

# Load historical data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df.set_index('dateTime', inplace=True)
    return df

# Calculate Average True Range (ATR)
def calculate_atr(df, period=10):
    # Calculate the true range components
    df['high-low'] = df['high'] - df['low']
    df['high-prevClose'] = abs(df['high'] - df['close'].shift(1))
    df['low-prevClose'] = abs(df['low'] - df['close'].shift(1))
    
    # Calculate the ATR
    true_range = df[['high-low', 'high-prevClose', 'low-prevClose']].max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

# Calculate Super Trend
def calculate_super_trend(df, period=7, multiplier=2):  # Adjusted parameters for sensitivity
    df['ATR'] = calculate_atr(df, period)
    df['Upper Band'] = df['close'].shift(1) + (multiplier * df['ATR'].shift(1))
    df['Lower Band'] = df['close'].shift(1) - (multiplier * df['ATR'].shift(1))
    
    df['Super Trend'] = np.nan  # Initialize with NaN values
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['Upper Band'].iloc[i - 1]:
            df.loc[df.index[i], 'Super Trend'] = df['Upper Band'].iloc[i]
        elif df['close'].iloc[i] < df['Lower Band'].iloc[i - 1]:
            df.loc[df.index[i], 'Super Trend'] = df['Lower Band'].iloc[i]
        else:
            df.loc[df.index[i], 'Super Trend'] = df['Super Trend'].iloc[i - 1]
    return df

# Backtesting function
def backtest_strategy(df):
    initial_capital = 10000  # Starting capital
    capital = initial_capital
    position = 0
    trades = []

    for i in range(1, len(df)):
        # Buy Signal
        if df['close'].iloc[i] > df['Super Trend'].iloc[i] and position == 0:
            print(f"Buy signal on {df.index[i]}: {df['close'].iloc[i]:.2f}")  # Debugging line
            position = capital / df['close'].iloc[i]  # Buy as many shares as possible
            trades.append((df.index[i], 'BUY', df['close'].iloc[i], position))
            capital = 0  # Invest all capital
        
        # Sell Signal
        elif df['close'].iloc[i] < df['Super Trend'].iloc[i] and position > 0:
            print(f"Sell signal on {df.index[i]}: {df['close'].iloc[i]:.2f}")  # Debugging line
            capital = position * df['close'].iloc[i]  # Sell all shares
            trades.append((df.index[i], 'SELL', df['close'].iloc[i], position))
            position = 0  # Clear position

    # Calculate final capital
    if position > 0:  # If still holding shares at the end
        capital = position * df['close'].iloc[-1]
    
    total_return = capital - initial_capital
    return total_return, trades

# Main execution
if __name__ == "__main__":
    # Load your historical data from a CSV file
    file_path = 'MSFT_bar_data.csv'  # Update this to your CSV file
    df = load_data(file_path)

    # Calculate Super Trend
    df = calculate_super_trend(df)
    print(df[['close', 'Super Trend']].head(20))  # Inspect the calculated values

    # Backtest the strategy
    total_return, trades = backtest_strategy(df)

    # Output results
    
    print("Trades:")
    for trade in trades:
        print(f"{trade[0]}: {trade[1]} at ${trade[2]:.2f}, Shares: {trade[3]:.2f}")
    print(f"\nTotal Return: ${total_return:.2f}")
