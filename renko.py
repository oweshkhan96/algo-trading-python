import requests
import pandas as pd
import mplfinance as mpf

# API parameters
token = "[Your Api Token here]"
symbol = "MSFT"
from_date = "2023-01-01"
to_date = "2024-09-28"

# Function to fetch bar data
def fetch_bar_data(symbol, from_date, to_date):
    url = "https://api.benzinga.com/api/v2/bars"
    querystring = {
        "token": token,
        "symbols": symbol,
        "from": from_date,
        "to": to_date,
        "interval": "1d"  # Change to daily intervals
    }
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers, params=querystring)
    
    # Check API response status
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return None

    data = response.json()
    
    # Print the full API response for debugging
    print(data)  
    
    # Extract bar data for the specified symbol
    bars_list = next((item for item in data if item["symbol"] == symbol), None)
    
    if bars_list is None or "candles" not in bars_list:
        print(f"No data available for symbol: {symbol}")
        return None
    
    bars_df = pd.DataFrame(bars_list["candles"])
    bars_df['dateTime'] = pd.to_datetime(bars_df['dateTime'], utc=True)
    
    return bars_df

# Main execution
bars_df = fetch_bar_data(symbol, from_date, to_date)

# Check if bar data was retrieved
if bars_df is not None and not bars_df.empty:
    # Save bar data to CSV
    csv_filename = f"{symbol}_bar_data.csv"
    bars_df.to_csv(csv_filename, index=False)
    print(f"Bar data saved to {csv_filename}")

    # Prepare DataFrame for Renko chart
    renko_df = bars_df[['dateTime', 'open', 'high', 'low', 'close']].set_index('dateTime')

    # Plot the Renko chart
    mpf.plot(renko_df, type='renko', style='yahoo', title=f'Renko Chart for {symbol}', volume=False)
else:
    print("No bar data available.")
