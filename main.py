import requests
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

token = "[the api key here]"
symbol = "MSFT"
from_date = "2019-01-01"
to_date = "2024-09-28"

url = "https://api.benzinga.com/api/v2/bars"
querystring = {"token": token, "symbols": symbol, "from": from_date, "to": to_date, "interval": "1D"}
headers = {"accept": "application/json"}
response = requests.request("GET", url, headers=headers, params=querystring)


data = response.json()


bars_list = next((item for item in data if item["symbol"] == symbol), None)["candles"]
bars_df = pd.DataFrame(bars_list)
bars_df['dateTime'] = pd.to_datetime(bars_df['dateTime'], utc=True)


short_window = 50
long_window = 200
neutral_threshold = 0.01

bars_df['Short_MA'] = bars_df['close'].rolling(window=short_window, min_periods=1).mean()
bars_df['Long_MA'] = bars_df['close'].rolling(window=long_window, min_periods=1).mean()


bars_df['MA_Distance'] = bars_df['Short_MA'] - bars_df['Long_MA']


bars_df['Technical_Signal'] = np.nan


bars_df.loc[short_window:, 'Technical_Signal'] = np.where(bars_df['MA_Distance'][short_window:] > neutral_threshold, 1,
                                                          np.where(bars_df['MA_Distance'][short_window:] < -neutral_threshold, -1, 0))


bars_df = bars_df.iloc[long_window:].copy()

# Get fundamentals data
url = "https://api.benzinga.com/api/v2.1/fundamentals"
querystring = {"token": token, "symbols": symbol, "asOf": from_date}
response = requests.request("GET", url, headers=headers, params=querystring)


data = response.json()

if 'result' in data and data['result'] and 'valuationRatios' in data['result'][0]:
    valuation_ratios = []

    for ratio in data['result'][0]["valuationRatios"]:
        if isinstance(ratio, dict) and isinstance(ratio.get('id'), dict):
            valuation_ratios.append({
                'shareClassId': ratio['id'].get('shareClassId', None),
                'date': ratio['id'].get('date', None),
                'peRatio': ratio.get('peRatio', None),
                'pbRatio': ratio.get('pbRatio', None),
                'fcfRatio': ratio.get('fcfRatio', None),
                'forwardPeRatio': ratio.get('forwardPeRatio', None),
                'pegRatio': ratio.get('pegRatio', None),
                'priceChange1M': ratio.get('priceChange1M', None),
            })

    fundamentals_df = pd.DataFrame(valuation_ratios)
else:
    print("Unexpected or missing data in fundamentals response")
    fundamentals_df = pd.DataFrame()

if not fundamentals_df.empty and 'date' in fundamentals_df.columns:
    fundamentals_df['date'] = pd.to_datetime(fundamentals_df['date'], utc=True)
    bars_df.rename(columns={'dateTime': 'date'}, inplace=True)
    bars_df = bars_df.sort_values(by='date')
    fundamentals_df = fundamentals_df.sort_values(by='date')
    df_to_analyse = pd.merge_asof(bars_df, fundamentals_df, on='date', direction='backward')
else:
    print("Skipping merge due to missing 'date' data in fundamentals")
    df_to_analyse = bars_df.copy()


for column in ["peRatio", "pbRatio", "fcfRatio", "forwardPeRatio", "pegRatio"]:
    if column in df_to_analyse.columns:
        df_to_analyse[column] = 1 - MinMaxScaler().fit_transform(df_to_analyse[[column]])

if 'priceChange1M' in df_to_analyse.columns:
    df_to_analyse["priceChange1M"] = MinMaxScaler().fit_transform(df_to_analyse[["priceChange1M"]])


weights = {
    "peRatio": 0.20,
    "pbRatio": 0.20,
    "fcfRatio": 0.20,
    "forwardPeRatio": 0.15,
    "pegRatio": 0.15,
    "priceChange1M": 0.10
}

df_to_analyse["Composite Score"] = (
    df_to_analyse.get("peRatio", 0) * weights["peRatio"] +
    df_to_analyse.get("pbRatio", 0) * weights["pbRatio"] +
    df_to_analyse.get("fcfRatio", 0) * weights["fcfRatio"] +
    df_to_analyse.get("forwardPeRatio", 0) * weights["forwardPeRatio"] +
    df_to_analyse.get("pegRatio", 0) * weights["pegRatio"] +
    df_to_analyse.get("priceChange1M", 0) * weights["priceChange1M"]
)


def generate_signal(score):
    if score > 0.8:
        return 1
    elif score < 0.2:
        return -1
    else:
        return 0

df_to_analyse["Fundamental_Signal"] = df_to_analyse["Composite Score"].apply(generate_signal)

df_to_analyse['Combined_Signal'] = np.where((df_to_analyse['Technical_Signal'] == 1) & (df_to_analyse['Fundamental_Signal'] == -1), 0, df_to_analyse['Technical_Signal'])

df_to_analyse['Daily_Returns'] = df_to_analyse['close'].pct_change()
df_to_analyse['Fundamental_Strategy_Returns'] = df_to_analyse['Daily_Returns'] * df_to_analyse['Fundamental_Signal'].shift(1)
df_to_analyse['Technical_Strategy_Returns'] = df_to_analyse['Daily_Returns'] * df_to_analyse['Technical_Signal'].shift(1)
df_to_analyse['Combined_Strategy_Returns'] = df_to_analyse['Daily_Returns'] * df_to_analyse['Combined_Signal'].shift(1)
df_to_analyse['Buy_n_Hold_Returns'] = df_to_analyse['Daily_Returns'] * 1  # Shift signal to align with returns


df_to_analyse['Fundamental_Strategy_Returns'].fillna(0, inplace=True)
df_to_analyse['Technical_Strategy_Returns'].fillna(0, inplace=True)
df_to_analyse['Combined_Strategy_Returns'].fillna(0, inplace=True)
df_to_analyse['Buy_n_Hold_Returns'].fillna(0, inplace=True)


df_to_analyse['Fundamental_Strategy_Equity_Curve'] = (1 + df_to_analyse['Fundamental_Strategy_Returns']).cumprod()
df_to_analyse['Technical_Strategy_Equity_Curve'] = (1 + df_to_analyse['Technical_Strategy_Returns']).cumprod()
df_to_analyse['Combined_Strategy_Equity_Curve'] = (1 + df_to_analyse['Combined_Strategy_Returns']).cumprod()
df_to_analyse['Buy_n_Hold_Equity_Curve'] = (1 + df_to_analyse['Buy_n_Hold_Returns']).cumprod()

df = df_to_analyse.copy()


def max_drawdown(equity_curve):
    drawdown = equity_curve / equity_curve.cummax() - 1
    return drawdown.min()


def sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() * (252**0.5)

equity_curves = [
    'Fundamental_Strategy_Equity_Curve',
    'Technical_Strategy_Equity_Curve',
    'Combined_Strategy_Equity_Curve',
    'Buy_n_Hold_Equity_Curve'
]

results = {
    'Equity Curve': [],
    'Sharpe Ratio': [],
    'Max Drawdown': []
}


for curve in equity_curves:
    results['Equity Curve'].append(curve)
    results['Sharpe Ratio'].append(sharpe_ratio(df_to_analyse[curve].pct_change()))
    results['Max Drawdown'].append(max_drawdown(df_to_analyse[curve]))

results_df = pd.DataFrame(results)


print(results_df)


plt.figure(figsize=(12, 8))
for curve in equity_curves:
    plt.plot(df_to_analyse[curve], label=curve)

plt.title(f'{symbol} Strategy Comparison')
plt.legend(loc='best')
plt.show()
