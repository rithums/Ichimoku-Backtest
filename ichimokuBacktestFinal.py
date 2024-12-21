import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np
from backtesting.test import SMA
from backtesting.test import GOOG
from statistics import mean 
import matplotlib.pyplot as plt

ticker_symbols = [
    'EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X', # Initial list
    'EURJPY=X', 'EURGBP=X', # Additional tickers
    'GBPJPY=X', 'EURCAD=X', 'EURAUD=X', 

]

ticker_symbols = ['EURUSD=X']
ticker_symbol = []



# Defining the strategy


def IchimokuCloud(high, low, close):
    """Calculate Ichimoku Cloud components."""
    # Conversion Line (α)
    tenkan_sen = (pd.Series(high).rolling(window=9).max() + pd.Series(low).rolling(window=9).min()) / 2
    # Base Line (β)
    kijun_sen = (pd.Series(high).rolling(window=26).max() + pd.Series(low).rolling(window=26).min()) / 2
    
    # The cloud (Senkou Span A and B) and the lagging span (Chikou Span) can be calculated here if needed for a full strategy.
    
    return tenkan_sen, kijun_sen

class IchimokuStrategy(Strategy):
    tenkan_sen = 9
    kijun_sen = 26

    def init(self):
        # Precompute Ichimoku components
        high, low, close = self.data.High, self.data.Low, self.data.Close
        self.tenkan_sen, self.kijun_sen = self.I(IchimokuCloud, high, low, close)
    
    def next(self):
        # Assuming we're only dealing with buy/sell orders and not considering existing positions for simplicity
        # Buy if conversion line (α) is above base line (β)
        if crossover(self.tenkan_sen, self.kijun_sen):
            if not self.position:
                self.buy()
        
        # Sell if base line (β) is above conversion line (α)
        elif crossover(self.kijun_sen, self.tenkan_sen):
            if not self.position.is_short:
                self.sell()
        
        # If an order is open and the opposite crossover happens, close the order.
        # The strategy in the algorithm assumes that there's a mechanism outside of this logic to count orders,
        # but in backtesting.py, you'd usually check if there's an open position.
        if self.position.is_long and crossover(self.kijun_sen, self.kijun_sen):
            self.position.close()
        elif self.position.is_short and crossover(self.kijun_sen, self.kijun_sen):
            self.position.close()



# Example backtest setup, replace with your own data and settings

backtest = Backtest(GOOG, IchimokuStrategy, cash=10_000, commission=.002)
stats = backtest.run()

print(stats)
backtest.plot()

# Backtest setup

# Function for walk-forward cross-validation
def walk_forward_validation(symbol, start_date, end_date, interval='1h', window_size=60, step_size=30):
    # Fetch the data
    dataset = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    
    # Store metrics
    sharpe_ratios = []
    max_drawdowns = []
    final_equities = []
    win_rates = []
    sortino_ratios = []
    calmar_ratios = []

    # Rolling window cross-validation
    for start in range(0, len(dataset) - window_size, step_size):
        train_data = dataset.iloc[start:start + window_size]
        test_data = dataset.iloc[start + window_size:start + window_size + step_size]

        # Skip if there's not enough data for testing
        if len(test_data) == 0:
            continue

        # Backtest on training data
        bt = Backtest(train_data, IchimokuStrategy, cash=10000, commission=.002, exclusive_orders=True)
        result = bt.optimize(
            tenkan_sen=range(7, 12, 1),
            kijun_sen=range(20, 31, 1),
            constraint=lambda p: all([p.tenkan_sen > 0, p.kijun_sen > 0]),
            maximize='Equity Final [$]'
        )

        # Backtest on test data with optimized parameters
        bt_test = Backtest(test_data, IchimokuStrategy, cash=10000, commission=.002, exclusive_orders=True)
        test_result = bt_test.run(tenkan_sen=result._strategy.tenkan_sen, kijun_sen=result._strategy.kijun_sen)

        # Store metrics
        sharpe_ratios.append(test_result['Sharpe Ratio'])
        max_drawdowns.append(test_result['Max. Drawdown [%]'])
        final_equities.append(test_result['Equity Final [$]'])
        win_rates.append(test_result['Win Rate [%]'])
        sortino_ratios.append(test_result['Sortino Ratio'])
        calmar_ratios.append(test_result['Calmar Ratio'])

    # Plot the test equity curve
    equity = test_result['_equity_curve']['Equity']
    plt.figure(figsize=(10, 5))
    plt.plot(equity)
    plt.title(f"Equity Curve: {symbol} Walk-Forward Step")
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()

    # Print average metrics across all test periods
    print(f"Symbol: {symbol}")
    print(f"Average Sharpe Ratio: {mean(sharpe_ratios):.4f}")
    print(f"Average Max Drawdown: {mean(max_drawdowns):.2f}%")
    print(f"Average Final Equity: ${mean(final_equities):.2f}")


for i in ticker_symbols:

    # Fetching data
    dataset = yf.download(i, start='2023-06-01', end='2024-01-01', interval='1h')
    split_date = '2023-08-01'
    data_df_insample = dataset[:split_date]
    data_df_outsample = dataset[split_date:]
    
    #backtest = Backtest(data_df_insample, IchimokuStrategy, cash=10000, commission=.002)
    walk_forward_validation(i, '2023-06-01', '2024-01-01', interval='1h', window_size=100, step_size=50)

sharpRatiosIS = []
maxDrawdownsIS = []
equityFinalsIS = []
winRatesIS = []
sortinoRatiosIS = []
calmarRatiosIS = []

sharpRatiosOS = []
maxDrawdownsOS = []
equityFinalsOS = []
winRatesOS = []
sortinoRatiosOS = []
calmarRatiosOS = []

for i in ticker_symbols:

    # Fetching data
    dataset = yf.download(i, start='2023-06-01', end='2024-01-01', interval='1h')
    split_date = '2023-08-01'
    data_df_insample = dataset[:split_date]
    data_df_outsample = dataset[split_date:]

    bt = Backtest(data_df_insample, IchimokuStrategy, cash=10000, commission=.002, exclusive_orders=True)

# Optimization

    result = bt.optimize(
        tenkan_sen=range(7, 12, 1),  # Testing periods from 10 to 45 in steps of 5
        kijun_sen=range(20, 31,1),
        constraint=lambda p: all([p.tenkan_sen > 0, p.kijun_sen > 0]),
        # Testing multipliers from 1.0 to 3.0 in steps of 0.5
        maximize='Equity Final [$]'
        # return_heatmap=True # Make sure constraints are satisfied
    )

    print("optimizing:", result)
    print("\n\n\n")


    sharpRatiosIS.append(result['Sharpe Ratio'])
    maxDrawdownsIS.append(result['Max. Drawdown [%]'])
    sortinoRatiosIS.append(result['Sortino Ratio'])
    equityFinalsIS.append(result['Equity Final [$]'])
    winRatesIS.append(result['Win Rate [%]'])
    calmarRatiosIS.append(result['Calmar Ratio'])

    optimizedTenkan = result._strategy.tenkan_sen
    optimizedKijun = result._strategy.kijun_sen


    equity1 = result['_equity_curve']['Equity']

    # Plot the equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(equity1)
    plt.title("Ichimoku Equity Curve Optimized on Insample")
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()

    btO = Backtest(data_df_outsample, IchimokuStrategy, cash=10000, commission=.002)


   

    outSampleRun = btO.run(tenkan_sen = optimizedTenkan, kijun_sen = optimizedKijun)
    print("OUTSAMPLE: ",outSampleRun)

    sharpRatiosOS.append(outSampleRun['Sharpe Ratio'])
    maxDrawdownsOS.append(outSampleRun['Max. Drawdown [%]'])
    sortinoRatiosOS.append(outSampleRun['Sortino Ratio'])
    equityFinalsOS.append(outSampleRun['Equity Final [$]'])
    winRatesOS.append(outSampleRun['Win Rate [%]'])
    calmarRatiosOS.append(outSampleRun['Calmar Ratio'])

    equity = outSampleRun['_equity_curve']['Equity']

    # Plot the equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(equity)
    plt.title("Ichimoku Out Sample Equity Curve")
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()



    

print("Average Sharpe Ratio ", mean(sharpRatiosIS),mean(sharpRatiosOS) )
print("Average Max Drawdown ", mean(maxDrawdownsIS), mean(maxDrawdownsOS))
print("Average Sortino Ratio", mean(sortinoRatiosIS), mean(sortinoRatiosOS))
print("Average Final Equity", mean(equityFinalsIS), mean(equityFinalsOS))
print("Average Win Rate", mean(winRatesIS), mean(winRatesOS))
print("Average Calmar Ratio", mean(calmarRatiosIS), mean(calmarRatiosOS))




        
