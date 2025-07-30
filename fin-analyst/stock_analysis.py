import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Calculate date range (6 years from today)
end_date = datetime.today()
start_date = end_date - timedelta(days=6*365)

print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download stock data
print("Downloading PLTR data...")
pltr = yf.download('PLTR', start=start_date, end=end_date, progress=False)
print(f"PLTR data points: {len(pltr)}")

print("Downloading TSLA data...")
tsla = yf.download('TSLA', start=start_date, end=end_date, progress=False)
print(f"TSLA data points: {len(tsla)}")

if len(pltr) == 0:
    print("No PLTR data available for the specified date range.")
    # Try from IPO date
    pltr = yf.download('PLTR', start='2020-09-30', end=end_date, progress=False)
    print(f"PLTR data from IPO: {len(pltr)} points")

if len(tsla) == 0:
    print("No TSLA data available for the specified date range.")
else:
    print("TSLA data loaded successfully")

# Find common date range
common_start = max(pltr.index.min(), tsla.index.min())
print(f"Common start date: {common_start}")

# Filter to common date range
pltr_common = pltr[pltr.index >= common_start]
tsla_common = tsla[tsla.index >= common_start]

# Calculate normalized prices for comparison
pltr_common['Normalized'] = pltr_common['Close'] / pltr_common['Close'].iloc[0] * 100
tsla_common['Normalized'] = tsla_common['Close'] / tsla_common['Close'].iloc[0] * 100

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle('PLTR vs TSLA Stock Performance Comparison', fontsize=16, y=0.95)

# Plot normalized prices
ax1.plot(pltr_common.index, pltr_common['Normalized'], label='PLTR', color='#2c5c8a', linewidth=2)
ax1.plot(tsla_common.index, tsla_common['Normalized'], label='TSLA', color='#c7254e', linewidth=2)
ax1.set_ylabel('Normalized Price (%)', fontsize=12)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot volume comparison
ax2.plot(pltr_common.index, pltr_common['Volume'], color='#2c5c8a', alpha=0.7, label='PLTR Volume', linewidth=1)
ax2.plot(tsla_common.index, tsla_common['Volume'], color='#c7254e', alpha=0.7, label='TSLA Volume', linewidth=1)
ax2.set_ylabel('Trading Volume', fontsize=12)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# Format x-axis
plt.xticks(rotation=45)
plt.xlabel('Date', fontsize=12)
plt.tight_layout()
plt.show()

# Performance analysis
print(f"\n{'='*50}")
print("PERFORMANCE SUMMARY")
print(f"{'='*50}")
print(f"Analysis Period: {common_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

print(f"\nPALANTIR (PLTR):")
print(f"Start Price: ${float(pltr_common['Close'].iloc[0]):.2f}")
print(f"End Price: ${float(pltr_common['Close'].iloc[-1]):.2f}")
print(f"Total Return: {float(pltr_common['Normalized'].iloc[-1] - 100):.2f}%")

print(f"\nTESLA (TSLA):")
print(f"Start Price: ${float(tsla_common['Close'].iloc[0]):.2f}")
print(f"End Price: ${float(tsla_common['Close'].iloc[-1]):.2f}")
print(f"Total Return: {float(tsla_common['Normalized'].iloc[-1] - 100):.2f}%")

# Additional metrics
pltr_volatility = float(pltr_common['Close'].pct_change().std() * (252**0.5) * 100)
tsla_volatility = float(tsla_common['Close'].pct_change().std() * (252**0.5) * 100)

print(f"\nVOLATILITY (Annualized):")
print(f"PLTR: {pltr_volatility:.2f}%")
print(f"TSLA: {tsla_volatility:.2f}%")

print(f"\n{'='*50}")
