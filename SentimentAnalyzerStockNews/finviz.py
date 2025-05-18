from finvizfinance.screener.overview import Overview

# Create screener object and set filters
foverview = Overview()
filters_dict = {'Market Cap': 'Large ($10bln to $200bln)', 'Sector': 'Technology'}
foverview.set_filter(filters_dict=filters_dict)

# Get the DataFrame of screened tickers
df = foverview.screener_view()
print(df.head())