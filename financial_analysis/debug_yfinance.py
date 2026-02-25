
import yfinance as yf
import pandas as pd

df = yf.download("IBM",
                 start="2020-01-01",
                 end="2020-12-31",
                 progress=False,
                 auto_adjust=True)

print("Columns:", df.columns)
print("Type of df['Close']:", type(df["Close"]))
print("df['Close'] shape:", df["Close"].shape)
print("Head of df['Close']:\n", df["Close"].head())

try:
    import talib
    print("Trying talib.SMA...")
    talib.SMA(df["Close"], timeperiod=20)
    print("talib.SMA worked!")
except Exception as e:
    print(f"talib.SMA failed: {e}")

# Check if accessing as series works
if isinstance(df["Close"], pd.DataFrame):
     print("It is a DataFrame. Trying to flatten/squeeze...")
     try:
         s = df["Close"].squeeze()
         print("Squeezed type:", type(s))
         talib.SMA(s, timeperiod=20)
         print("talib.SMA with squeeze worked!")
     except Exception as e:
         print(f"talib.SMA with squeeze failed: {e}")
