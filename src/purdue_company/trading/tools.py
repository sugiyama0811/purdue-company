import yfinance as yf
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


class StockDataInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL, TSLA, NVDA")
    period: str = Field(
        default="5d",
        description="Data period: 1d, 5d, 1mo. Use 5d for day trading context.",
    )
    interval: str = Field(
        default="15m",
        description="Candle interval: 1m, 5m, 15m, 30m, 1h. Use 15m for intraday.",
    )


class StockDataTool(BaseTool):
    name: str = "stock_data_fetcher"
    description: str = (
        "Fetches real-time OHLCV stock data, computes key technical indicators "
        "(RSI, MACD, Bollinger Bands, VWAP, ATR, volume analysis), and returns "
        "a structured summary for day trading analysis. Input a ticker symbol."
    )
    args_schema: Type[BaseModel] = StockDataInput

    def _run(self, ticker: str, period: str = "5d", interval: str = "15m") -> str:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                return f"No data returned for {ticker}. Check ticker symbol."

            latest = df.iloc[-1]
            prev_close = df.iloc[-2]["Close"] if len(df) > 1 else latest["Close"]
            change_pct = ((latest["Close"] - prev_close) / prev_close) * 100

            # RSI(14)
            delta = df["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]

            # MACD(12, 26, 9)
            ema12 = df["Close"].ewm(span=12).mean()
            ema26 = df["Close"].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            macd_hist = (macd_line - signal_line).iloc[-1]
            macd_val = macd_line.iloc[-1]
            signal_val = signal_line.iloc[-1]

            # Bollinger Bands(20)
            sma20 = df["Close"].rolling(20).mean()
            std20 = df["Close"].rolling(20).std()
            bb_upper = (sma20 + 2 * std20).iloc[-1]
            bb_lower = (sma20 - 2 * std20).iloc[-1]
            bb_mid = sma20.iloc[-1]

            # VWAP
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            vwap = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
            vwap_val = vwap.iloc[-1]

            # ATR(14)
            high_low = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift()).abs()
            low_close = (df["Low"] - df["Close"].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]

            # Volume
            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
            vol_ratio = latest["Volume"] / avg_vol if avg_vol > 0 else 1.0

            # Support / Resistance
            recent = df.tail(20)
            support = recent["Low"].min()
            resistance = recent["High"].max()

            info = stock.info
            company_name = info.get("longName", ticker)
            market_cap = info.get("marketCap", "N/A")
            sector = info.get("sector", "N/A")

            return f"""
=== STOCK DATA: {ticker} ({company_name}) ===
Sector: {sector} | Market Cap: {market_cap if market_cap == 'N/A' else f'{market_cap:,} USD'}

--- PRICE (as of {df.index[-1]}) ---
Open:  {latest['Open']:.2f}
High:  {latest['High']:.2f}
Low:   {latest['Low']:.2f}
Close: {latest['Close']:.2f}
Change: {change_pct:+.2f}%
Volume: {int(latest['Volume']):,}

--- TECHNICAL INDICATORS ---
RSI(14):         {rsi:.1f}  {'[OVERBOUGHT >70]' if rsi > 70 else '[OVERSOLD <30]' if rsi < 30 else '[NEUTRAL]'}
MACD:            {macd_val:.4f}
MACD Signal:     {signal_val:.4f}
MACD Histogram:  {macd_hist:.4f}  {'[BULLISH]' if macd_hist > 0 else '[BEARISH]'}
VWAP:            {vwap_val:.2f}  {'[PRICE ABOVE VWAP — bullish]' if latest['Close'] > vwap_val else '[PRICE BELOW VWAP — bearish]'}
ATR(14):         {atr:.2f}  (volatility measure)
Bollinger Upper: {bb_upper:.2f}
Bollinger Mid:   {bb_mid:.2f}
Bollinger Lower: {bb_lower:.2f}

--- VOLUME ---
Current Volume:  {int(latest['Volume']):,}
20-period Avg:   {int(avg_vol):,}
Volume Ratio:    {vol_ratio:.2f}x  {'[HIGH VOLUME — confirm moves]' if vol_ratio > 1.5 else '[LOW VOLUME — caution]'}

--- KEY LEVELS ---
Resistance (20-candle): {resistance:.2f}
Support    (20-candle): {support:.2f}
Price Position: {'Near resistance' if abs(latest['Close'] - resistance) < atr else 'Near support' if abs(latest['Close'] - support) < atr else 'Mid-range'}

--- RECENT CANDLES (last 5) ---
{df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5).to_string()}
"""
        except Exception as e:
            return f"Error fetching data for {ticker}: {str(e)}"
