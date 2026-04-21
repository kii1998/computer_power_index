import time
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots


PORTFOLIO_WEIGHTS = {
    "NVDA": 0.25,
    "TSM": 0.10,
    "AMD": 0.05,
    "AVGO": 0.10,
    "MRVL": 0.05,
    "VRT": 0.10,
    "CEG": 0.10,
    "DELL": 0.05,
    "MSFT": 0.067,
    "GOOGL": 0.067,
    "AMZN": 0.066,
}

START_DATE = "2023-01-01"
END_DATE = "2026-04-20"


def validate_weights(weights: dict[str, float]) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Weight sum must be 1.0, got {total:.6f}")


def download_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    # yfinance cache can sometimes cause issues on Windows
    # We will try to download and if it fails with 'database is locked', we rely on retries

    max_retries = 3
    data = pd.DataFrame()
    
    print(f"Start: {start}, End: {end}")
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading data (attempt {attempt + 1}/{max_retries})...")
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=True,
            )
            
            if not data.empty:
                # Handle MultiIndex check properly
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Ticker' in data.columns.names:
                        downloaded_tickers = data.columns.get_level_values('Ticker').unique()
                    else:
                        downloaded_tickers = data.columns.get_level_values(1).unique()
                else:
                    downloaded_tickers = [tickers[0]] if not data.empty else []

                missing_tickers = [t for t in tickers if t not in downloaded_tickers]
                
                # Check for incomplete data (all NaNs)
                if not missing_tickers:
                    for t in tickers:
                        if 'Close' in data and t in data['Close'] and data['Close'][t].dropna().empty:
                            missing_tickers.append(t)
                
                if not missing_tickers:
                    print("All tickers downloaded successfully.")
                    return data
                else:
                    print(f"Warning: Missing or incomplete data for {missing_tickers}.")
            else:
                print(f"Attempt {attempt + 1}: Received empty DataFrame from yfinance.")
            
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed with exception: {e}")
            
        if attempt < max_retries - 1:
            print("Retrying in 2 seconds...")
            time.sleep(2)

    if data.empty:
        raise ValueError("No data downloaded. After multiple attempts, the return was empty. Check network or date range.")
    return data


def build_gac_index(data: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    index_df = pd.DataFrame(index=data.index)
    index_df["Open"] = 0.0
    index_df["High"] = 0.0
    index_df["Low"] = 0.0
    index_df["Close"] = 0.0
    index_df["Turnover"] = 0.0

    for ticker, weight in weights.items():
        stock_open = data["Open"][ticker].ffill()
        stock_high = data["High"][ticker].ffill()
        stock_low = data["Low"][ticker].ffill()
        stock_close = data["Close"][ticker].ffill()
        stock_volume = data["Volume"][ticker].fillna(0.0)

        base_candidates = stock_close.dropna()
        if base_candidates.empty:
            raise ValueError(f"{ticker} has no valid close prices in selected range.")

        base_price = float(base_candidates.iloc[0])
        if base_price <= 0:
            raise ValueError(f"Invalid base close for {ticker}: {base_price}")

        index_df["Open"] += (stock_open / base_price) * 100 * weight
        index_df["High"] += (stock_high / base_price) * 100 * weight
        index_df["Low"] += (stock_low / base_price) * 100 * weight
        index_df["Close"] += (stock_close / base_price) * 100 * weight

        stock_turnover = stock_volume * stock_close
        index_df["Turnover"] += stock_turnover.fillna(0.0)

    index_df = index_df.dropna(subset=["Open", "High", "Low", "Close"])
    return index_df


def plot_gac_index(index_df: pd.DataFrame) -> None:
    # 计算 MACD
    close = index_df["Close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = (dif - dea) * 2

    # 计算 均线
    ma113 = close.rolling(window=113).mean()
    ma226 = close.rolling(window=226).mean()
    ma565 = close.rolling(window=565).mean()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.25, 0.15],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
        subplot_titles=("GAC-Index K线", "成交金额 (十亿美元)", "MACD")
    )

    # 主K线
    fig.add_trace(
        go.Candlestick(
            x=index_df.index,
            open=index_df["Open"],
            high=index_df["High"],
            low=index_df["Low"],
            close=index_df["Close"],
            name="GAC-Index K",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1,
        col=1,
    )

    # 添加均线
    fig.add_trace(
        go.Scatter(x=index_df.index, y=ma113, name="MA113", line=dict(color="#00BCD4", width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=index_df.index, y=ma226, name="MA226", line=dict(color="#FFEB3B", width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=index_df.index, y=ma565, name="MA565", line=dict(color="#E91E63", width=2)),
        row=1, col=1
    )

    # 成交额
    colors = [
        "green" if c >= o else "red"
        for c, o in zip(index_df["Close"], index_df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=index_df.index,
            y=index_df["Turnover"] / 1e9,
            name="Turnover (Billion USD)",
            marker_color=colors,
            opacity=0.8,
        ),
        row=2,
        col=1,
    )

    # MACD
    macd_bar_colors = ["#26A69A" if v >= 0 else "#EF5350" for v in macd_hist]
    fig.add_trace(
        go.Bar(
            x=index_df.index,
            y=macd_hist,
            name="MACD Hist",
            marker_color=macd_bar_colors,
            opacity=0.8,
            showlegend=False
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=index_df.index,
            y=dif,
            mode="lines",
            name="DIF",
            line=dict(color="#FFA726", width=1.5),
            showlegend=False
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=index_df.index,
            y=dea,
            mode="lines",
            name="DEA",
            line=dict(color="#29B6F6", width=1.5),
            showlegend=False
        ),
        row=3,
        col=1,
    )

    # 生成成分股列表文本
    weights_sorted = sorted(PORTFOLIO_WEIGHTS.items(), key=lambda x: x[1], reverse=True)
    weights_text = "<b>成分股占比 (Constituents)</b><br>"
    weights_text += "<br>".join([f"{t}: {w*100:>5.1f}%" for t, w in weights_sorted])

    fig.update_layout(
        title="GAC-Index Global AI Compute Index (2023-2026)",
        yaxis_title="指数点位 (基准=100)",
        yaxis2_title="成交金额 (Billion USD)",
        yaxis3_title="MACD",
        xaxis_rangeslider_visible=True,  # 启用范围选择器
        template="plotly_dark",
        height=1000,
        margin=dict(r=180),  # 增加右边距以放置文字
    )

    # 在右侧添加成分股比例标注
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.16,
        y=0.98,
        text=weights_text,
        showarrow=False,
        align="left",
        font=dict(family="Courier New, monospace", size=11, color="white"),
        bordercolor="#444",
        borderwidth=1,
        borderpad=4,
        bgcolor="#222",
        opacity=0.8
    )

    fig.show()


def main() -> None:
    validate_weights(PORTFOLIO_WEIGHTS)
    tickers = list(PORTFOLIO_WEIGHTS.keys())

    print(f"Downloading {len(tickers)} tickers: {tickers}")
    data = download_data(tickers, START_DATE, END_DATE)

    print("Building index...")
    index_df = build_gac_index(data, PORTFOLIO_WEIGHTS)

    print(f"Rows in index: {len(index_df)}")
    index_df.to_csv("gac_index_ohlc_turnover.csv", encoding="utf-8-sig")
    print("Saved: gac_index_ohlc_turnover.csv")

    print("Rendering chart...")
    plot_gac_index(index_df)


if __name__ == "__main__":
    main()
