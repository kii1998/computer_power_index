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
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise ValueError("No data downloaded. Check tickers, date range, or network.")
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
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

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

    fig.update_layout(
        title="GAC-Index Global AI Compute Index (2023-2026)",
        yaxis_title="Index Level (Base=100)",
        yaxis2_title="Turnover (Billion USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=850,
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
