"""Download the public market data into data/ (FRED, ECB, Yahoo Finance)."""
from pathlib import Path
from urllib.request import urlopen

import yfinance as yf

D = Path(__file__).resolve().parents[1] / "data"
D.mkdir(exist_ok=True)
for s in ["DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30", "BAA10Y"]:
    (D / f"{s}.csv").write_bytes(urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}").read())
for m in ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]:
    url = f"https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{m}?format=csvdata"
    rows = urlopen(url).read().decode().splitlines()
    (D / f"ecb_SR_{m}.csv").write_text("\n".join(",".join(r.split(",")[8:10]) for r in rows) + "\n")
yf.download(["^GSPC", "EURUSD=X"], start="2000-01-01", auto_adjust=True, progress=False)["Close"].to_csv(
    D / "yahoo_spx_eurusd.csv")
print("data written to", D)
