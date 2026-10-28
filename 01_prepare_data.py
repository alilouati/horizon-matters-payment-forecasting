"""
01_prepare_data.py — raw SAMA export -> model-ready monthly series.

Input : data/pos-transactions.csv  (4,689 records; monthly, quarterly, annual)
Output: data/monthly_all_indicators.csv  (all nine indicators, monthly, SAR units)
        data/monthly_series.csv          (period_start, pos_sales_SAR, split)

Steps (identical to the original preprocessing script, plus the split assignment):
  1. rename columns, parse dates and numeric values
  2. map the nine SAMA indicator names to short identifiers
  3. build a period start date for every record
  4. drop duplicate (frequency, period, indicator) records, keeping the last
  5. convert sales indicators from thousand SAR to SAR (x 1000)
  6. keep monthly records, pivot to wide format
  7. assign train / val / test by date (see common.py)
"""
import numpy as np
import pandas as pd
from common import RAW_FILE, SERIES_FILE, WIDE_FILE, TARGET, TRAIN_END, VAL_END

INDICATORS = {
    "Total POS :  Sales (In Thousand Riyals)": "pos_sales",
    "Total POS :  Number of Transactions": "pos_transactions",
    "Total POS :  Number of Points of Sale Terminals": "pos_terminals",
    "POS Transactions Using Near Field Communication Technology : Number of Transactions :  Mobile Transactions": "nfc_tx_mobile",
    "POS Transactions Using Near Field Communication Technology : Number of Transactions :  Cards Transactions": "nfc_tx_card",
    "POS Transactions Using Near Field Communication Technology : Sales (In Thousand Riyals) :  Sales Using Mobile": "nfc_sales_mobile",
    "POS Transactions Using Near Field Communication Technology : Sales (In Thousand Riyals) :  Sales Using Cards": "nfc_sales_card",
    "E-Commerce Transactions Using Mada Cards : Sales (In Thousand Riyals)": "mada_ecom_sales",
    "E-Commerce Transactions Using Mada Cards : Number of Transactions": "mada_ecom_transactions",
}
SALES = {"pos_sales", "nfc_sales_mobile", "nfc_sales_card", "mada_ecom_sales"}


def main():
    df = pd.read_csv(RAW_FILE, encoding="utf-8-sig")
    print(f"raw records: {len(df)}")
    df = df.rename(columns={"Value in Different Units": "value", "Periodicity": "frequency"})
    df["frequency"] = df["frequency"].map({"Monthly": "monthly", "Quarterly": "quarterly",
                                           "Annually": "annual"})
    for c in ["Year", "Quarter", "Month"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["indicator"] = df["Indicator"].map(INDICATORS)
    unknown = df["indicator"].isna().sum()
    if unknown:
        raise ValueError(f"{unknown} records with unrecognised indicator names")

    m = df[df["frequency"] == "monthly"].copy()
    # use Year/Month where present; otherwise fall back to the Date_Object column
    ok = m["Year"].notna() & m["Month"].notna()
    m["period_start"] = pd.to_datetime(m["Date_Object"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    m.loc[ok, "period_start"] = pd.to_datetime(
        dict(year=m.loc[ok, "Year"].astype(int), month=m.loc[ok, "Month"].astype(int), day=1))
    m = m.dropna(subset=["period_start"])
    m = m.drop_duplicates(subset=["period_start", "indicator"], keep="last")
    m.loc[m["indicator"].isin(SALES), "value"] *= 1000.0          # thousand SAR -> SAR
    m["indicator"] = np.where(m["indicator"].isin(SALES), m["indicator"] + "_SAR", m["indicator"])

    wide = (m.pivot_table(index="period_start", columns="indicator", values="value", aggfunc="last")
              .sort_index().reset_index())
    wide.to_csv(WIDE_FILE, index=False)

    # target plus the two series used in the additional experiments (second series, covariates)
    s = wide[["period_start", TARGET, "pos_transactions", "pos_terminals"]].dropna().copy()
    # the series must be a gap-free monthly sequence
    expected = pd.date_range(s["period_start"].min(), s["period_start"].max(), freq="MS")
    if len(expected) != len(s):
        raise ValueError("monthly target series has gaps")
    s["split"] = np.select(
        [s["period_start"] <= TRAIN_END, s["period_start"] <= VAL_END], ["train", "val"], "test")
    s.to_csv(SERIES_FILE, index=False)

    print(f"monthly target series: {len(s)} months, "
          f"{s['period_start'].min():%Y-%m} to {s['period_start'].max():%Y-%m}")
    print(s["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print(f"saved {SERIES_FILE.name} and {WIDE_FILE.name}")


if __name__ == "__main__":
    main()
