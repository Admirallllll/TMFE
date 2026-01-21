from __future__ import annotations

import pandas as pd


LAG_COLUMNS: tuple[str, ...] = (
    "peforw_qavg",
    "peforw_eoq",
    "eps12mtrailing_qavg",
    "eps12mtrailing_eoq",
    "eps12mfwd_qavg",
    "eps12mfwd_eoq",
    "eps_lt",
)


def build_panel_with_targets(df: pd.DataFrame, *, logger) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ticker", "quarter_index"]).reset_index(drop=True)

    g = df.groupby("ticker", sort=False)

    for c in LAG_COLUMNS:
        df[f"lag_{c}"] = g[c].shift(1)

    df["lead_peforw_qavg"] = g["peforw_qavg"].shift(-1)
    df["lead_peforw_eoq"] = g["peforw_eoq"].shift(-1)

    df["delta_peforw_qavg"] = df["lead_peforw_qavg"] - df["peforw_qavg"]
    df["delta_peforw_eoq"] = df["lead_peforw_eoq"] - df["peforw_eoq"]

    before = len(df)
    df = df.loc[df["delta_peforw_qavg"].notna()].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped:,} last-observation rows with missing lead values")

    q75 = (
        df.groupby(["sector", "datacqtr"], sort=False)["delta_peforw_qavg"]
        .transform(lambda x: x.quantile(0.75))
        .astype(float)
    )
    df["valuation_upgrade"] = (df["delta_peforw_qavg"] >= q75).astype("int8")

    return df

