"""media.csv and sales.csv into a complete daily panel.

Two things happen here that matter more on real data than on the benchmark:

1. Campaigns are aggregated to channel level. The synthetic export happens to
   carry one campaign per channel-day, so this is a no-op there -- but the real
   Sellforte export is campaign-grained, and a campaign ending is not a channel
   holdout. Spec section 11 names this as the top real-data risk.
2. Every series is reindexed onto a complete date grid, and a `present` mask
   records whether a row actually existed. Once reindexed, a missing row and a
   zero-spend row are indistinguishable -- and that is the difference between an
   eight-week dark period and eight weeks of broken ingestion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Panel:
    sid: str
    dates: pd.DatetimeIndex
    spend: pd.DataFrame        # index=dates, columns=MultiIndex(country, channel)
    present: pd.DataFrame      # same shape, bool
    impressions: pd.DataFrame
    clicks: pd.DataFrame
    sales: pd.DataFrame        # index=dates, columns=country_code

    @property
    def countries(self) -> list[str]:
        return sorted({c for c, _ in self.spend.columns})

    @property
    def channels(self) -> list[str]:
        return sorted({ch for _, ch in self.spend.columns})

    def channels_in(self, country: str) -> list[str]:
        """Channels this market actually ran -- i.e. spent on at least once."""
        out = []
        for c, ch in self.spend.columns:
            if c == country and (self.spend[(c, ch)] > 0).any():
                out.append(ch)
        return sorted(out)

    def series(self, country: str, channel: str) -> pd.Series:
        key = (country, channel)
        if key not in self.spend.columns:
            return pd.Series(0.0, index=self.dates, name=f"{country}/{channel}")
        return self.spend[key]

    def present_mask(self, country: str, channel: str) -> pd.Series:
        key = (country, channel)
        if key not in self.present.columns:
            return pd.Series(False, index=self.dates)
        return self.present[key]


def _pivot(df: pd.DataFrame, values: str, dates: pd.DatetimeIndex,
           fill: float) -> pd.DataFrame:
    wide = df.pivot_table(index="date", columns=["country_code",
                                                 "advertising_channel"],
                          values=values, aggfunc=lambda x: x.sum(min_count=1))
    return wide.reindex(dates).fillna(fill)


def build_panel(media_df: pd.DataFrame, sales_df: pd.DataFrame | None = None,
                sid: str = "") -> Panel:
    required = {"date", "country_code", "advertising_channel", "media_investment"}
    missing = required - set(media_df.columns)
    if missing:
        raise ValueError(f"Missing media columns: {', '.join(sorted(missing))}")
    media = media_df.copy()
    spend_values = pd.to_numeric(media["media_investment"], errors="coerce")
    if (~np.isfinite(spend_values) | (spend_values < 0)).any():
        raise ValueError("media_investment must contain finite, non-negative values; unknown spend is not zero")
    if media[["date", "country_code", "advertising_channel"]].isna().any().any():
        raise ValueError("Media dates, countries and channels must not be missing")
    media["media_investment"] = spend_values
    for column in ("impressions", "clicks"):
        if column not in media:
            media[column] = float("nan")
    media["date"] = pd.to_datetime(media["date"])

    dates = pd.date_range(media["date"].min(), media["date"].max(), freq="D")

    spend = _pivot(media, "media_investment", dates, 0.0)
    impressions = _pivot(media, "impressions", dates, float("nan"))
    clicks = _pivot(media, "clicks", dates, float("nan"))

    # Presence is counted BEFORE any fill, so a row that existed with spend 0.0
    # is distinguishable from a row that never existed at all.
    #
    # pivot_table(..., aggfunc="size") with a scalar `values=` silently drops
    # a level from the result columns -- UNCONDITIONALLY on the pandas version
    # pinned for this project, not only when a level has one distinct value, as
    # an earlier version of this comment claimed. The repro that found it
    # crashes outright on a multi-country panel. aggfunc="sum" always preserves
    # the full MultiIndex.
    # groupby(...).size().unstack(...) does not have that failure mode, so
    # presence is counted that way instead.
    present = (media.groupby(["date", "country_code", "advertising_channel"])
               .size()
               .unstack(["country_code", "advertising_channel"], fill_value=0)
               .reindex(dates, fill_value=0) > 0)
    present = present.reindex(columns=spend.columns, fill_value=False)

    if sales_df is not None and len(sales_df):
        sales = sales_df.copy()
        sales["date"] = pd.to_datetime(sales["date"])
        sales = (sales.pivot_table(index="date", columns="country_code",
                                   values="turnover", aggfunc=lambda x: x.sum(min_count=1))
                 .reindex(dates))
    else:
        sales = pd.DataFrame(index=dates)

    return Panel(sid=sid, dates=dates, spend=spend, present=present,
                 impressions=impressions, clicks=clicks, sales=sales)
