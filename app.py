"""Local interactive viewer: python -m streamlit run app.py

One page, read top to bottom: what was found (summary and period map), which
findings matter most (ranked cards and table), and what the selected finding
lets you measure (rough readout, charts, scores and caveats).
"""
import zlib
from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui

from detection.io.panel import build_panel
from detectors import registry
from ui.charts import carryover_chart, event_chart, period_map
from ui.labels import (TYPES, event_id, evidence_band, priority_band, ranked, score_rows,
                       title)
from ui.readout import readout

SAMPLE = Path(__file__).parent / "synthetic_data_generator/data"
ROUGH = "Estimates only. They don't prove what caused the change."
DEFINITIONS = ("'Vs other markets' is the sales change other markets didn't see. It counts as "
               "small when it is within twice the usual swing. 'Return' is extra sales per extra "
               "unit of spend, shown only when the change is not small and spend moved at least 10%.")
# The combined Team detector scores best on the dev split; fall back to the
# first registered detector when it is missing.
UI_DEFAULT_DETECTOR = "team"
CARRYOVER_NOTE = "Ads keep working for a while after they stop, so a pause's effect is understated."


@st.cache_data(show_spinner=False)
def load_sample():
    return pd.read_csv(SAMPLE / "media.csv"), pd.read_csv(SAMPLE / "sales.csv")


@st.cache_data(show_spinner=False)
def analyze(media, sales, detector_key: str | None = None):
    detector = registry.get(detector_key) if detector_key else registry.default()
    events = detector.detect(media, sales)
    panel = build_panel(media, sales, "viewer")
    return events, panel, {event_id(e): readout(e, panel, events) for e in events}


def compact(x: float) -> str:
    """1,284 / 12.9K / 4.2M -- short enough for a metric tile."""
    if abs(x) >= 1e9:
        return f"{x / 1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e4:
        return f"{x / 1e3:.1f}K"
    return f"{x:,.0f}"


def fmt_day(day) -> str:
    return f"{day.day} {day:%b %Y}"


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except AttributeError:
        return False


def load_data():
    source = st.sidebar.radio("Data", ["Sample data", "Upload CSV files"])
    if source == "Sample data":
        return (*load_sample(), "Sample data")
    media_file = st.sidebar.file_uploader("Media CSV", type="csv")
    sales_file = st.sidebar.file_uploader("Sales CSV (optional)", type="csv")
    if media_file is None:
        st.info("Upload a media CSV to start (date, country_code, advertising_channel, "
                "media_investment). A sales CSV is optional (date, country_code, turnover).")
        return None
    try:
        media = pd.read_csv(media_file)
        sales = pd.read_csv(sales_file) if sales_file is not None else None
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the CSV: {exc}")
        return None
    return media, sales, f"Uploaded: {media_file.name}"


def select(eid: str, from_table: bool = False) -> None:
    st.session_state["selected"] = eid
    if not from_table:
        # A keyed table keeps its row tick across reruns, so a selection made
        # anywhere else gets the table a new key and clears a stale tick.
        st.session_state["table_nonce"] = st.session_state.get("table_nonce", 0) + 1


def pick_from_map() -> None:
    points = st.session_state["map"].selection.points
    if points and points[0].get("customdata"):
        select(points[0]["customdata"][0])


def pick_from_table(key: str) -> None:
    rows = st.session_state[key].selection.rows
    if rows:
        select(st.session_state["table_ids"][rows[0]], from_table=True)


def subject_label(e, r) -> str:
    if e.event_type == "dark_period":
        return "all channels"
    if e.event_type == "single_channel":
        return "the other channels"
    return ", ".join(r.subject_channels) if r else str(e.channel)




def comparison_label(r) -> str:
    """Only say "comparison" when comparison markets were actually used."""
    return "Vs other markets" if r.comparison_markets else "Sales change"


SALES_CHANGE_HELP = (
    "How much this market's sales changed during the period, beyond what other markets saw. "
    "Small changes, within the market's usual week-to-week ups and downs, may be chance. "
    "Select the row for the full numbers.")


def sales_column(r) -> str:
    if r is None or r.effect is None:
        return "No sales data"
    return f"{r.effect:+.1%}"


def baseline_help(r) -> str:
    text = "Average daily sales with all ads off."
    if r.readable and r.effect is not None and r.effect < 0:
        text += f" About {-r.effect:.0%} of sales came with advertising (estimate)."
    return text


def caveats(e, r) -> list[str]:
    out = []
    if e.validity != "ok":
        out.append(f"Data quality ({e.validity.replace('_', ' ')}): "
                   + "; ".join(e.validity_reasons))
    if e.evidence.get("censored_start"):
        out.append("Starts on the first day of the data, so it may have started earlier.")
    if e.evidence.get("censored_end"):
        out.append("Runs to the end of the data, so it may still be going.")
    if "confounded" in e.tags:
        out.append("Overlaps another period in this market, "
                   "so their effects are hard to separate.")
    if r is None:
        out.append("Not enough data around this period to read sales.")
    elif r.sales_reference is not None and not r.comparison_markets:
        out.append("No other market was quiet in these weeks, "
                   "so the sales change also includes any seasonality or trend.")
    verification = e.evidence.get("launch_verification")
    if isinstance(verification, dict) and verification.get("status"):
        out.append(f"Launch analysis readiness: {str(verification['status']).replace('_', ' ')}.")
    return out












def table(visible, rank, reads) -> None:
    rows = []
    for e in visible:
        r = reads[event_id(e)]
        rows.append({
            "Rank": rank[event_id(e)], "Market": e.country_code,
            "Pattern": TYPES[e.event_type].name, "Channel": e.channel or "All channels",
            "Start": e.start.date(), "End": e.end.date(), "Days": e.n_days,
            "Priority": priority_band(e.informativeness),
            "Evidence": evidence_band(e.detection_confidence),
            "Sales change": sales_column(r),
            "Data quality": "OK" if e.validity == "ok" else e.validity.replace("_", " "),
        })
    frame = pd.DataFrame(rows)
    ids = [event_id(e) for e in visible]
    st.session_state["table_ids"] = ids
    # The rows shown are part of the key too: a filter change must not leave a
    # tick on whatever finding now sits at that row number.
    key = f"table-{st.session_state.get('table_nonce', 0)}-{zlib.crc32('|'.join(ids).encode())}"
    # On screen only what is needed to pick a period: no end date, evidence or
    # data-quality column (warnings show in the panel). Start stays a real date
    # so it sorts by time. The download keeps every column.
    shown = frame.drop(columns=["End", "Evidence", "Data quality"])
    small = st.column_config.Column(width="small")
    st.dataframe(shown, hide_index=True, width="stretch", key=key,
                 height=35 * (min(len(shown), 10) + 1) + 3,
                 column_config={"Start": st.column_config.DateColumn(format="D MMM YYYY"),
                                "Sales change": st.column_config.TextColumn(help=SALES_CHANGE_HELP),
                                **{c: small for c in ("Rank", "Market", "Days")}},
                 on_select=partial(pick_from_table, key), selection_mode="single-row")
    export = frame.assign(
        **{"Priority score": [e.informativeness for e in visible],
           "Evidence score": [e.detection_confidence for e in visible],
           "Sales change ratio": [getattr(reads[event_id(e)], "effect", None)
                                          for e in visible],
           "Rough return per 1 spent": [getattr(reads[event_id(e)], "rough_return", None)
                                        for e in visible]})
    st.download_button("Download CSV", export.to_csv(index=False),
                       "findings.csv", "text/csv")


def kpi_tiles(e, r, panel) -> list[dict]:
    """label / value / description for the readout tiles; empty without a readout."""
    if r is None:
        return []

    def versus(now, before):
        change = f"{now / before - 1:+.0%} vs reference" if before else "reference was 0"
        return f"{change} ({compact(before)}/day)"

    tiles = [dict(label="Spend/day", value=compact(r.spend_during),
                  description=f"{subject_label(e, r)[:1].upper()}{subject_label(e, r)[1:]}: "
                              f"{versus(r.spend_during, r.spend_reference)}")]
    if r.sales_reference is not None:
        tiles.append(dict(label="Sales/day", value=compact(r.sales_during),
                          description=versus(r.sales_during, r.sales_reference)))
        if r.swing is None:
            judged = "not enough history to judge"
        else:
            judged = f"usual swing ±{r.swing:.1%}"
        tiles.append(dict(label=comparison_label(r), value=f"{r.effect:+.1%}",
                          description=judged))
        tiles.append(dict(
            label="Return",
            value=f"{r.rough_return:,.2f}" if r.rough_return is not None else "—",
            description="extra sales per 1 spent" if r.rough_return is not None
            else "not shown, see the note below" if r.readable
            else "not shown for a small change"))
    if e.event_type == "dark_period" and r.baseline is not None:
        tiles.append(dict(label="Sales with ads off", value=compact(r.baseline),
                          description=baseline_help(r)))
    elif e.event_type == "step_change" and r.spend_reference > 0:
        tiles.append(dict(label="Spend change",
                          value=f"{r.spend_during / r.spend_reference:.1f}×",
                          description="during vs before"))
    elif e.event_type == "single_channel" and e.channel:
        running = panel.series(e.country_code, e.channel)[r.during].mean()
        tiles.append(dict(label="Channel still on/day", value=compact(running),
                          description=e.channel))
    return tiles


def kpis(e, r, panel) -> None:
    if r is None:
        ui.alert("No sales data", "Not enough data around this period to read sales.",
                 key="no-readout")
        return
    tiles = kpi_tiles(e, r, panel)
    for col, tile in zip(st.columns(len(tiles)), tiles):
        with col:
            ui.metric_card(tile["label"], tile["value"], description=tile["description"],
                           key=f"kpi-{tile['label']}")

    st.caption(f"{ROUGH} {r.note}".strip())


WEAK_SCORE = 0.5
SCORE_GROUPS = (("How sure", "confidence_sub_scores"),
                ("How useful", "informativeness_drivers"))


def score_checks(e) -> list[tuple[str, list[tuple[str, bool]]]]:
    """(group, [(criterion name, met?)]) for the groups the detector reported."""
    return [(heading, [(name, value >= WEAK_SCORE)
                       for name, _, value in score_rows(e.evidence[field])])
            for heading, field in SCORE_GROUPS if e.evidence.get(field)]


def score_summary(e) -> list[str]:
    """One line per score group: the overall band, and only the criteria that
    fall short."""
    bands = {"How sure": evidence_band(e.detection_confidence),
             "How useful": priority_band(e.informativeness)}
    lines = []
    for heading, checks in score_checks(e):
        weak = [name.lower() for name, ok in checks if not ok]
        tail = f"Falls short on: {', '.join(weak)}." if weak else "Nothing falls short."
        lines.append(f"**{heading}: {bands[heading]}.** {tail}")
    return lines


def why_flagged(e) -> None:
    lines = score_summary(e)
    if not lines:
        ui.alert("No scores from this detector",
                 "This detector doesn't report scores.",
                 key="no-scores")
        return
    for line, (heading, checks) in zip(lines, score_checks(e)):
        text, button = st.columns([6, 1], vertical_alignment="center")
        text.markdown(line)
        # A popover, not an expander: this already sits inside Details, and
        # Streamlit does not nest expanders.
        with button.popover("Details", width="stretch"):
            for name, ok in checks:
                st.markdown(f"{'✓' if ok else '✗'} {name}")


def method_note(r) -> str:
    """How the readout was computed, for the Details section."""
    side = {"before": "the same number of days just before it",
            "after": "the same number of days just after it (the data starts too late for a before window)",
            "on-days": "the pulse's own on-days between pauses"}[r.reference_side]
    peers = ", ".join(r.comparison_markets) or "none"
    return (f"Compared with {side}. Other markets used: {peers}. "
            f"{DEFINITIONS} {CARRYOVER_NOTE}")


def inspect(e, r, rank, panel, dark) -> None:
    info = TYPES[e.event_type]
    found = caveats(e, r)
    st.subheader(f"#{rank} · {title(e)}")
    st.caption(f"{fmt_day(e.start)} – {fmt_day(e.end)} · {e.n_days} days · "
               f"Priority {priority_band(e.informativeness)} · "
               f"Evidence {evidence_band(e.detection_confidence)} · {info.measures}")
    if e.validity != "ok":
        # A data-quality flag can undo every number below, so it is never tucked away.
        ui.alert("Data quality", next(line for line in found if line.startswith("Data quality")),
                 variant="destructive", key="data-quality")
    kpis(e, r, panel)
    st.plotly_chart(event_chart(e, panel, r, dark), key="detail", width="stretch")
    after = carryover_chart(e, panel, r, dark)
    if after is not None:
        st.markdown("**Sales around each pause, averaged over the pauses**")
        st.plotly_chart(after, key="carryover", width="stretch")
    for line in found:
        if not line.startswith("Data quality"):
            st.caption(f"⚠ {line}")
    with st.expander("Details"):
        why_flagged(e)
        if r is not None:
            st.caption(method_note(r))
        st.markdown(f"**Detector notes.** "
                    f"{e.explanation or 'This detector gives no explanation.'}")


def main():
    st.set_page_config(page_title="Informative periods", layout="wide")
    st.title("Informative periods")
    loaded = load_data()
    if loaded is None:
        return
    media, sales, label = loaded
    detectors = registry.available()
    detector = registry.default()
    if len(detectors) > 1:
        keys = [d.key for d in detectors]
        default = UI_DEFAULT_DETECTOR if UI_DEFAULT_DETECTOR in keys else keys[0]
        chosen = st.sidebar.selectbox("Detector", keys, index=keys.index(default),
                                      format_func=lambda k: registry.get(k).name,
                                      key="detector")
        detector = registry.get(chosen)
        label = f"{label} · Detector: {detector.name}"
    try:
        with st.spinner("Scanning…"):
            events, panel, reads = analyze(media, sales, detector.key)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Couldn't read the data: {exc}")
        return
    dates = panel.dates
    st.caption(f"{label} · {len(panel.countries)} markets · {len(panel.channels)} channels · "
               f"{fmt_day(dates[0])} – {fmt_day(dates[-1])}")
    if not events:
        st.info("No periods found.")
        return

    order = ranked(events)
    rank = {event_id(e): i + 1 for i, e in enumerate(order)}
    markets = st.sidebar.multiselect("Markets", panel.countries, default=panel.countries)
    kinds = [k for k in TYPES
             if k in detector.event_types and any(e.event_type == k for e in events)]
    chosen = st.sidebar.multiselect("Pattern types", kinds, default=kinds,
                                    format_func=lambda k: TYPES[k].name)
    hide = st.sidebar.checkbox("Hide periods with data warnings")
    visible = [e for e in order if e.country_code in markets and e.event_type in chosen
               and not (hide and e.validity != "ok")]

    if not visible:
        st.info("Nothing matches these filters.")
        return
    ids = [event_id(e) for e in visible]
    if st.session_state.get("selected") not in ids:
        select(ids[0])
    selected = st.session_state["selected"]
    dark = is_dark()

    st.markdown("#### Timeline")
    st.caption("One bar per period. Click a bar to see it below.")
    st.plotly_chart(period_map(visible, sorted(markets), selected, (dates[0], dates[-1]), dark),
                    key="map", on_select=pick_from_map, selection_mode="points",
                    width="stretch", config={"displayModeBar": False})

    st.markdown("#### Periods")
    st.caption("Most useful first. Click a row to see it below.")
    table(visible, rank, reads)

    st.divider()
    e = next(x for x in visible if event_id(x) == selected)
    inspect(e, reads[selected], rank[selected], panel, dark)


if __name__ == "__main__":
    main()
