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
ROUGH = "Rough reading, not a causal estimate."
DEFINITIONS = ("Vs comparison is the sales change the comparison markets did not share; it "
               "counts as readable when it is at least twice the typical swing at other "
               "times of year. Rough return is extra sales per extra 1 spent, shown only for "
               "a readable change where spend moved by at least 10%.")
CARRYOVER_NOTE = ("Averages over the whole window understate a pause's effect, "
                  "because advertising keeps working for a while after it stops.")


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
    source = st.sidebar.radio("Data", ["Included sample", "Upload CSV files"])
    if source == "Included sample":
        return (*load_sample(), "Synthetic sample with known events")
    media_file = st.sidebar.file_uploader("Media CSV", type="csv")
    sales_file = st.sidebar.file_uploader("Sales CSV (optional)", type="csv")
    if media_file is None:
        st.info("Upload media data to begin. Media columns: date, country_code, "
                "advertising_channel, media_investment. Sales columns (optional): "
                "date, country_code, turnover.")
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
    return "Vs comparison" if r.comparison_markets else "Sales change"


def sales_column(r) -> str:
    if r is None or r.effect is None:
        return "No sales reading"
    text = f"{r.effect:+.1%}"
    if r.swing is None:
        text += " · too little history"
    elif not r.readable:
        text += " · unclear"
    if not r.comparison_markets:
        text += " · no comparison"
    return text


def baseline_help(r) -> str:
    text = "Average daily sales while no advertising ran."
    if r.readable and r.effect is not None and r.effect < 0:
        text += f" About {-r.effect:.0%} of sales went with advertising (rough)."
    return text


def caveats(e, r) -> list[str]:
    out = []
    if e.validity != "ok":
        out.append(f"Data quality flagged ({e.validity.replace('_', ' ')}): "
                   + "; ".join(e.validity_reasons))
    if e.evidence.get("censored_start"):
        out.append("Starts on the first day of the data, so it may have begun earlier.")
    if e.evidence.get("censored_end"):
        out.append("Runs to the last day of the data, so it may continue.")
    if "confounded" in e.tags:
        out.append("Another finding overlaps this one in the same market, "
                   "so their effects are hard to separate.")
    if r is None:
        out.append("Not enough data around this period for a sales reading.")
    elif r.sales_reference is not None and not r.comparison_markets:
        out.append("No comparison markets without their own findings in these weeks, "
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
            "Finding": TYPES[e.event_type].name, "Channel": e.channel or "All channels",
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
    # On screen the end date is left out so every column stays readable beside
    # the sidebar; Start stays a real date so it sorts by time. The download
    # keeps Start, End and Days.
    shown = frame.drop(columns=["End"])
    small = st.column_config.Column(width="small")
    st.dataframe(shown, hide_index=True, width="stretch", key=key,
                 height=35 * (min(len(shown), 10) + 1) + 3,
                 column_config={"Start": st.column_config.DateColumn(format="D MMM YYYY"),
                                **{c: small for c in ("Rank", "Market", "Days")}},
                 on_select=partial(pick_from_table, key), selection_mode="single-row")
    export = frame.assign(
        **{"Priority score": [e.informativeness for e in visible],
           "Evidence score": [e.detection_confidence for e in visible],
           "Sales change ratio": [getattr(reads[event_id(e)], "effect", None)
                                          for e in visible],
           "Rough return per 1 spent": [getattr(reads[event_id(e)], "rough_return", None)
                                        for e in visible]})
    st.download_button("Download findings", export.to_csv(index=False),
                       "findings.csv", "text/csv")


def kpi_tiles(e, r, panel) -> list[dict]:
    """label / value / description for the readout tiles; empty without a readout."""
    if r is None:
        return []

    def versus(now, before):
        change = f"{now / before - 1:+.0%} vs reference" if before else "reference was 0"
        return f"{change} ({compact(before)}/day)"

    tiles = [dict(label="Affected spend/day", value=compact(r.spend_during),
                  description=f"{subject_label(e, r)[:1].upper()}{subject_label(e, r)[1:]}: "
                              f"{versus(r.spend_during, r.spend_reference)}")]
    if r.sales_reference is not None:
        tiles.append(dict(label="Market sales/day", value=compact(r.sales_during),
                          description=versus(r.sales_during, r.sales_reference)))
        if r.swing is None:
            judged = "too little history to judge"
        else:
            judged = f"{'readable' if r.readable else 'unclear'} · typical swing ±{r.swing:.1%}"
        tiles.append(dict(label=comparison_label(r), value=f"{r.effect:+.1%}",
                          description=judged))
        tiles.append(dict(
            label="Rough return",
            value=f"{r.rough_return:,.2f}" if r.rough_return is not None else "—",
            description="extra sales per extra 1 spent" if r.rough_return is not None
            else "withheld for this finding — see the note below" if r.readable
            else "shown only for a readable change"))
    if e.event_type == "dark_period" and r.baseline is not None:
        tiles.append(dict(label="Baseline sales/day", value=compact(r.baseline),
                          description=baseline_help(r)))
    elif e.event_type == "step_change" and r.spend_reference > 0:
        tiles.append(dict(label="Spend multiple",
                          value=f"{r.spend_during / r.spend_reference:.1f}×",
                          description="spend during the step ÷ spend before it"))
    elif e.event_type == "single_channel" and e.channel:
        running = panel.series(e.country_code, e.channel)[r.during].mean()
        tiles.append(dict(label="Running channel/day", value=compact(running),
                          description=f"{e.channel}, the one channel still running"))
    return tiles


def kpis(e, r, panel) -> None:
    if r is None:
        ui.alert("No sales reading", "Not enough data around this period for a sales reading.",
                 key="no-readout")
        return
    tiles = kpi_tiles(e, r, panel)
    for col, tile in zip(st.columns(len(tiles)), tiles):
        with col:
            ui.metric_card(tile["label"], tile["value"], description=tile["description"],
                           key=f"kpi-{tile['label']}")

    st.caption(f"{ROUGH} {r.note}".strip())


def why_flagged(e) -> None:
    if not (e.evidence.get("confidence_sub_scores") or e.evidence.get("informativeness_drivers")):
        ui.alert("No scores from this detector",
                 "This detector does not report the scores behind its findings.",
                 key="no-scores")
        return
    st.caption("Rule-based scores from 0 to 1, not probabilities.")
    groups = (("Evidence that it happened", "confidence_sub_scores", "evidence"),
              ("How useful it is", "informativeness_drivers", "usefulness"))
    for col, (heading, field, key) in zip(st.columns(2), groups):
        with col, ui.elements(key=f"scores-{key}") as el:
            with el.card():
                with el.card_header():
                    el.heading(heading, level=4)
                with el.card_content():
                    with el.stack(gap="md"):
                        for name, meaning, value in score_rows(e.evidence.get(field, {})):
                            el.progress(round(value * 100), label=f"{name} · {value:.2f}")
                            el.text(meaning, variant="muted")






def method_note(r) -> str:
    """How the readout was computed, for the Details section."""
    side = {"before": "the same number of days just before it",
            "after": "the same number of days just after it (the data starts too late for a before window)",
            "on-days": "the pulse's own on-days between pauses"}[r.reference_side]
    peers = ", ".join(r.comparison_markets) or "none"
    return (f"Reference window: {side}. Comparison markets: {peers}. "
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
        st.markdown(f"**Detector's explanation.** "
                    f"{e.explanation or 'This detector gives no explanation.'}")


def main():
    st.set_page_config(page_title="Informative periods", layout="wide")
    st.title("Informative periods in marketing data")
    loaded = load_data()
    if loaded is None:
        return
    media, sales, label = loaded
    detectors = registry.available()
    detector = registry.default()
    if len(detectors) > 1:
        names = [d.name for d in detectors]
        chosen_name = st.sidebar.selectbox("Detector", names)
        detector = detectors[names.index(chosen_name)]
        label = f"{label} · Detector: {detector.name}"
    try:
        with st.spinner("Finding periods…"):
            events, panel, reads = analyze(media, sales, detector.key)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Please check your input data: {exc}")
        return
    dates = panel.dates
    st.caption(f"{label} · {len(panel.countries)} markets · {len(panel.channels)} channels · "
               f"{fmt_day(dates[0])} – {fmt_day(dates[-1])}")
    if not events:
        st.info("No informative periods were found in this data.")
        return

    order = ranked(events)
    rank = {event_id(e): i + 1 for i, e in enumerate(order)}
    markets = st.sidebar.multiselect("Markets", panel.countries, default=panel.countries)
    kinds = [k for k in TYPES
             if k in detector.event_types and any(e.event_type == k for e in events)]
    chosen = st.sidebar.multiselect("Pattern types", kinds, default=kinds,
                                    format_func=lambda k: TYPES[k].name)
    hide = st.sidebar.checkbox("Hide findings with data warnings")
    visible = [e for e in order if e.country_code in markets and e.event_type in chosen
               and not (hide and e.validity != "ok")]

    if not visible:
        st.info("No findings match these filters.")
        return
    ids = [event_id(e) for e in visible]
    if st.session_state.get("selected") not in ids:
        select(ids[0])
    selected = st.session_state["selected"]
    dark = is_dark()

    st.markdown("#### Where the informative periods are")
    st.caption("Each bar is one finding. Click a bar to inspect it.")
    st.plotly_chart(period_map(visible, sorted(markets), selected, (dates[0], dates[-1]), dark),
                    key="map", on_select=pick_from_map, selection_mode="points",
                    width="stretch", config={"displayModeBar": False})

    st.markdown("#### Ranked findings")
    st.caption("Most informative first. Click a row to inspect it.")
    table(visible, rank, reads)

    st.divider()
    e = next(x for x in visible if event_id(x) == selected)
    inspect(e, reads[selected], rank[selected], panel, dark)


if __name__ == "__main__":
    main()
