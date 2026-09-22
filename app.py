"""Local interactive viewer: python -m streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from detection.io.panel import build_panel
from detection.pipeline import run_detection

SAMPLE = Path(__file__).parent / "synthetic_data_generator/data"


@st.cache_data
def analyze(media, sales):
    return run_detection(media, sales, "viewer")


def main():
    st.set_page_config(page_title="Marketing periods", layout="wide")
    st.title("Find informative marketing periods")
    st.write("Inspect changes in advertising activity and identify periods worth investigating.")
    source = st.sidebar.radio("Data", ["Included sample", "Upload CSV files"])
    if source == "Included sample":
        media = pd.read_csv(SAMPLE / "media.csv")
        sales = pd.read_csv(SAMPLE / "sales.csv")
        st.caption("Synthetic demonstration data with known events.")
    else:
        media_file = st.sidebar.file_uploader("Media CSV", type="csv")
        sales_file = st.sidebar.file_uploader("Sales CSV (optional)", type="csv")
        if media_file is None:
            st.info("Upload media data to begin. Required columns: date, country_code, advertising_channel, media_investment.")
            return
        try:
            media = pd.read_csv(media_file)
            sales = pd.read_csv(sales_file) if sales_file is not None else None
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f"Could not read the CSV: {exc}")
            return

    try:
        with st.spinner("Finding periods…"):
            events = analyze(media, sales)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Please check your input data: {exc}")
        return
    if not events:
        st.info("No periods found with the current detection rules.")
        return

    countries = sorted({e.country_code for e in events})
    selected_countries = st.sidebar.multiselect("Markets", countries, default=countries)
    kinds = sorted({e.event_type for e in events})
    selected_kinds = st.sidebar.multiselect("Event types", kinds, default=kinds)
    ordered = sorted((e for e in events if e.country_code in selected_countries
                      and e.event_type in selected_kinds), key=lambda e: -e.informativeness)
    st.caption("Scores are heuristics, not probabilities or estimates of marketing impact. Control labels identify comparison candidates, not validated causal controls.")
    if not ordered:
        st.info("No periods match these filters.")
        return

    table = pd.DataFrame([{
        "Market": e.country_code, "Channel": e.channel or "All channels",
        "Type": e.event_type, "Start": e.start.date(), "End observed": e.end.date(),
        "Usefulness score": round(e.informativeness, 3),
        "Evidence score": round(e.detection_confidence, 3), "Data quality": e.validity,
    } for e in ordered])
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Download findings", table.to_csv(index=False), "findings.csv", "text/csv")
    index = st.selectbox("Inspect a period", range(len(ordered)), format_func=lambda i:
                         f"{ordered[i].country_code} · {ordered[i].channel or 'All channels'} · "
                         f"{ordered[i].event_type} · {ordered[i].start.date()}")
    event = ordered[index]
    st.write(event.explanation)
    if event.validity != "ok":
        st.warning("Check the data-quality caveats before using this period for analysis.")
    panel = build_panel(media, sales)
    spend = panel.spend[event.country_code].copy()
    # Unknown rows are gaps in the chart, even though detection uses a filled grid.
    spend = spend.where(panel.present[event.country_code])
    spend.index.name = "Date"
    chart_data = spend.reset_index().melt(id_vars="Date", var_name="Channel", value_name="Spend")
    fig = px.line(chart_data, x="Date", y="Spend", color="Channel", title=f"Daily spend — {event.country_code}")
    windows = event.components if event.event_type == "channel_pulse" else ((event.start, event.end),)
    for start, end in windows:
        fig.add_vrect(x0=start, x1=end + pd.Timedelta(days=1), fillcolor="orange", opacity=0.2, line_width=0)
    st.plotly_chart(fig, width="stretch")
    if event.country_code in panel.sales:
        st.line_chart(panel.sales[[event.country_code]].rename(columns={event.country_code: "Sales"}))
    with st.expander("Score details"):
        st.json(event.evidence)


if __name__ == "__main__":
    main()
