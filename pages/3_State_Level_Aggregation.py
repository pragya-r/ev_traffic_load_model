import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="State Level Aggregation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
HOUR_COLS   = [f"hour_{h:02d}_mbps" for h in range(24)]
HOUR_LABELS = [
    "12AM","1AM","2AM","3AM","4AM","5AM",
    "6AM","7AM","8AM","9AM","10AM","11AM",
    "12PM","1PM","2PM","3PM","4PM","5PM",
    "6PM","7PM","8PM","9PM","10PM","11PM",
]
GRADIENT_STEPS = 12
BAR_COLOR      = "#A23B72"
PEAK_COLOR     = "#d62728"
GRID_ALPHA     = 0.3

# ─────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/ev_bandwidth_results.csv", dtype={"postcode": str})
    df.columns = df.columns.str.strip()

    # Suburb lookup
    try:
        lookup = pd.read_csv(
            "utils/postcode_suburb_lookup.csv", dtype={"POSTCODE": str}
        ).rename(columns={"POSTCODE": "postcode"})
        df = df.merge(lookup, on="postcode", how="left")
        df["SUBURBS_JOINED"] = df.get("SUBURBS_JOINED", pd.Series("", index=df.index)).fillna("(Unknown)")
    except FileNotFoundError:
        df["SUBURBS_JOINED"] = "(Suburb data unavailable)"

    # Derive state if not already present
    if "state" not in df.columns and "STATE" in df.columns:
        df.rename(columns={"STATE": "state"}, inplace=True)
    df["state"] = df.get("state", pd.Series("Unknown", index=df.index)).fillna("Unknown")

    # Centroids
    centroids = pd.read_csv(
        "utils/postcode_centroids.csv", dtype={"postcode": str}
    )[["postcode", "latitude", "longitude"]]
    df = df.merge(centroids, on="postcode", how="left")

    return df


@st.cache_data
def load_geodata():
    # Postcode polygons
    gdf = gpd.read_file("utils/postal_areas.geojson")
    for col in gdf.columns:
        if col.upper() in ("POA_CODE", "POSTCODE", "POA_CODE21"):
            gdf = gdf.rename(columns={col: "postcode"})
            break
    gdf["postcode"] = gdf["postcode"].astype(str)

    # State polygons
    gdf_states = gpd.read_file("utils/states.geojson")
    # STATE_NAME is the join key — normalise to "state" to match CSV
    gdf_states = gdf_states.rename(columns={"STATE_NAME": "state"})

    return gdf, gdf_states


@st.cache_data
def get_global_colour_ranges(df):
    # Postcode-level range
    postcode_ranges = {
        "total_upstream_gb": (
            df["total_upstream_gb"].min(),
            df["total_upstream_gb"].max(),
        ),
        "ev_count": (
            df["ev_count"].min(),
            df["ev_count"].max(),
        ),
    }

    # State-level aggregation first
    state_df = (
        df.groupby("state")
        .agg(
            ev_count=("ev_count", "sum"),
            total_upstream_gb=("total_upstream_gb", "sum"),
        )
        .reset_index()
    )

    state_ranges = {
        "total_upstream_gb": (
            state_df["total_upstream_gb"].min(),
            state_df["total_upstream_gb"].max(),
        ),
        "ev_count": (
            state_df["ev_count"].min(),
            state_df["ev_count"].max(),
        ),
    }

    return postcode_ranges, state_ranges


# ─────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────
def _gradient_colors():
    """Red-white gradient: intense red → white (low)."""
    gb = np.linspace(0, 240, GRADIENT_STEPS).astype(int)
    return [[255, int(g), int(g)] for g in gb]


GRADIENT = _gradient_colors()


def value_to_color(value, min_val, max_val, use_log=True, alpha=200):
    if value <= 0:
        return [255, 255, 255, 80]
    v  = np.log1p(value)  if use_log else value
    mn = np.log1p(min_val) if use_log else min_val
    mx = np.log1p(max_val) if use_log else max_val
    norm = np.clip((v - mn) / (mx - mn + 1e-9), 0, 1)
    step = int(norm * (GRADIENT_STEPS - 1))
    step = GRADIENT_STEPS - 1 - step   # flip: high value → deep red
    return GRADIENT[step] + [alpha]


# ─────────────────────────────────────────────────────────────
# MAP BUILDER
# ─────────────────────────────────────────────────────────────
def build_postcode_map(gdf_areas, df_year, metric, colour_range):
    agg = (
        df_year.groupby("postcode")
        .agg(
            ev_count          =("ev_count",           "sum"),
            total_upstream_gb =("total_upstream_gb",  "sum"),
            SUBURBS_JOINED    =("SUBURBS_JOINED",     "first"),
            state             =("state",              "first"),
        )
        .reset_index()
    )
    gdf = gdf_areas.merge(agg, on="postcode", how="left").fillna(
        {"ev_count": 0, "total_upstream_gb": 0,
         "SUBURBS_JOINED": "(Unknown)", "state": "(Unknown)"}
    )
    mn, mx = colour_range
    gdf["fill_color"]       = gdf[metric].apply(lambda v: value_to_color(v, mn, mx))
    gdf["line_color"]       = [[180, 180, 180, 160]] * len(gdf)
    gdf["EVs_Display"]      = gdf["ev_count"].apply(lambda x: f"{int(x):,}")
    gdf["Data_Display"]     = gdf["total_upstream_gb"].apply(lambda x: f"{x:,.2f} GB")
    return json.loads(gdf.to_json())


# Abbreviation → full name, matching states.geojson STATE_NAME values
STATE_ABBREV = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "WA":  "Western Australia",
    "SA":  "South Australia",
    "TAS": "Tasmania",
    "ACT": "Australian Capital Territory",
    "NT":  "Northern Territory",
}


def build_state_map(gdf_states, df_year, metric, colour_range):
    """GeoJSON choropleth aggregated at state level."""
    agg = (
        df_year.groupby("state")
        .agg(
            ev_count          =("ev_count",           "sum"),
            total_upstream_gb =("total_upstream_gb",  "sum"),
        )
        .reset_index()
    )
    # Map abbreviations to full names so they join with STATE_NAME in GeoJSON
    agg["state"] = agg["state"].map(STATE_ABBREV).fillna(agg["state"])
    gdf = gdf_states.merge(agg, on="state", how="left").fillna(
        {"ev_count": 0, "total_upstream_gb": 0}
    )
    mn, mx = colour_range
    gdf["fill_color"]   = gdf[metric].apply(lambda v: value_to_color(v, mn, mx))
    gdf["line_color"]   = [[120, 120, 120, 200]] * len(gdf)
    gdf["EVs_Display"]  = gdf["ev_count"].apply(lambda x: f"{int(x):,}")
    gdf["Data_Display"] = gdf["total_upstream_gb"].apply(lambda x: f"{x:,.2f} GB")
    return json.loads(gdf.to_json())


# ─────────────────────────────────────────────────────────────
# HOURLY CHART  (matches notebook style)
# ─────────────────────────────────────────────────────────────
def hourly_chart(hourly_mbps, title, ev_count, total_gb):
    peak_h    = int(np.argmax(hourly_mbps))
    peak_mbps = float(hourly_mbps[peak_h])

    colors = [PEAK_COLOR if h == peak_h else BAR_COLOR for h in range(24)]

    fig = go.Figure()
    # Use numeric x (0-23) so add_vline works; remap tick labels below
    fig.add_trace(go.Bar(
        x=list(range(24)),
        y=hourly_mbps,
        marker_color=colors,
        marker_line_color="black",
        marker_line_width=0.6,
        opacity=0.85,
        name="Bandwidth",
    ))
    # Peak vertical line — numeric x required by plotly
    fig.add_vline(
        x=peak_h,
        line_dash="dash",
        line_color="red",
        line_width=2.5,
        # annotation_text=f"Peak: {peak_mbps:,.0f} Mbps @ {HOUR_LABELS[peak_h]}",
        # annotation_position="top right",
        # annotation_font_color="red",
        # annotation_font_size=12,
    )
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b>  "
                f"<span style='font-size:13px; color:grey'>  "
                f"EVs: {ev_count:,} &nbsp;|&nbsp; "
                f"Total Upload: {total_gb:,.1f} GB &nbsp;|&nbsp; "
                f"Peak: {peak_mbps:,.0f} Mbps at {HOUR_LABELS[peak_h]}</span>"
            ),
            font_size=16,
        ),
        xaxis=dict(
            title="Hour of Day",
            tickangle=-45,
            showgrid=False,
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=HOUR_LABELS,
        ),
        yaxis=dict(title="Aggregate Bandwidth (Mbps)", gridcolor="lightgrey"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=420,
        margin=dict(t=80, b=60, l=60, r=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
try:
    df       = load_data()
    gdf_areas, gdf_states = load_geodata()
    postcode_ranges, state_ranges = get_global_colour_ranges(df)
    data_ok  = True
except FileNotFoundError as e:
    st.error(f"Missing data file: {e}\n\nPlace all required CSVs/GeoJSON in `utils`.")
    st.stop()

years = sorted(df["year"].unique())
scenario = sorted(df["scenario"].unique())

selected_year  = st.sidebar.selectbox(
    "Year",
    options=years,
    index=0,
)

selected_scenario  = st.sidebar.selectbox(
    "Scenario",
    options=scenario,
    index=0,
)

metric = st.sidebar.radio(
    "Map Metric",
    ["total_upstream_gb", "ev_count"],
    format_func=lambda x: "Daily Upload (GB)" if "gb" in x else "Number of EVs",
)

map_level = st.sidebar.radio(
    "Map Resolution",
    ["Postcode level", "State level"],
)

# ─────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────
df_filtered = df[df["scenario"] == selected_scenario].copy()

df_year = df_filtered[
    (df["year"] == selected_year)
    & (df["scenario"] == selected_scenario)
].copy()

# ─────────────────────────────────────────────────────────────
# HEADER METRICS
# ─────────────────────────────────────────────────────────────
st.title("EV Broadband Impact Dashboard", anchor=False)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total EVs",             f"{df_year['ev_count'].sum():,.0f}")
c2.metric("Daily Upload (TB)",     f"{df_year['total_upstream_gb'].sum() / 1000:.2f}")
c3.metric("Peak Bandwidth (Mbps)", f"{df_year[HOUR_COLS].sum().max():,.0f}")
c4.metric("Postcodes with EVs",    f"{len(df_year):,}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────
st.subheader(f"Geographic Distribution ({map_level})", anchor=False)
view_state = pdk.ViewState(latitude=-25.5, longitude=134.0, zoom=3, pitch=0)

tooltip_postcode = {
    "html": """
    <div style="font-family:Arial,sans-serif;padding:10px;background:rgba(255,255,255,0.95);
                border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.2);max-width:280px;">
        <b style="font-size:14px;">Postcode: {postcode}</b><br/>
        <span style="color:#666;font-size:12px;">{state} — {SUBURBS_JOINED}</span>
        <hr style="border:none;border-top:1px solid #ddd;margin:6px 0;">
        <b>EVs:</b> {EVs_Display}<br/>
        <b>Daily Upload:</b> {Data_Display}
    </div>""",
    "style": {"backgroundColor": "transparent", "border": "none"},
}

tooltip_state = {
    "html": """
    <div style="font-family:Arial,sans-serif;padding:10px;background:rgba(255,255,255,0.95);
                border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size:14px;">{state}</b><br/>
        <b>EVs:</b> {EVs_Display}<br/>
        <b>Daily Upload:</b> {Data_Display}
    </div>""",
    "style": {"backgroundColor": "transparent", "border": "none"},
}

def render_map(df_y, year_label=""):
    if map_level == "Postcode level":
        geo = build_postcode_map(gdf_areas, df_y, metric, postcode_ranges[metric])
        layer = pdk.Layer(
            "GeoJsonLayer", geo,
            opacity=0.85, stroked=True, filled=True,
            get_fill_color="properties.fill_color",
            get_line_color="properties.line_color",
            line_width_min_pixels=1,
            pickable=True, auto_highlight=True,
        )
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                        map_style="mapbox://styles/mapbox/light-v10",
                        tooltip=tooltip_postcode)
    else:
        geo = build_state_map(gdf_states, df_y, metric, state_ranges[metric])
        layer = pdk.Layer(
            "GeoJsonLayer", geo,
            opacity=0.85, stroked=True, filled=True,
            get_fill_color="properties.fill_color",
            get_line_color="properties.line_color",
            line_width_min_pixels=1,
            pickable=True, auto_highlight=True,
        )
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                        map_style="mapbox://styles/mapbox/light-v10",
                        tooltip=tooltip_state)
    return deck


st.pydeck_chart(render_map(df_year), use_container_width=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# HOURLY BANDWIDTH PROFILES
# ─────────────────────────────────────────────────────────────
st.subheader("Hourly Upload Bandwidth Profile", anchor=False)

tab_national, tab_state, tab_postcode = st.tabs(
    ["National", "By State", "By Postcode"]
)

with tab_national:
    nat = df_year
    hourly = nat[HOUR_COLS].sum().values
    fig = hourly_chart(
        hourly,
        f"National · {selected_year}",
        int(nat["ev_count"].sum()),
        float(nat["total_upstream_gb"].sum()),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_state:
    avail_states = sorted(df_year["state"].unique())
    chosen_state = st.selectbox("Choose state", avail_states, key="state_hourly")
    sub = df_year[df_year["state"] == chosen_state]
    hourly = sub[HOUR_COLS].sum().values
    fig = hourly_chart(
        hourly,
        f"{chosen_state} · {selected_year}",
        int(sub["ev_count"].sum()),
        float(sub["total_upstream_gb"].sum()),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_postcode:
    avail_pcs = sorted(df_year["postcode"].unique())
    chosen_pc = st.selectbox("Choose postcode", avail_pcs, key="pc_hourly")
    sub = df_year[df_year["postcode"] == chosen_pc]
    hourly = sub[HOUR_COLS].sum().values
    suburb_name = sub["SUBURBS_JOINED"].iloc[0] if len(sub) else ""
    fig = hourly_chart(
        hourly,
        f"{chosen_pc} ({suburb_name}) · {selected_year}",
        int(sub["ev_count"].sum()),
        float(sub["total_upstream_gb"].sum()),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────
st.subheader("Export", anchor=False)
csv = df_year.to_csv(index=False)
st.download_button(
    "Download filtered data as CSV",
    data=csv,
    file_name=f"ev_bandwidth_{selected_year}.csv",
    mime="text/csv",
)