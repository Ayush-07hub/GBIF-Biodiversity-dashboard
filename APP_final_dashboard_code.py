"""Run using: " streamlit run APP_final_dashboard_code.py " in the terminal"""

# APP_final_dashboard_code.py - GBIF Biodiversity Dashboard
# ------------------------------------------------------------
# Biodiversity Dashboard on Cleaned GBIF Data
# - Assumes a cleaned CSV produced by your DataCleaning & EDA notebook.
# - Includes:
#   * Filters by kingdom, countryCode, country, year, species search
#   * Overview + filter summary
#   * Taxonomy distributions (including Animalia focus)
#   * Temporal trends (year/month, animated map)
#   * Spatial maps: density map + scatter map
#   * Correlation heatmap (only meaningful numeric features)
#   * Diversity and richness
#   * People / provenance (identifiedBy, rightsHolder, recordedBy)


import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="GBIF Biodiversity Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================
# DASHBOARD TITLE
# ===============================================================

st.markdown(
    """
    <h1 style="text-align:center; font-size:38px; margin-bottom:10px;">
        🌍 GBIF Biodiversity Insights Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

# ===============================================================
# Load Cleaned Dataset
# ===============================================================

@st.cache_data
def load_dataset():
    preferred = "APP_final_dashboard_dataset_8_clean.csv"
    fallback = ["APP_final_dataset_8_clean.csv"]

    path = None
    if os.path.exists(preferred):
        path = preferred
    else:
        for f in fallback:
            if os.path.exists(f):
                path = f
                break

    if not path:
        st.error("Cleaned dataset not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Parse date
    if "eventDate" in df.columns:
        df["eventDate"] = pd.to_datetime(df["eventDate"], errors="coerce")

    # Clean numeric columns
    numeric_cols = [
        "year", "month", "day",
        "decimalLatitude", "decimalLongitude",
        "elevation", "depth", "individualCount"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean text
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Coordinate cleaning
    if {"decimalLatitude", "decimalLongitude"}.issubset(df.columns):
        df = df[
            df["decimalLatitude"].between(-90, 90)
            & df["decimalLongitude"].between(-180, 180)
        ]

    # Sort for temporal visualization
    if "year" in df.columns:
        df = df.sort_values("year")

    return df


df = load_dataset()
if df.empty:
    st.stop()

# ===============================================================
# Country Name + Code Mapping (Option C)
# ===============================================================

country_mapping = {}
if {"Country", "countryCode"}.issubset(df.columns):
    for _, row in df[["Country", "countryCode"]].dropna().drop_duplicates().iterrows():
        cname = str(row["Country"])
        ccode = str(row["countryCode"])
        country_mapping[cname] = ccode


def format_country_label(cname):
    if cname in country_mapping:
        return f"{cname} ({country_mapping[cname]})"
    return cname


def clean_country_label(label):
    if "(" in label:
        return label.split("(")[0].strip()
    return label


# ===============================================================
# Sidebar Filters
# ===============================================================

st.sidebar.title("Filters")
df_filtered = df.copy()

# Kingdom filter
if "kingdom" in df.columns:
    kingdoms = sorted(df["kingdom"].dropna().unique())
    selected_k = st.sidebar.multiselect("Kingdom", kingdoms, default=kingdoms)
    df_filtered = df_filtered[df_filtered["kingdom"].isin(selected_k)]

# Country filter (NAME only, showing code)
if "Country" in df.columns:
    country_list = sorted(df["Country"].dropna().unique())
    display_list = [format_country_label(c) for c in country_list]

    selected_display = st.sidebar.multiselect("Country", display_list)

    if selected_display:
        selected_countries = [clean_country_label(x) for x in selected_display]
        df_filtered = df_filtered[df_filtered["Country"].isin(selected_countries)]

# Year Range
if "year" in df.columns:
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    yrs = st.sidebar.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))
    df_filtered = df_filtered[df_filtered["year"].between(yrs[0], yrs[1])]

# Month filter
if "month" in df.columns:
    months = sorted(df["month"].dropna().unique())
    sm = st.sidebar.multiselect("Month", months)
    if sm:
        df_filtered = df_filtered[df_filtered["month"].isin(sm)]

# Species search
search = st.sidebar.text_input("Search species/scientific name")
if search:
    df_filtered = df_filtered[
        df_filtered["scientificName"].str.contains(search, case=False, na=False)
        | df_filtered["species"].str.contains(search, case=False, na=False)
    ]

# Dynamic filters
st.sidebar.markdown("### Dynamic Filters")
dyn_fields = [
    c for c in ["phylum","class","order","family","genus","species","stateProvince"]
    if c in df.columns
]

for i in range(1, 4):
    with st.sidebar.expander(f"Dynamic Filter {i}", expanded=False):
        attr = st.selectbox(f"Select attribute {i}", ["None"] + dyn_fields, key=f"A{i}")
        if attr != "None":
            vals = sorted(df[attr].dropna().unique())
            sel = st.multiselect(f"Values ({attr})", vals, key=f"V{i}")
            if sel:
                df_filtered = df_filtered[df_filtered[attr].isin(sel)]

# ===============================================================
# Filter Summary
# ===============================================================

def show_filter_summary():
    rec = len(df_filtered)
    kings = df_filtered["kingdom"].nunique() if "kingdom" in df else 0
    cn = df_filtered["Country"].nunique() if "Country" in df else 0
    yrs = df_filtered["year"].dropna()
    y1 = int(yrs.min()) if not yrs.empty else "-"
    y2 = int(yrs.max()) if not yrs.empty else "-"

    st.markdown(f"""
    ### Filter Summary  
    **Records:** {rec}  
    **Kingdoms:** {kings}  
    **Countries:** {cn}  
    **Years:** {y1} – {y2}  
    """)

# ===============================================================
# Tabs
# ===============================================================

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 Overview",
    "📊 EDA Overview",
    "🧬 Taxonomy",
    "⏱ Temporal trends",
    "🗺 Spatial maps",
    "🧮 Diversity & relationships",
    "👥 People"
])

# ===============================================================
# TAB 0 — OVERVIEW
# ===============================================================

with tab0:
    st.header("Cleaned Dataset Overview")

    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("Columns")
    st.write(list(df.columns))

    st.subheader("Missing Values (%)")
    mv = (df.isna().mean() * 100).round(1)
    st.dataframe(mv.to_frame("Missing %"), use_container_width=True)

    st.subheader("View Cleaned Dataset")

    mode = st.selectbox("Select view:", ["First 200 rows", "Last 200 rows", "Full dataset"])

    if mode == "First 200 rows":
        st.dataframe(df.head(200), use_container_width=True)

    elif mode == "Last 200 rows":
        st.dataframe(df.tail(200), use_container_width=True)

    else:
        st.warning("Rendering full dataset may be slow.")
        if st.button("Load Full Dataset"):
            st.dataframe(df, use_container_width=True)

# ===============================================================
# TAB 1 — EDA OVERVIEW
# ===============================================================

with tab1:
    st.header("EDA Overview")
    show_filter_summary()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(df_filtered))
    c2.metric("Species", df_filtered["species"].nunique())
    c3.metric("Genera", df_filtered["genus"].nunique())
    c4.metric("Countries", df_filtered["Country"].nunique())

    st.subheader("Missing Values (%)")
    mvf = (df_filtered.isna().mean() * 100).round(1).sort_values(ascending=False)
    fig = px.bar(mvf)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr_cols = [
        c for c in [
            "year","month",
            "decimalLatitude","decimalLongitude",
            "elevation","depth","individualCount"
        ]
        if c in df.columns
    ]
    corr_df = df_filtered[corr_cols]
    fig_corr = px.imshow(
        corr_df.corr(),
        text_auto=True,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ===============================================================
# TAB 2 — TAXONOMY
# ===============================================================

with tab2:
    st.header("Taxonomy")

    st.subheader("Top Phyla")
    if "phylum" in df_filtered:
        ph = df_filtered["phylum"].fillna("Unknown").value_counts().head(20)
        st.plotly_chart(px.bar(ph), use_container_width=True)

    st.subheader("Top Species (Overall)")
    if "species" in df_filtered:
        sp = df_filtered["species"].fillna("Unknown").value_counts().head(20)
        st.plotly_chart(px.bar(sp), use_container_width=True)

    if {"kingdom","species"}.issubset(df_filtered):
        st.subheader("Top Species by Kingdom")
        for k in sorted(df_filtered["kingdom"].dropna().unique()):
            st.write(f"**{k}**")
            tk = (
                df_filtered[df_filtered["kingdom"] == k]["species"]
                .fillna("Unknown")
                .value_counts()
                .head(10)
            )
            st.plotly_chart(px.bar(tk), use_container_width=True)

    tax_cols = ["kingdom","phylum","class","order","family","genus","species"]
    tax_cols = [c for c in tax_cols if c in df]
    if len(tax_cols) >= 2:
        tx = df_filtered[tax_cols].fillna("Unknown")
        txg = tx.groupby(tax_cols).size().reset_index(name="count")
        st.subheader("Taxonomic Treemap")
        st.plotly_chart(px.treemap(txg, path=tax_cols, values="count"),
                        use_container_width=True)

# ===============================================================
# TAB 3 — TEMPORAL TRENDS
# ===============================================================

with tab3:
    st.header("Temporal Trends")

    if "year" in df_filtered:
        y = df_filtered.groupby("year").size().reset_index(name="count")
        st.subheader("Yearly Observation Trend")
        st.plotly_chart(px.line(y, x="year", y="count", markers=True),
                        use_container_width=True)

    if {"year","month"}.issubset(df_filtered):
        dfm = df_filtered.dropna(subset=["year","month"])
        dfm["ym"] = pd.to_datetime(
            dfm["year"].astype(str)+"-"+dfm["month"].astype(str)+"-01",
            errors="coerce"
        )
        m = dfm.groupby("ym").size().reset_index(name="count")
        st.subheader("Monthly Observation Trend")
        st.plotly_chart(px.line(m, x="ym", y="count"), use_container_width=True)

    # Strict ascending animation
    st.subheader("Animated Spatial Timeline (1960–2023)")

    if {"decimalLatitude","decimalLongitude","year"}.issubset(df):
        anim = (
            df_filtered
            .query("1960 <= year <= 2023")
            .dropna(subset=["decimalLatitude","decimalLongitude"])
            .sort_values("year")
        )

        if len(anim) > 6000:
            anim = anim.sample(6000, random_state=42)

        fig_anim = px.scatter_geo(
            anim,
            lat="decimalLatitude",
            lon="decimalLongitude",
            animation_frame="year",
            color="kingdom",
            opacity=0.7,
        )
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] = True
        st.plotly_chart(fig_anim, use_container_width=True)

# ===============================================================
# TAB 4 — SPATIAL MAPS (UPDATED: Readable Density + Upgrades)
# ===============================================================

with tab4:
    st.header("Spatial Maps")

    df_geo = df_filtered.dropna(subset=["decimalLatitude","decimalLongitude"])

    # -------------------------
    # Readable density heatmap with country labels, opacity slider, and region zoom
    # -------------------------
    st.subheader("Density Map")

    if not df_geo.empty:

        # --- Controls: region select, opacity slider, labels toggle
        controls_col1, controls_col2, controls_col3 = st.columns([1, 1, 1])
        with controls_col1:
            region = st.selectbox(
                "Region",
                options=[
                    "World",
                    "Asia",
                    "Africa",
                    "Europe",
                    "North America",
                    "South America",
                    "Oceania"
                ],
                index=0
            )
        with controls_col2:
            opacity = st.slider("Density opacity", min_value=0.15, max_value=1.0, value=0.85, step=0.05)
        with controls_col3:
            show_labels = st.checkbox("Show country labels", value=True)

        # Define axis ranges for regions (lon_min, lon_max, lat_min, lat_max)
        regions = {
            "World": (-180, 180, -90, 90),
            "Asia": (25, 170, -15, 80),
            "Africa": (-20, 55, -35, 37),
            "Europe": (-25, 40, 35, 72),
            "North America": (-170, -30, 5, 72),
            "South America": (-90, -30, -60, 15),
            "Oceania": (110, 180, -50, 10)
        }

        lon_min, lon_max, lat_min, lat_max = regions.get(region, regions["World"])

        # Ensure we have some padding if data range smaller than region
        pad_lon = (lon_max - lon_min) * 0.03
        pad_lat = (lat_max - lat_min) * 0.03
        lon_min, lon_max = lon_min + pad_lon, lon_max - pad_lon
        lat_min, lat_max = lat_min + pad_lat, lat_max - pad_lat

        # Create density heatmap (binned)
        # Using nbins high enough for detail, but safe for Streamlit Cloud
        fig_dm = px.density_heatmap(
            df_geo,
            x="decimalLongitude",
            y="decimalLatitude",
            nbinsx=180,
            nbinsy=120,
            color_continuous_scale="Viridis",
            labels={"decimalLongitude": "Longitude", "decimalLatitude": "Latitude"},
            height=700
        )

        # Apply opacity control and hover template
        fig_dm.update_traces(
            opacity=opacity,
            hovertemplate="Lat: %{y:.3f}<br>Lon: %{x:.3f}<br>Density: %{z}<extra></extra>"
        )

        # Natural aspect + clean layout
        fig_dm.update_layout(
            title="Global Species Observation Density",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            yaxis=dict(scaleanchor="x", scaleratio=0.75),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_ticks="",
            yaxis_ticks="",
            plot_bgcolor="#f0f0f0",
            paper_bgcolor="white",
            coloraxis_colorbar=dict(title="Density", thickness=16, len=0.75),
            margin=dict(l=10, r=10, t=50, b=10)
        )

        # Set axis zoom to selected region
        fig_dm.update_xaxes(range=[lon_min, lon_max])
        fig_dm.update_yaxes(range=[lat_min, lat_max])

        # Add readable country labels at data-driven centroids (if requested)
        if show_labels and "Country" in df_geo.columns:
            centroids = (
                df_geo.groupby("Country")[["decimalLongitude", "decimalLatitude"]]
                .mean()
                .reset_index()
            )

            # Attempt to avoid overlapping labels by filtering out extremely close duplicates
            # (simple approach: keep top centroid per rounded lon/lat cell)
            centroids["lon_round"] = centroids["decimalLongitude"].round(1)
            centroids["lat_round"] = centroids["decimalLatitude"].round(1)
            centroids = centroids.drop_duplicates(subset=["lon_round", "lat_round"])

            fig_dm.add_scatter(
                x=centroids["decimalLongitude"],
                y=centroids["decimalLatitude"],
                mode="text",
                text=centroids["Country"],
                textfont=dict(size=10, color="black"),
                hoverinfo="skip",
                showlegend=False
            )

        # Add subtle scatter overlay to show point clusters (sampled for performance)
        sample_n = min(3000, len(df_geo))
        sampled = df_geo.sample(sample_n, random_state=42)
        fig_dm.add_scatter(
            x=sampled["decimalLongitude"],
            y=sampled["decimalLatitude"],
            mode="markers",
            marker=dict(size=1.5, color="white", opacity=0.35),
            showlegend=False,
            hoverinfo="skip"
        )

        st.plotly_chart(fig_dm, use_container_width=True)

    # Scatter map (unchanged)
    st.subheader("Scatter Map")

    sc = df_geo.copy()
    if len(sc) > 8000:
        sc = sc.sample(8000)

    fig_sc = px.scatter_geo(
        sc,
        lat="decimalLatitude",
        lon="decimalLongitude",
        color="kingdom",
        hover_name="scientificName",
        opacity=0.65,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ===============================================================
# TAB 5 — DIVERSITY & RELATIONSHIPS
# ===============================================================

with tab5:
    st.header("Diversity & Relationships")

    if "species" in df_filtered:
        group_opts = [c for c in ["Country","year","stateProvince"] if c in df]
        sel = st.selectbox("Group by", group_opts)

        d = df_filtered.dropna(subset=["species"])
        grouped = d.groupby([sel,"species"]).size().reset_index(name="n")

        def shannon(counts):
            counts = counts[counts > 0]
            if counts.sum() == 0:
                return np.nan
            p = counts / counts.sum()
            return float(-(p * np.log(p)).sum())

        rows = []
        for key, sub in grouped.groupby(sel):
            rows.append({
                sel: key,
                "records": int(sub["n"].sum()),
                "species": int(sub["species"].nunique()),
                "shannon": shannon(sub["n"])
            })

        rdf = pd.DataFrame(rows).dropna(subset=["shannon"]).sort_values("shannon", ascending=False)

        st.subheader("Shannon Diversity")
        st.dataframe(rdf, use_container_width=True)

        st.plotly_chart(
            px.bar(rdf.head(20), x=sel, y="shannon").update_layout(xaxis_tickangle=-45),
            use_container_width=True
        )

# ===============================================================
# TAB 6 — PEOPLE SECTION
# ===============================================================

with tab6:
    st.header("People Behind Observations")

    c1, c2, c3 = st.columns(3)
    c1.metric("Identifiers", df_filtered["identifiedBy"].nunique())
    c2.metric("Rights Holders", df_filtered["rightsHolder"].nunique())
    c3.metric("Recorders", df_filtered["recordedBy"].nunique())

    def ppl_chart(col, title):
        vc = df_filtered[col].fillna("Unknown").value_counts().head(15)
        fig = px.bar(vc)
        fig.update_layout(xaxis_tickangle=-45)
        st.subheader(title)
        st.plotly_chart(fig, use_container_width=True)

    ppl_chart("identifiedBy", "Top Identifiers")
    ppl_chart("rightsHolder", "Top Rights Holders")
    ppl_chart("recordedBy", "Top Recorders")

    st.subheader("Sample Records")
    cols = [
        "scientificName","kingdom","family","genus","species",
        "Country","countryCode","identifiedBy","rightsHolder","recordedBy","eventDate"
    ]
    cols = [c for c in cols if c in df]
    st.dataframe(df_filtered[cols].head(200), use_container_width=True)
