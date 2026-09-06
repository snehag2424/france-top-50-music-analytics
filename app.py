# ============================================================
# 1. IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="France Top 50 Music Analytics",
    page_icon="🎵",
    layout="wide"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    data = pd.read_csv(
        "Data/Atlantic_France_Cleaned.csv"
    )

    # Convert date
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    # Convert duration
    data["duration_min"] = (
            data["duration_ms"] / 60000
    )

    return data


df = load_data()

# ============================================================
# 4. DASHBOARD TITLE
# ============================================================

st.title(
    "🎵 France Top 50 Music Analytics Dashboard"
)

st.markdown(
    """
    ### Audience Sensitivity, Content Compliance &
    Format Preference Analysis

    This dashboard analyzes France Top 50 playlist data
    to understand explicit content, release format,
    song duration, album structure and ranking patterns.
    """
)

st.divider()

# ============================================================
# 5. SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🎛️ Dashboard Filters")

# -------------------------
# Date Filter
# -------------------------

min_date = df["date"].min().date()
max_date = df["date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# -------------------------
# Rank Tier Filter
# -------------------------

rank_tier = st.sidebar.selectbox(
    "Select Rank Tier",
    [
        "Top 50",
        "Top 25",
        "Top 10"
    ]
)

# -------------------------
# Explicit Content Filter
# -------------------------

explicit_filter = st.sidebar.selectbox(
    "Explicit Content",
    [
        "All",
        "Explicit Only",
        "Clean Only"
    ]
)

# -------------------------
# Album Type Filter
# -------------------------

album_filter = st.sidebar.multiselect(
    "Album Type",
    options=sorted(
        df["album_type"]
        .dropna()
        .unique()
        .tolist()
    ),
    default=sorted(
        df["album_type"]
        .dropna()
        .unique()
        .tolist()
    )
)

# ============================================================
# 6. APPLY FILTERS
# ============================================================

# Base dataset after Date + Explicit + Album Type filters
# Used for ranking comparisons and analysis

base_filtered_df = df.copy()

# -------------------------
# Date filtering
# -------------------------

if isinstance(selected_dates, tuple):

    if len(selected_dates) == 2:
        start_date = pd.Timestamp(
            selected_dates[0]
        )

        end_date = pd.Timestamp(
            selected_dates[1]
        )

        base_filtered_df = base_filtered_df[
            (base_filtered_df["date"] >= start_date) &
            (base_filtered_df["date"] <= end_date)
            ]

# -------------------------
# Explicit filtering
# -------------------------

if explicit_filter == "Explicit Only":

    base_filtered_df = base_filtered_df[
        base_filtered_df["is_explicit"] == True
        ]

elif explicit_filter == "Clean Only":

    base_filtered_df = base_filtered_df[
        base_filtered_df["is_explicit"] == False
        ]

# -------------------------
# Album type filtering
# -------------------------

base_filtered_df = base_filtered_df[
    base_filtered_df["album_type"].isin(
        album_filter
    )
]

# -------------------------
# Apply selected Rank Tier
# -------------------------

filtered_df = base_filtered_df.copy()

if rank_tier == "Top 10":

    filtered_df = filtered_df[
        filtered_df["position"] <= 10
        ]

elif rank_tier == "Top 25":

    filtered_df = filtered_df[
        filtered_df["position"] <= 25
        ]

elif rank_tier == "Top 50":

    filtered_df = filtered_df[
        filtered_df["position"] <= 50
        ]

# ============================================================
# 7. CHECK FILTERED DATA
# ============================================================

if filtered_df.empty:
    st.warning(
        "No songs match the selected filters. "
        "Please change the filters."
    )

    st.stop()

# ============================================================
# 8. KPI CALCULATIONS
# ============================================================

total_songs = len(filtered_df)

average_popularity = (
    filtered_df["popularity"].mean()
)

explicit_share = (
        filtered_df["is_explicit"].mean() * 100
)

clean_share = (
        100 - explicit_share
)

average_duration = (
    filtered_df["duration_min"].mean()
)

single_share = (
        (
                filtered_df["album_type"] == "single"
        ).mean() * 100
)

album_share = (
        (
                filtered_df["album_type"] == "album"
        ).mean() * 100
)

average_album_size = (
    filtered_df["total_tracks"].mean()
)

# ============================================================
# 9. KPI CARDS
# ============================================================

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Songs",
        total_songs
    )

with col2:
    st.metric(
        "Average Popularity",
        f"{average_popularity:.2f}"
    )

with col3:
    st.metric(
        "Explicit Content",
        f"{explicit_share:.2f}%"
    )

with col4:
    st.metric(
        "Clean Content",
        f"{clean_share:.2f}%"
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        "Average Duration",
        f"{average_duration:.2f} min"
    )

with col6:
    st.metric(
        "Single Share",
        f"{single_share:.2f}%"
    )

with col7:
    st.metric(
        "Album Share",
        f"{album_share:.2f}%"
    )

with col8:
    st.metric(
        "Average Album Size",
        f"{average_album_size:.2f} tracks"
    )

st.divider()

# ============================================================
# 10. EXPLICIT VS CLEAN ANALYSIS
# ============================================================

st.subheader(
    "🔞 Explicit vs Clean Content Analysis"
)

explicit_data = (
    filtered_df["is_explicit"]
    .map({
        True: "Explicit",
        False: "Clean"
    })
    .value_counts()
    .reset_index()
)

explicit_data.columns = [
    "Content Type",
    "Number of Songs"
]

col1, col2 = st.columns(2)

with col1:
    fig_explicit = px.bar(
        explicit_data,
        x="Content Type",
        y="Number of Songs",
        title="Explicit vs Clean Songs"
    )

    st.plotly_chart(
        fig_explicit,
        use_container_width=True
    )

with col2:
    fig_pie = px.pie(
        explicit_data,
        names="Content Type",
        values="Number of Songs",
        title="Content Share"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

# ============================================================
# 11. POPULARITY: EXPLICIT VS CLEAN
# ============================================================

popularity_content = (
    filtered_df
    .groupby("is_explicit")["popularity"]
    .mean()
    .reset_index()
)

popularity_content["Content Type"] = (
    popularity_content["is_explicit"]
    .map({
        True: "Explicit",
        False: "Clean"
    })
)

fig_popularity = px.bar(
    popularity_content,
    x="Content Type",
    y="popularity",
    title="Average Popularity: Explicit vs Clean",
    labels={
        "popularity": "Average Popularity"
    }
)

st.plotly_chart(
    fig_popularity,
    use_container_width=True
)

# ============================================================
# 12. SINGLE VS ALBUM ANALYSIS
# ============================================================

st.subheader(
    "💿 Single vs Album Format Analysis"
)

format_data = (
    filtered_df["album_type"]
    .value_counts()
    .reset_index()
)

format_data.columns = [
    "Album Type",
    "Number of Songs"
]

col1, col2 = st.columns(2)

with col1:
    fig_format = px.bar(
        format_data,
        x="Album Type",
        y="Number of Songs",
        title="Single vs Album Tracks"
    )

    st.plotly_chart(
        fig_format,
        use_container_width=True
    )

with col2:
    fig_format_pie = px.pie(
        format_data,
        names="Album Type",
        values="Number of Songs",
        title="Format Distribution"
    )

    st.plotly_chart(
        fig_format_pie,
        use_container_width=True
    )

# ============================================================
# 13. FORMAT VS POPULARITY
# ============================================================

format_popularity = (
    filtered_df
    .groupby("album_type")["popularity"]
    .mean()
    .reset_index()
)

fig_format_popularity = px.bar(
    format_popularity,
    x="album_type",
    y="popularity",
    title="Average Popularity by Album Type",
    labels={
        "album_type": "Album Type",
        "popularity": "Average Popularity"
    }
)

st.plotly_chart(
    fig_format_popularity,
    use_container_width=True
)

# ============================================================
# 14. SONG DURATION ANALYSIS
# ============================================================

st.subheader(
    "⏱️ Song Duration Analysis"
)

col1, col2 = st.columns(2)

with col1:
    fig_duration = px.histogram(
        filtered_df,
        x="duration_min",
        nbins=20,
        title="Song Duration Distribution",
        labels={
            "duration_min": "Duration (minutes)"
        }
    )

    st.plotly_chart(
        fig_duration,
        use_container_width=True
    )

with col2:
    fig_duration_pop = px.scatter(
        filtered_df,
        x="duration_min",
        y="popularity",
        hover_data=[
            "song",
            "artist",
            "position"
        ],
        title="Duration vs Popularity",
        labels={
            "duration_min": "Duration (minutes)",
            "popularity": "Popularity"
        }
    )

    st.plotly_chart(
        fig_duration_pop,
        use_container_width=True
    )

# ============================================================
# 15. DURATION CATEGORY ANALYSIS
# ============================================================

duration_categories = pd.cut(
    filtered_df["duration_min"],
    bins=[
        -np.inf,
        2.5,
        3,
        3.5,
        4,
        np.inf
    ],
    labels=[
        "< 2.5 min",
        "2.5 - 3 min",
        "3 - 3.5 min",
        "3.5 - 4 min",
        "> 4 min"
    ]
)

duration_category_data = (
    duration_categories
    .value_counts()
    .sort_index()
    .reset_index()
)

duration_category_data.columns = [
    "Duration Category",
    "Number of Songs"
]

fig_duration_category = px.bar(
    duration_category_data,
    x="Duration Category",
    y="Number of Songs",
    title="Songs by Duration Category"
)

st.plotly_chart(
    fig_duration_category,
    use_container_width=True
)

# ============================================================
# 16. RANK VS POPULARITY
# ============================================================

st.subheader(
    "🏆 Ranking Analysis"
)

fig_rank = px.scatter(
    filtered_df,
    x="position",
    y="popularity",
    hover_data=[
        "song",
        "artist"
    ],
    title="Chart Position vs Popularity",
    labels={
        "position": "Chart Position",
        "popularity": "Popularity"
    }
)

fig_rank.update_xaxes(
    autorange="reversed"
)

st.plotly_chart(
    fig_rank,
    use_container_width=True
)

# ============================================================
# 17. CONTENT ATTRIBUTE CONCENTRATION
# ============================================================

st.subheader(
    "🎯 Content Attribute Concentration by Ranking Tier"
)

content_results = []

for tier_name, rank_limit in [
    ("Top 10", 10),
    ("Top 25", 25),
    ("Top 50", 50)
]:

    tier_df = base_filtered_df[
        base_filtered_df["position"] <= rank_limit
        ]

    if not tier_df.empty:
        content_results.append({

            "Ranking Tier": tier_name,

            "Explicit Share": (
                    tier_df["is_explicit"].mean() * 100
            ),

            "Single Share": (
                    (tier_df["album_type"] == "single").mean() * 100
            ),

            "Album Share": (
                    (tier_df["album_type"] == "album").mean() * 100
            )
        })

content_profile = pd.DataFrame(
    content_results
)

if not content_profile.empty:

    content_melted = content_profile.melt(
        id_vars="Ranking Tier",
        value_vars=[
            "Explicit Share",
            "Single Share",
            "Album Share"
        ],
        var_name="Attribute",
        value_name="Percentage"
    )

    fig_content = px.bar(
        content_melted,
        x="Ranking Tier",
        y="Percentage",
        color="Attribute",
        barmode="group",
        title="Content Attributes by Ranking Tier",
        labels={
            "Ranking Tier": "Ranking Tier",
            "Percentage": "Percentage (%)"
        }
    )

    st.plotly_chart(
        fig_content,
        use_container_width=True
    )

else:

    st.info(
        "No ranking data available for the selected filters."
    )
# ============================================================
# 18. ALBUM SIZE VS POPULARITY
# ============================================================

st.subheader(
    "💿 Album Structure Analysis"
)

album_df = filtered_df[
    filtered_df["album_type"] == "album"
    ]

if not album_df.empty:

    fig_album = px.scatter(
        album_df,
        x="total_tracks",
        y="popularity",
        hover_data=[
            "song",
            "artist"
        ],
        title="Album Size vs Track Popularity",
        labels={
            "total_tracks": "Total Tracks in Album",
            "popularity": "Popularity"
        }
    )

    st.plotly_chart(
        fig_album,
        use_container_width=True
    )

else:

    st.info(
        "No album tracks available for "
        "the selected filters."
    )

# ============================================================
# 19. RANKING TIER TABLE
# ============================================================

st.subheader(
    "📋 Ranking Tier Comparison"
)

tier_results = []

for tier_name, rank_limit in [
    ("Top 10", 10),
    ("Top 25", 25),
    ("Top 50", 50)
]:

    tier_df = base_filtered_df[
        base_filtered_df["position"] <= rank_limit
        ]

    if not tier_df.empty:
        explicit_percentage = (
                tier_df["is_explicit"].mean() * 100
        )

        single_percentage = (
                (tier_df["album_type"] == "single").mean() * 100
        )

        album_percentage = (
                (tier_df["album_type"] == "album").mean() * 100
        )

        tier_results.append({

            "Ranking Tier": tier_name,

            "Songs": len(tier_df),

            "Explicit %": explicit_percentage,

            "Average Popularity":
                tier_df["popularity"].mean(),

            "Average Duration":
                tier_df["duration_min"].mean(),

            "Single %":
                single_percentage,

            "Album %":
                album_percentage
        })

tier_analysis = pd.DataFrame(
    tier_results
)

tier_analysis = tier_analysis.round(2)

st.dataframe(
    tier_analysis,
    use_container_width=True
)
# ============================================================
# 21. TOP 10 UNIQUE SONGS
# ============================================================

st.subheader("🎵 Top 10 Songs")

# Use the selected date/content filters, but consider
# all ranking positions within the selected period
song_summary = (
    base_filtered_df
    .groupby(["song", "artist"], as_index=False)
    .agg(
        avg_popularity=("popularity", "mean"),
        appearances=("song", "size"),
        avg_duration=("duration_min", "mean"),
        album_type=("album_type", "first"),
        is_explicit=("is_explicit", "first")
    )
)

# Rank unique songs by average popularity
top_10_songs = (
    song_summary
    .sort_values(
        ["avg_popularity", "appearances"],
        ascending=[False, False]
    )
    .head(10)
    .copy()
)

# Rename columns for dashboard display
top_10_songs = top_10_songs.rename(
    columns={
        "song": "Song",
        "artist": "Artist",
        "avg_popularity": "Average Popularity",
        "appearances": "Chart Appearances",
        "avg_duration": "Average Duration (min)",
        "album_type": "Album Type",
        "is_explicit": "Explicit"
    }
)

top_10_songs["Average Popularity"] = (
    top_10_songs["Average Popularity"].round(2)
)

top_10_songs["Average Duration (min)"] = (
    top_10_songs["Average Duration (min)"].round(2)
)

st.dataframe(
    top_10_songs,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# 22. AUTOMATIC KEY INSIGHTS
# ============================================================

st.subheader(
    "💡 Key Insights"
)

if not tier_analysis.empty:
    highest_popularity_tier = (
        tier_analysis.loc[
            tier_analysis["Average Popularity"].idxmax(),
            "Ranking Tier"
        ]
    )

    highest_explicit_tier = (
        tier_analysis.loc[
            tier_analysis["Explicit %"].idxmax(),
            "Ranking Tier"
        ]
    )

    longest_duration_tier = (
        tier_analysis.loc[
            tier_analysis["Average Duration"].idxmax(),
            "Ranking Tier"
        ]
    )

    st.markdown(
        f"""
        **1. Content Profile:**  
        The currently selected data contains
        **{explicit_share:.2f}%** explicit tracks and
        **{clean_share:.2f}%** clean tracks.

        **2. Popularity:**  
        The **{highest_popularity_tier}** tier has the
        highest average popularity among the ranking tiers.

        **3. Explicit Content Distribution:**  
        The **{highest_explicit_tier}** tier has the highest
        proportion of explicit tracks.

        **4. Song Duration:**  
        The average song duration is approximately
        **{average_duration:.2f} minutes**.

        **5. Format Distribution:**  
        Single tracks represent **{single_share:.2f}%**,
        while album tracks represent **{album_share:.2f}%**
        of the selected data.

        **6. Album Structure:**  
        The average represented album size is approximately
        **{average_album_size:.2f} tracks**.

        **7. Duration by Ranking:**  
        The **{longest_duration_tier}** tier has the highest
        average song duration.
        """
    )
# ============================================================
# 23. FOOTER
# ============================================================

st.divider()

st.caption(
    "France Top 50 Music Analytics | "
    "Audience Sensitivity, Content Compliance "
    "& Format Preference Analysis"
)
