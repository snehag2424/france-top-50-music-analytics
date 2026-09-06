import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Data/Atlantic_France.csv")

# Basic information
print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nAlbum Types:")
print(df["album_type"].value_counts())

print("\nExplicit Content:")
print(df["is_explicit"].value_counts())

# data validation ```python


print("\n" + "=" * 50)
print("DATA VALIDATION")
print("=" * 50)

# Check date format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\nDate Range:")
print(df["date"].min(), "to", df["date"].max())

# Check positions
print("\nMinimum Position:")
print(df["position"].min())

print("\nMaximum Position:")
print(df["position"].max())

# Check whether each date has 50 songs
songs_per_day = df.groupby("date").size()

print("\nSongs Per Day:")
print(songs_per_day.value_counts())

print("\nDays with exactly 50 songs:")
print((songs_per_day == 50).sum())

# Check invalid positions
invalid_positions = df[
    (df["position"] < 1) |
    (df["position"] > 50)
    ]

print("\nInvalid Positions:")
print(len(invalid_positions))

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

#           dsts clesaning

print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

# Make a copy
clean_df = df.copy()

# Remove duplicate rows
clean_df = clean_df.drop_duplicates()

# Remove rows where important values are missing
clean_df = clean_df.dropna(
    subset=["date", "position", "song", "artist"]
)

# Convert numerical columns
numeric_columns = [
    "position",
    "popularity",
    "duration_ms",
    "total_tracks"
]

for column in numeric_columns:
    clean_df[column] = pd.to_numeric(
        clean_df[column],
        errors="coerce"
    )

# Fill missing popularity with median
clean_df["popularity"] = clean_df["popularity"].fillna(
    clean_df["popularity"].median()
)

# Fill missing duration with median
clean_df["duration_ms"] = clean_df["duration_ms"].fillna(
    clean_df["duration_ms"].median()
)

# Convert explicit column to boolean
# Convert explicit column safely to boolean
clean_df["is_explicit"] = (
    clean_df["is_explicit"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False
    })
)

# Remove rows where explicit value could not be identified
clean_df = clean_df.dropna(subset=["is_explicit"])

print("\nExplicit Flag Values After Cleaning:")
print(clean_df["is_explicit"].value_counts())

# ============================================================
# EXPLICIT FLAG VALIDATION
# ============================================================

print("\n" + "=" * 50)
print("EXPLICIT FLAG VALIDATION")
print("=" * 50)

print("\nExplicit Flag Values:")
print(clean_df["is_explicit"].value_counts())

invalid_explicit = clean_df[
    ~clean_df["is_explicit"].isin([True, False])
]

print("\nInvalid Explicit Flags:")
print(len(invalid_explicit))

print("\nShape Before Cleaning:")
print(df.shape)

print("\nShape After Cleaning:")
print(clean_df.shape)

print("\nMissing Values After Cleaning:")
print(clean_df.isnull().sum())

# ============================================================
# ALBUM TYPE STANDARDIZATION
# ============================================================

print("\n" + "=" * 50)
print("ALBUM TYPE STANDARDIZATION")
print("=" * 50)

clean_df["album_type"] = (
    clean_df["album_type"]
    .str.lower()
    .str.strip()
)

print("\nStandardized Album Types:")
print(clean_df["album_type"].value_counts())

print("\nUnique Album Types:")
print(clean_df["album_type"].unique())

# create new colums

# ============================================
# 6. FEATURE ENGINEERING
# ============================================

# Convert duration from milliseconds to minutes
clean_df["duration_min"] = (
        clean_df["duration_ms"] / 60000
)


# Create ranking groups
def ranking_group(position):
    if position <= 10:
        return "Top 10"

    elif position <= 25:
        return "Top 25"

    else:
        return "Top 50"


clean_df["ranking_group"] = (
    clean_df["position"].apply(ranking_group)
)


# Create duration categories
def duration_category(minutes):
    if minutes < 2.5:
        return "< 2.5 min"

    elif minutes < 3:
        return "2.5 - 3 min"

    elif minutes < 3.5:
        return "3 - 3.5 min"

    elif minutes < 4:
        return "3.5 - 4 min"

    else:
        return "> 4 min"


clean_df["duration_category"] = (
    clean_df["duration_min"].apply(duration_category)
)

print("\nNew Columns:")
print(clean_df.columns.tolist())

# ============================================
# 7. EXPLICIT CONTENT ANALYSIS
# ============================================

print("\n" + "=" * 50)
print("EXPLICIT CONTENT ANALYSIS")
print("=" * 50)

explicit_counts = (
    clean_df["is_explicit"].value_counts()
)

print("\nExplicit vs Clean Songs:")
print(explicit_counts)

explicit_percentage = (
        clean_df["is_explicit"].mean() * 100
)

print("\nExplicit Song Percentage:")
print(round(explicit_percentage, 2), "%")

clean_percentage = 100 - explicit_percentage

print("\nClean Song Percentage:")
print(round(clean_percentage, 2), "%")

# Popularity comparison

explicit_popularity = clean_df[
    clean_df["is_explicit"] == True
    ]["popularity"].mean()

clean_popularity = clean_df[
    clean_df["is_explicit"] == False
    ]["popularity"].mean()

print("\nAverage Popularity - Explicit:")
print(round(explicit_popularity, 2))

print("\nAverage Popularity - Clean:")
print(round(clean_popularity, 2))

# ============================================
# 8. EXPLICIT CONTENT BY RANKING
# ============================================

ranking_explicit = pd.crosstab(
    clean_df["ranking_group"],
    clean_df["is_explicit"]
)

print("\nExplicit Content by Ranking:")
print(ranking_explicit)

ranking_percentage = pd.crosstab(
    clean_df["ranking_group"],
    clean_df["is_explicit"],
    normalize="index"
) * 100

print("\nExplicit Percentage by Ranking Group:")
print(ranking_percentage)

# ============================================
# 9. SINGLE VS ALBUM ANALYSIS
# ============================================

print("\n" + "=" * 50)
print("SINGLE VS ALBUM ANALYSIS")
print("=" * 50)

print("\nAlbum Type Count:")
print(
    clean_df["album_type"].value_counts()
)

print("\nAlbum Type Percentage:")
print(
    clean_df["album_type"]
    .value_counts(normalize=True) * 100
)

print("\nAverage Popularity by Album Type:")
print(
    clean_df.groupby("album_type")["popularity"]
    .mean()
)

print("\nAverage Position by Album Type:")
print(
    clean_df.groupby("album_type")["position"]
    .mean()
)

# ============================================
# 10. ALBUM SIZE ANALYSIS
# ============================================

print("\n" + "=" * 50)
print("ALBUM SIZE ANALYSIS")
print("=" * 50)

print("\nAverage Album Size:")
print(
    clean_df["total_tracks"].mean()
)

print("\nMinimum Album Size:")
print(
    clean_df["total_tracks"].min()
)

print("\nMaximum Album Size:")
print(
    clean_df["total_tracks"].max()
)

print("\nAlbum Size vs Popularity:")
print(
    clean_df.groupby("total_tracks")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# ============================================
# 11. SONG DURATION ANALYSIS
# ============================================

print("\n" + "=" * 50)
print("SONG DURATION ANALYSIS")
print("=" * 50)

print("\nAverage Duration:")
print(
    round(
        clean_df["duration_min"].mean(),
        2
    ),
    "minutes"
)

print("\nShortest Song:")
print(
    round(
        clean_df["duration_min"].min(),
        2
    ),
    "minutes"
)

print("\nLongest Song:")
print(
    round(
        clean_df["duration_min"].max(),
        2
    ),
    "minutes"
)

print("\nAverage Duration by Ranking:")
print(
    clean_df.groupby(
        "ranking_group"
    )["duration_min"].mean()
)

# ============================================
# 12. DURATION CATEGORY ANALYSIS
# ============================================

print("\nSongs by Duration Category:")

print(
    clean_df["duration_category"]
    .value_counts()
)

print("\nPopularity by Duration Category:")

print(
    clean_df.groupby(
        "duration_category"
    )["popularity"]
    .mean()
    .sort_values(
        ascending=False
    )
)

# ============================================
# 13. RANKING ANALYSIS
# ============================================

print("\n" + "=" * 50)
print("RANKING ANALYSIS")
print("=" * 50)

print("\nAverage Popularity by Ranking Group:")

print(
    clean_df.groupby(
        "ranking_group"
    )["popularity"].mean()
)

print("\nAverage Duration by Ranking Group:")

print(
    clean_df.groupby(
        "ranking_group"
    )["duration_min"].mean()
)

print("\nExplicit Percentage by Ranking Group:")

print(
    clean_df.groupby(
        "ranking_group"
    )["is_explicit"].mean() * 100
)

# ============================================
# 14. TOP 10 SONGS
# ============================================

print("\n" + "=" * 50)
print("TOP 10 SONGS")
print("=" * 50)

top_10 = clean_df[
    clean_df["position"] <= 10
    ].sort_values("position")

print(
    top_10[
        [
            "position",
            "song",
            "artist",
            "popularity"
        ]
    ].to_string(index=False)
)

# ============================================
# 15. ARTIST ANALYSIS
# ============================================

print("\n" + "=" * 50)
print("ARTIST ANALYSIS")
print("=" * 50)

artist_counts = (
    clean_df["artist"].value_counts()
)

print("\nTop 10 Artists:")

print(
    artist_counts.head(10)
)

print("\nAverage Popularity by Artist:")

print(
    clean_df.groupby("artist")["popularity"]
    .mean()
    .sort_values(
        ascending=False
    )
    .head(10)
)

# ============================================
# 16. DATA VISUALIZATIONS
# ============================================
# Chart 1 — Explicit vs Clean
plt.figure(figsize=(7, 5))

clean_df["is_explicit"].value_counts().plot(
    kind="bar"
)

plt.title("Explicit vs Clean Songs")
plt.xlabel("Explicit Content")
plt.ylabel("Number of Songs")

plt.tight_layout()

plt.savefig(
    "outputs/charts/explicit_vs_clean.png"
)

plt.close()

# Chart 2 — Single vs Album
plt.figure(figsize=(7, 5))

clean_df["album_type"].value_counts().plot(
    kind="bar"
)

plt.title("Single vs Album Tracks")
plt.xlabel("Album Type")
plt.ylabel("Number of Songs")

plt.tight_layout()

plt.savefig(
    "outputs/charts/single_vs_album.png"
)

plt.close()

# Chart 4 — Duration vs Popularity
plt.figure(figsize=(8, 5))

plt.scatter(
    clean_df["duration_min"],
    clean_df["popularity"],
    alpha=0.5
)

plt.title("Song Duration vs Popularity")
plt.xlabel("Duration (minutes)")
plt.ylabel("Popularity")

plt.tight_layout()

plt.savefig(
    "outputs/charts/duration_vs_popularity.png"
)

plt.close()

# Chart 5 — Rank vs Popularity
plt.figure(figsize=(8, 5))

plt.scatter(
    clean_df["position"],
    clean_df["popularity"],
    alpha=0.5
)

plt.title("Chart Position vs Popularity")
plt.xlabel("Chart Position")
plt.ylabel("Popularity")

plt.gca().invert_xaxis()

plt.tight_layout()

plt.savefig(
    "outputs/charts/rank_vs_popularity.png"
)

plt.close()

plt.figure(figsize=(8, 5))

plt.scatter(
    clean_df["total_tracks"],
    clean_df["popularity"],
    alpha=0.5
)

plt.title("Album Size vs Popularity")
plt.xlabel("Total Tracks")
plt.ylabel("Popularity")

plt.tight_layout()

plt.savefig(
    "outputs/charts/album_size_vs_popularity.png"
)

plt.close()

# Chart 7 — Correlation Heatmap

plt.figure(figsize=(8, 6))

numeric_data = clean_df[
    [
        "position",
        "popularity",
        "duration_ms",
        "total_tracks"
    ]
]

correlation = numeric_data.corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "outputs/charts/correlation_heatmap.png"
)

plt.close()

# ============================================
# 17. FINAL KPIs
# ============================================

print("\n" + "=" * 50)
print("FINAL KPIs")
print("=" * 50)

total_songs = len(clean_df)

average_popularity = (
    clean_df["popularity"].mean()
)

explicit_share = (
        clean_df["is_explicit"].mean() * 100
)

average_duration = (
    clean_df["duration_min"].mean()
)

single_share = (
        (
                clean_df["album_type"] == "single"
        ).mean() * 100
)

album_share = (
        (
                clean_df["album_type"] == "album"
        ).mean() * 100
)

average_album_size = (
    clean_df["total_tracks"].mean()
)

print("Total Songs:", total_songs)

print(
    "Average Popularity:",
    round(average_popularity, 2)
)

print(
    "Explicit Content Share:",
    round(explicit_share, 2),
    "%"
)

print(
    "Average Song Duration:",
    round(average_duration, 2),
    "minutes"
)

print(
    "Single Share:",
    round(single_share, 2),
    "%"
)

print(
    "Album Share:",
    round(album_share, 2),
    "%"
)

print(
    "Average Album Size:",
    round(average_album_size, 2)
)

# save data setclean_df.to_csv
# Save cleaned dataset

clean_df.to_csv(
    "Data/Atlantic_France_Cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

# 1. Standardize Album Type

# ============================================================
# 2.CONTENT ATTRIBUTE CONCENTRATION ANALYSIS
# ============================================================

print("\n" + "=" * 50)
print("CONTENT ATTRIBUTE CONCENTRATION ANALYSIS")
print("=" * 50)

content_profile = clean_df.groupby("ranking_group").agg(
    explicit_share=("is_explicit", "mean"),
    average_popularity=("popularity", "mean"),
    average_duration=("duration_min", "mean"),
    single_share=("album_type", lambda x: (x == "single").mean()),
    album_share=("album_type", lambda x: (x == "album").mean())
)

content_profile["explicit_share"] *= 100
content_profile["single_share"] *= 100
content_profile["album_share"] *= 100

print("\nContent Profile by Ranking Tier:")
print(content_profile)

# ============================================================
# PREFERRED CONTENT PROFILE
# ============================================================

print("\n" + "=" * 50)
print("PREFERRED CONTENT PROFILE")
print("=" * 50)

best_popularity_tier = content_profile[
    "average_popularity"
].idxmax()

best_explicit_tier = content_profile[
    "explicit_share"
].idxmax()

best_single_tier = content_profile[
    "single_share"
].idxmax()

best_album_tier = content_profile[
    "album_share"
].idxmax()

print("\nHighest Average Popularity Tier:")
print(best_popularity_tier)

print("\nHighest Explicit Content Tier:")
print(best_explicit_tier)

print("\nHighest Single Representation Tier:")
print(best_single_tier)

print("\nHighest Album Representation Tier:")
print(best_album_tier)

print("\nPreferred Content Profile:")
print(content_profile.loc[best_popularity_tier])
# ============================================================
# RANK-BASED FORMAT COMPARISON
# ============================================================

print("\n" + "=" * 50)
print("RANK-BASED FORMAT COMPARISON")
print("=" * 50)

# ============================================================
# 3. CLEAN CONTENT DOMINANCE RATIO
# ============================================================

clean_count = (
        clean_df["is_explicit"] == False
).sum()

explicit_count = (
        clean_df["is_explicit"] == True
).sum()

# Clean content percentage
clean_content_share = (
                              clean_count / len(clean_df)
                      ) * 100

# Clean-to-explicit ratio
if explicit_count > 0:
    clean_content_dominance_ratio = (
            clean_count / explicit_count
    )
else:
    clean_content_dominance_ratio = np.nan

print("\nClean Content Share:")
print(
    round(clean_content_share, 2),
    "%"
)

print("\nClean Content Dominance Ratio:")
print(
    round(clean_content_dominance_ratio, 2)
)

# PART 4 — Single vs Album Track Ratio
# ============================================================
# SINGLE VS ALBUM TRACK RATIO
# ============================================================

single_count = (clean_df["album_type"] == "single").sum()
album_count = (clean_df["album_type"] == "album").sum()

single_album_ratio = single_count / album_count

print("\nSingle vs Album Track Ratio:")
print(round(single_album_ratio, 2))

# ============================================================
# 5,RANK-BASED FORMAT COMPARISON
# ============================================================

format_rank_analysis = clean_df.groupby(
    ["ranking_group", "album_type"]
).agg(
    track_count=("song", "count"),
    average_popularity=("popularity", "mean"),
    average_position=("position", "mean")
).reset_index()

print("\nRank-Based Format Comparison:")
print(format_rank_analysis)

# ============================================================
# 6..ALBUM DILUTION VS CONCENTRATION
# ============================================================

album_analysis = clean_df[
    clean_df["album_type"] == "album"
    ].groupby("total_tracks").agg(
    track_count=("song", "count"),
    average_popularity=("popularity", "mean"),
    average_position=("position", "mean")
).reset_index()

print("\nAlbum Size vs Popularity:")
print(album_analysis.sort_values(
    "average_popularity",
    ascending=False
).head(15))

album_size_correlation = clean_df[
    clean_df["album_type"] == "album"
    ][["total_tracks", "popularity"]].corr().iloc[0, 1]

print("\nAlbum Size vs Popularity Correlation:")
print(round(album_size_correlation, 3))

# ============================================================
# ALBUM SIZE IMPACT INDEX
# ============================================================

album_size_impact_index = abs(album_size_correlation) * 100

print("\nAlbum Size Impact Index:")
print(round(album_size_impact_index, 2))

# ============================================================
# ALBUM DILUTION VS CONCENTRATION EFFECT
# ============================================================

print("\n" + "=" * 50)
print("ALBUM DILUTION VS CONCENTRATION EFFECT")
print("=" * 50)

album_tracks = clean_df[
    clean_df["album_type"] == "album"
    ]

album_dilution_analysis = album_tracks.groupby(
    "total_tracks"
).agg(
    track_count=("song", "count"),
    average_popularity=("popularity", "mean")
).reset_index()

album_dilution_analysis["popularity_per_track"] = (
        album_dilution_analysis["average_popularity"] /
        album_dilution_analysis["total_tracks"]
)

print("\nAlbum Dilution Analysis:")
print(
    album_dilution_analysis.sort_values(
        "total_tracks"
    )
)

# Compare smaller and larger albums

small_albums = album_tracks[
    album_tracks["total_tracks"] <= 10
    ]

large_albums = album_tracks[
    album_tracks["total_tracks"] > 20
    ]

small_album_popularity = small_albums["popularity"].mean()
large_album_popularity = large_albums["popularity"].mean()

print("\nAverage Popularity of Smaller Albums (<=10 tracks):")
print(round(small_album_popularity, 2))

print("\nAverage Popularity of Larger Albums (>20 tracks):")
print(round(large_album_popularity, 2))

print("\nPopularity Difference:")
print(
    round(
        small_album_popularity - large_album_popularity,
        2
    )
)

# ============================================================
# 5. SONG DURATION PREFERENCE ANALYSIS
# ============================================================

print("\n" + "=" * 50)
print("SONG DURATION PREFERENCE ANALYSIS")
print("=" * 50)

print("\nDuration Statistics:")
print(clean_df["duration_min"].describe())

print("\nAverage Song Duration:")
print(round(clean_df["duration_min"].mean(), 2), "minutes")

print("\nDuration Category Distribution:")
print(
    clean_df["duration_category"].value_counts()
)
# ============================================================
# DURATION VS POPULARITY AND RANK ALIGNMENT
# ============================================================

print("\n" + "=" * 50)
print("DURATION VS POPULARITY AND RANK ALIGNMENT")
print("=" * 50)

duration_analysis = clean_df.groupby(
    "duration_category"
).agg(
    track_count=("song", "count"),
    average_popularity=("popularity", "mean"),
    average_position=("position", "mean"),
    average_duration=("duration_min", "mean")
).reset_index()

print("\nDuration Category Analysis:")
print(duration_analysis)

duration_popularity_correlation = clean_df[
    ["duration_min", "popularity"]
].corr().iloc[0, 1]

duration_rank_correlation = clean_df[
    ["duration_min", "position"]
].corr().iloc[0, 1]

print("\nDuration vs Popularity Correlation:")
print(round(duration_popularity_correlation, 3))

print("\nDuration vs Rank Correlation:")
print(round(duration_rank_correlation, 3))

# ============================================================
# DURATION AND RANK ALIGNMENT BY CATEGORY
# ============================================================

print("\n" + "=" * 50)
print("DURATION AND RANK ALIGNMENT BY CATEGORY")
print("=" * 50)

duration_rank_analysis = clean_df.groupby(
    "duration_category"
).agg(
    track_count=("song", "count"),
    average_position=("position", "mean"),
    average_popularity=("popularity", "mean")
).reset_index()

duration_rank_analysis = duration_rank_analysis.sort_values(
    "average_position"
)

print("\nDuration and Rank Analysis:")
print(duration_rank_analysis)
# ============================================================
# CONTENT ACCEPTANCE SCORE
# ============================================================

top_10 = clean_df[clean_df["position"] <= 10]

top_10_average_popularity = top_10["popularity"].mean()
overall_average_popularity = clean_df["popularity"].mean()

content_acceptance_score = (
                                   top_10_average_popularity /
                                   overall_average_popularity
                           ) * 100

print("\nContent Acceptance Score:")
print(round(content_acceptance_score, 2))

# next steps
print("\nAverage Popularity by Artist:")
print(
    clean_df.groupby("artist")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
# ============================================================
# DATA VISUALIZATIONS
# ============================================================
# ============================================================
# CONTENT ATTRIBUTE CONCENTRATION ANALYSIS
# ============================================================

print("\n" + "=" * 50)
print("CONTENT ATTRIBUTE CONCENTRATION ANALYSIS")
print("=" * 50)

content_profile = clean_df.groupby("ranking_group").agg(
    explicit_share=("is_explicit", "mean"),
    average_popularity=("popularity", "mean"),
    average_duration=("duration_min", "mean"),
    single_share=("album_type", lambda x: (x == "single").mean()),
    album_share=("album_type", lambda x: (x == "album").mean())
)

content_profile["explicit_share"] *= 100
content_profile["single_share"] *= 100
content_profile["album_share"] *= 100

print("\nContent Profile by Ranking Tier:")
print(content_profile)
