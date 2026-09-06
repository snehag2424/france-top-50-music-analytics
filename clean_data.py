import pandas as pd

# Load dataset
df = pd.read_csv("Data/Atlantic_France.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# Convert duration from milliseconds to minutes
df["duration_min"] = df["duration_ms"] / 60000

# Standardize album type
df["album_type"] = (
    df["album_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Standardize album labels
df["album_type"] = df["album_type"].replace({
    "single": "Single",
    "album": "Album"
})

# Convert explicit flag to boolean
df["is_explicit"] = df["is_explicit"].astype(bool)

# Create content label
df["content_type"] = df["is_explicit"].map({
    True: "Explicit",
    False: "Clean"
})


# Create rank tier
def rank_tier(position):
    if position <= 10:
        return "Top 10"
    elif position <= 25:
        return "Top 25"
    else:
        return "Top 50"


df["rank_tier"] = df["position"].apply(rank_tier)


# Duration categories
def duration_category(duration):
    if duration < 3:
        return "Short (<3 min)"
    elif duration < 4:
        return "Medium (3–4 min)"
    else:
        return "Long (4+ min)"


df["duration_category"] = df["duration_min"].apply(duration_category)

# Save cleaned data
df.to_csv("Data/Atlantic_France_Cleaned.csv", index=False)

print("Cleaning completed.")
print(df.head())
