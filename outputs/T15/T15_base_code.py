import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset and parse dates
df = pd.read_csv("owid-covid-data.csv", parse_dates=["date"])

# 2. Filter data to 2023
df_2023 = df[df["date"].dt.year == 2023].copy()

# 3. Keep countries, excluding regional and global aggregates
df_2023 = df_2023[
    df_2023["continent"].notna()
    & ~df_2023["iso_code"].fillna("").str.startswith("OWID_")
]

# 4. Ensure the stringency index is numeric
df_2023["stringency_index"] = pd.to_numeric(
    df_2023["stringency_index"], errors="coerce"
)

# 5. Calculate average stringency for each country
country_stats = (
    df_2023.dropna(subset=["stringency_index"])
    .groupby("location")
    .agg(
        average_stringency=("stringency_index", "mean"),
        reporting_days=("stringency_index", "count")
    )
)

# Require at least 30 days of reported data
country_stats = country_stats[
    country_stats["reporting_days"] >= 30
]

# Select the 15 countries with the highest average stringency
top_15 = country_stats.nlargest(
    15, "average_stringency"
).sort_values("average_stringency")

# 6. Create the horizontal bar chart
plt.figure(figsize=(11, 7))

bars = plt.barh(
    top_15.index,
    top_15["average_stringency"]
)

# Add average values beside the bars
for bar in bars:
    value = bar.get_width()
    plt.text(
        value + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}",
        va="center",
        fontsize=9
    )

plt.title(
    "Countries with the Highest Average COVID-19 "
    "Stringency in 2023"
)
plt.xlabel("Average Stringency Index (0–100)")
plt.ylabel("Country")
plt.xlim(0, 110)
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()

# 7. Display the chart
plt.show()