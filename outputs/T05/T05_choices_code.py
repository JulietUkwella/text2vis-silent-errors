import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
df = pd.read_csv("owid-covid-data.csv")

# Check that the necessary columns exist
required_columns = {"location", "continent", "date", "total_deaths"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"The dataset is missing these columns: {missing_columns}"
    )

# Convert columns to suitable data types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["total_deaths"] = pd.to_numeric(
    df["total_deaths"], errors="coerce"
)
df["continent"] = df["continent"].astype("string").str.strip()

# --------------------------------------------------
# 2. Select usable country/territory records
# --------------------------------------------------
# Exclude rows with missing continents, dates, or
# cumulative death counts. Aggregate rows without a
# continent assignment will also be excluded.

usable_rows = df.loc[
    df["continent"].notna()
    & df["continent"].ne("")
    & df["date"].notna()
    & df["total_deaths"].notna(),
    ["location", "continent", "date", "total_deaths"]
].copy()

if usable_rows.empty:
    raise ValueError("No usable rows were found in the dataset.")

# --------------------------------------------------
# 3. Keep the latest available death count per location
# --------------------------------------------------
# Do not sum across dates: total_deaths is cumulative.

latest_by_location = (
    usable_rows
    .sort_values(["location", "date"])
    .groupby(["location", "continent"], sort=False)
    .tail(1)
)

# --------------------------------------------------
# 4. Sum the location totals by continent
# --------------------------------------------------
continent_deaths = (
    latest_by_location
    .groupby("continent")["total_deaths"]
    .sum()
    .sort_values(ascending=False)
)

print("Reported cumulative COVID-19 deaths by continent:")
print(continent_deaths.round(0).astype("int64").map(
    lambda n: f"{n:,}"
))

# Show the range of latest observation dates included
# for each continent to reveal possible date differences.
date_coverage = (
    latest_by_location
    .groupby("continent")["date"]
    .agg(["min", "max"])
)

print("\nRange of latest observation dates used:")
print(date_coverage)

# --------------------------------------------------
# 5. Create the bar chart
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    continent_deaths.index,
    continent_deaths.values
)

ax.set_title(
    "Reported COVID-19 Deaths by Continent\n"
    "(Latest Available Cumulative Totals)"
)
ax.set_xlabel("Continent")
ax.set_ylabel("Reported deaths (people)")

# Format the vertical axis with thousands separators
ax.yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

# Add the death total above each bar
for bar, value in zip(bars, continent_deaths.values):
    ax.annotate(
        f"{value:,.0f}",
        xy=(bar.get_x() + bar.get_width() / 2,
            bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

plt.xticks(rotation=0)
fig.tight_layout()
plt.show()