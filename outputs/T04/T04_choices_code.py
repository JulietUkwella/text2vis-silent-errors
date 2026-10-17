import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv", low_memory=False)

# Ensure dates and case counts have appropriate data types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["total_cases"] = pd.to_numeric(df["total_cases"], errors="coerce")

# 2. Keep country-like locations and valid observations.
# A non-missing continent excludes most aggregate rows, such as
# the world, continents, and income groups. Territories may remain.
eligible = df.loc[
    df["continent"].notna()
    & df["date"].notna()
    & df["total_cases"].notna(),
    ["location", "continent", "date", "total_cases"]
].copy()

if eligible.empty:
    raise ValueError(
        "No country-like locations with valid total_cases observations found."
    )

# 3. Get the latest non-missing cumulative case total for each location.
latest = (
    eligible
    .sort_values(["location", "date"])
    .drop_duplicates(subset="location", keep="last")
)

# 4. Select and rank the 10 locations with the highest case totals.
top10 = (
    latest
    .nlargest(10, "total_cases")
    .sort_values("total_cases", ascending=True)
    .copy()
)

# Convert case totals to millions for easier chart reading.
top10["cases_millions"] = top10["total_cases"] / 1_000_000

# 5. Create the horizontal bar chart.
fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(
    top10["location"],
    top10["cases_millions"]
)

# Add the total and the observation date beside each bar.
max_cases = top10["cases_millions"].max()

for bar, (_, row) in zip(bars, top10.iterrows()):
    cases = row["cases_millions"]
    report_date = row["date"].strftime("%Y-%m-%d")

    ax.text(
        bar.get_width() + max_cases * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{cases:,.1f}M  ({report_date})",
        va="center",
        fontsize=9
    )

# 6. Format the chart.
ax.set_title(
    "Top 10 Countries/Country-like Locations by COVID-19 Cases",
    fontsize=14,
    pad=14
)
ax.set_xlabel("Cumulative reported COVID-19 cases (millions)")
ax.set_ylabel("Location")

# Keep enough room for the case totals and dates displayed beside the bars.
ax.set_xlim(0, max_cases * 1.35 if max_cases > 0 else 1)

ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x:,.0f}")
)

ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# Remove unnecessary borders for a cleaner chart.
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()