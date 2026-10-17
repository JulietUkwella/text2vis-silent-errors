import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv")

# Parse dates and ensure death totals are numeric.
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["total_deaths"] = pd.to_numeric(
    df["total_deaths"], errors="coerce"
)

# 2. Keep country-level locations, excluding most regional/global aggregates.
# A nonmissing continent is used as the country-level filter.
# This can still include territories and special locations.
country_rows = df.loc[
    df["continent"].notna()
    & df["iso_code"].notna()
    & df["location"].notna()
    & df["date"].notna()
    & df["total_deaths"].notna()
    & (df["total_deaths"] >= 0)
].copy()

# 3. Handle duplicate location-date records.
# After invalid death totals have been removed, keep the last valid
# record for each location and date.
country_rows = country_rows.sort_values(
    ["location", "date"]
)
country_rows = country_rows.drop_duplicates(
    subset=["location", "date"],
    keep="last"
)

# 4. Get each location's latest valid cumulative death total.
# Do not add cumulative totals across dates.
latest = country_rows.loc[
    country_rows.groupby("location")["date"].idxmax()
].copy()

# 5. Rank locations and select the top 10.
top10 = (
    latest.sort_values("total_deaths", ascending=False)
    .head(10)
    .copy()
)

# Print values and dates to make differences in reporting periods visible.
print("Top 10 locations by cumulative reported COVID-19 deaths")
print(
    top10[["location", "date", "total_deaths"]].to_string(
        index=False,
        formatters={
            "date": lambda d: d.strftime("%Y-%m-%d"),
            "total_deaths": lambda n: f"{n:,.0f}"
        }
    )
)

# Stop if the dataset contains no eligible observations.
if top10.empty:
    raise ValueError(
        "No valid country-level death observations were found."
    )

# 6. Prepare data for a horizontal bar chart.
# Ascending order puts the largest death toll at the top.
plot_data = top10.sort_values(
    "total_deaths", ascending=True
)

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    plot_data["location"],
    plot_data["total_deaths"]
)

# Format the x-axis and add labels.
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, position: f"{x:,.0f}")
)

ax.set_title(
    "Top 10 Locations by Cumulative Reported COVID-19 Deaths",
    pad=14
)
ax.set_xlabel(
    "Cumulative reported deaths (latest valid observation)"
)
ax.set_ylabel("")

# Show exact death totals at the end of each bar.
for bar, value in zip(bars, plot_data["total_deaths"]):
    ax.annotate(
        f"{value:,.0f}",
        xy=(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2
        ),
        xytext=(5, 0),
        textcoords="offset points",
        va="center"
    )

ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.margins(x=0.15)

fig.tight_layout()
plt.show()