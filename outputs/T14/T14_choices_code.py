import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
df = pd.read_csv("owid-covid-data.csv", low_memory=False)

# Ensure dates and death rates have the correct data types.
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["total_deaths_per_million"] = pd.to_numeric(
    df["total_deaths_per_million"],
    errors="coerce"
)

# --------------------------------------------------
# 2. Select country and territory records
# --------------------------------------------------
# A missing continent is used as a practical indicator
# of an aggregate location, such as World or a region.
countries = df.loc[
    df["continent"].notna()
    & df["location"].notna()
    & df["date"].notna()
    & df["total_deaths_per_million"].notna(),
    [
        "location",
        "date",
        "total_deaths_per_million"
    ]
].copy()

# --------------------------------------------------
# 3. Find the latest valid observation per location
# --------------------------------------------------
# Filter missing rates first so that a missing value on
# a later date does not replace an earlier valid value.
latest = (
    countries
    .sort_values("date")
    .groupby("location", sort=False)
    .tail(1)
    .sort_values("total_deaths_per_million")
    .reset_index(drop=True)
)

if latest.empty:
    raise ValueError(
        "No eligible locations have a valid "
        "total_deaths_per_million value."
    )

# Include the observation date in each country label.
latest["chart_label"] = latest.apply(
    lambda row: (
        f"{row['location']} "
        f"({row['date']:%Y-%m-%d})"
    ),
    axis=1
)

# --------------------------------------------------
# 4. Create the horizontal bar chart
# --------------------------------------------------
rates = latest["total_deaths_per_million"]
labels = latest["chart_label"]

# Scale the figure height to keep country labels readable.
fig_height = max(8, 0.24 * len(latest))

fig, ax = plt.subplots(
    figsize=(12, fig_height)
)

bars = ax.barh(
    labels,
    rates
)

# Titles and axis labels.
ax.set_title(
    "Latest Available Cumulative COVID-19 Deaths per Million",
    fontsize=14,
    pad=12
)

ax.set_xlabel("Reported COVID-19 deaths per million people")
ax.set_ylabel("Country / territory (latest observation date)")

# Display each rate at the end of its bar.
ax.bar_label(
    bars,
    fmt="%.1f",
    padding=3,
    fontsize=7
)

# Leave room for the numeric labels at the right edge.
max_rate = rates.max()
ax.set_xlim(
    0,
    max_rate * 1.12 if max_rate > 0 else 1
)

# Add subtle gridlines to help compare values.
ax.set_axisbelow(True)
ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.35
)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. Optional: inspect the data used in the chart
# --------------------------------------------------
print(
    latest[
        [
            "location",
            "date",
            "total_deaths_per_million"
        ]
    ]
    .sort_values(
        "total_deaths_per_million",
        ascending=False
    )
    .to_string(index=False)
)