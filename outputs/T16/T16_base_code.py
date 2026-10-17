import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv", parse_dates=["date"])

# 2. Select the metric for cumulative excess mortality (%)
metric = "excess_mortality_cumulative"

# Keep country-level records, excluding regional and global aggregates
countries = df.loc[
    df["continent"].notna(),
    ["location", "date", metric]
].copy()

# Ensure the mortality values are numeric
countries[metric] = pd.to_numeric(
    countries[metric], errors="coerce"
)

# Remove rows without a valid date or excess mortality value
countries = countries.dropna(subset=["date", metric])

# 3. Get each country's latest available observation
latest = (
    countries
    .sort_values("date")
    .drop_duplicates(subset="location", keep="last")
)

# 4. Select the 15 countries with the highest excess mortality
top_countries = (
    latest.nlargest(15, metric)
    .sort_values(metric, ascending=True)
)

# 5. Create a horizontal bar chart
fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    top_countries["location"],
    top_countries[metric]
)

# Add chart title and axis labels
ax.set_title(
    "Countries with the Highest Cumulative Excess Mortality",
    fontsize=15,
    pad=15
)
ax.set_xlabel("Cumulative excess mortality (% above expected deaths)")
ax.set_ylabel("Country")

# Display the values beside the bars
ax.bar_label(bars, fmt="%.1f%%", padding=4)

# Improve readability
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.set_xlim(right=max(0, top_countries[metric].max()) * 1.15)

plt.tight_layout()
plt.show()