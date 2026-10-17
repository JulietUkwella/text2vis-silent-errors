import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv", parse_dates=["date"])

# 2. Keep country rows and exclude aggregate locations
countries = df[
    df["continent"].notna()
    & ~df["iso_code"].astype(str).str.startswith("OWID_")
].copy()

# 3. Ensure the death-rate column is numeric
countries["total_deaths_per_million"] = pd.to_numeric(
    countries["total_deaths_per_million"], errors="coerce"
)

# 4. Remove rows where the death-rate value is missing
countries = countries.dropna(subset=["total_deaths_per_million"])

# 5. Get each country's latest available observation
latest = (
    countries.sort_values("date")
    .groupby("location", as_index=False)
    .tail(1)
    .sort_values("total_deaths_per_million")
)

# 6. Create a horizontal bar chart
fig, ax = plt.subplots(
    figsize=(12, max(8, len(latest) * 0.25))
)

ax.barh(
    latest["location"],
    latest["total_deaths_per_million"]
)

ax.set_title(
    "Latest Cumulative COVID-19 Deaths per Million by Country",
    fontsize=15
)
ax.set_xlabel("Total COVID-19 deaths per million people")
ax.set_ylabel("Country")

ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(
    "latest_covid_death_rate_per_million.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()