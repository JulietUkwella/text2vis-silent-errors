
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv")

# 2. Prepare the fields used in the analysis
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["total_deaths"] = pd.to_numeric(
    df["total_deaths"], errors="coerce"
)

# 3. Keep country/territory rows with standard three-letter ISO codes.
# This excludes aggregate rows such as World and continents.
iso3_mask = (
    df["iso_code"]
    .astype("string")
    .str.upper()
    .str.fullmatch(r"[A-Z]{3}", na=False)
)

locations = df.loc[
    iso3_mask,
    ["iso_code", "location", "date", "total_deaths"]
].copy()

# 4. Keep observations with a usable date and cumulative death total
locations = locations.dropna(
    subset=["date", "total_deaths"]
)

if locations.empty:
    raise ValueError(
        "No country/territory rows with valid dates and death totals "
        "were found. Check the CSV file and its columns."
    )

# 5. For each ISO code, take the latest non-missing cumulative death total.
# Do not sum total_deaths across dates: it is already cumulative.
latest = (
    locations
    .sort_values(["iso_code", "date"])
    .groupby("iso_code", sort=False)
    .tail(1)
)

# 6. Select the 10 locations with the highest reported death toll
top10 = (
    latest
    .nlargest(10, "total_deaths")
    .sort_values("total_deaths")
)

# 7. Plot a horizontal bar chart
fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    top10["location"],
    top10["total_deaths"]
)

# Add formatted death totals at the ends of the bars
for bar in bars:
    deaths = bar.get_width()
    ax.annotate(
        f"{deaths:,.0f}",
        xy=(deaths, bar.get_y() + bar.get_height() / 2),
        xytext=(5, 0),
        textcoords="offset points",
        va="center"
    )

ax.set_xlabel("Reported cumulative COVID-19 deaths")
ax.set_ylabel("Country or territory")
ax.set_title(
    "Top 10 Countries and Territories by Reported COVID-19 Deaths"
)

# Format the horizontal axis with thousands separators
ax.xaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

# Leave room for the death-total labels
max_deaths = top10["total_deaths"].max()
ax.set_xlim(0, max_deaths * 1.12 if max_deaths > 0 else 1)

ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

fig.text(
    0.01,
    0.01,
    "Uses the latest non-missing cumulative death total per location. "
    "Observation dates may differ; totals are reported deaths.",
    ha="left",
    fontsize=9
)

fig.tight_layout(rect=(0, 0.05, 1, 1))
plt.show()

# 8. Print the location with the highest reported death toll
winner = latest.loc[latest["total_deaths"].idxmax()]

print(
    f"Highest reported cumulative death toll: {winner['location']}\n"
    f"Reported deaths: {winner['total_deaths']:,.0f}\n"
    f"Last reported observation: {winner['date']:%Y-%m-%d}"
)
