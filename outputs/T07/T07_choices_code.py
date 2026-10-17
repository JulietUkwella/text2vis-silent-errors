import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
df = pd.read_csv(
    "owid-covid-data.csv",
    parse_dates=["date"]
)

# --------------------------------------------------
# 2. Keep only United States country-level records
# --------------------------------------------------
us = df.loc[
    (df["iso_code"] == "USA") &
    (df["location"] == "United States"),
    ["date", "total_deaths"]
].copy()

if us.empty:
    raise ValueError(
        "No United States records found. Check the CSV file and filters."
    )

us["date"] = pd.to_datetime(us["date"], errors="coerce")
us["total_deaths"] = pd.to_numeric(
    us["total_deaths"], errors="coerce"
)
us = us.dropna(subset=["date"])

# The comparison requires one record per date.
if us["date"].duplicated().any():
    raise ValueError(
        "Duplicate United States dates found. "
        "Check the data before calculating annual totals."
    )

# --------------------------------------------------
# 3. Get cumulative deaths at each year-end
# --------------------------------------------------
year_end_dates = [
    pd.Timestamp("2020-12-31"),
    pd.Timestamp("2021-12-31"),
    pd.Timestamp("2022-12-31"),
]

cumulative_deaths = {}

for date in year_end_dates:
    values = us.loc[
        us["date"] == date, "total_deaths"
    ].dropna()

    if values.empty:
        raise ValueError(
            f"Missing cumulative death total for {date.date()}. "
            "Check whether the CSV contains a valid value "
            "for this exact date."
        )

    cumulative_deaths[date.year] = float(values.iloc[0])

# --------------------------------------------------
# 4. Calculate the increase during each calendar year
# --------------------------------------------------
annual_deaths = {
    2021: (
        cumulative_deaths[2021] - cumulative_deaths[2020]
    ),
    2022: (
        cumulative_deaths[2022] - cumulative_deaths[2021]
    ),
}

print("Increase in cumulative reported COVID-19 deaths:")
for year, deaths in annual_deaths.items():
    print(f"{year}: {deaths:,.0f}")

if annual_deaths[2021] > annual_deaths[2022]:
    print("\nThere were more reported COVID-19 deaths in 2021.")
elif annual_deaths[2022] > annual_deaths[2021]:
    print("\nThere were more reported COVID-19 deaths in 2022.")
else:
    print("\nThe reported totals were equal.")

# --------------------------------------------------
# 5. Create the bar chart
# --------------------------------------------------
years = list(annual_deaths.keys())
deaths = list(annual_deaths.values())

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    [str(year) for year in years],
    deaths
)

# Add the numerical total above each bar.
ax.bar_label(
    bars,
    labels=[f"{value:,.0f}" for value in deaths],
    padding=4
)

ax.set_title(
    "Reported COVID-19 Deaths Added in the United States\n"
    "2021 Compared with 2022"
)
ax.set_xlabel("Calendar year")
ax.set_ylabel("Increase in cumulative reported deaths")
ax.yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

fig.text(
    0.01,
    0.01,
    "Calculated from December 31 cumulative totals. "
    "Historical figures may be revised.",
    fontsize=8
)

fig.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()