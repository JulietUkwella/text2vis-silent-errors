import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Configure the file, countries, and measure
# --------------------------------------------------
file_path = "owid-covid-data.csv"

g7_countries = [
    "Canada",
    "France",
    "Germany",
    "Italy",
    "Japan",
    "United Kingdom",
    "United States"
]

metric = "people_fully_vaccinated_per_hundred"

# Load only the columns needed for this analysis
df = pd.read_csv(
    file_path,
    usecols=["location", "date", metric],
    parse_dates=["date"]
)

# --------------------------------------------------
# 2. Keep only the seven G7 countries
# --------------------------------------------------
g7 = df.loc[df["location"].isin(g7_countries)].copy()

# Make sure the vaccination measure is numeric
g7[metric] = pd.to_numeric(g7[metric], errors="coerce")

# Missing dates or rates cannot be used for the chart
g7 = g7.dropna(subset=["date", metric])

# --------------------------------------------------
# 3. Get each country's latest usable observation
# --------------------------------------------------
latest = (
    g7.sort_values(["location", "date"])
      .drop_duplicates(subset="location", keep="last")
      .copy()
)

# Do not silently omit countries with no usable data
missing_countries = sorted(
    set(g7_countries) - set(latest["location"])
)

if missing_countries:
    raise ValueError(
        "No usable vaccination data found for: "
        + ", ".join(missing_countries)
    )

# Add each observation date to the country label
latest["country_label"] = latest.apply(
    lambda row: (
        f"{row['location']} ({row['date']:%Y-%m-%d})"
    ),
    axis=1
)

# Sort from lowest to highest vaccination percentage
latest = latest.sort_values(metric, ascending=True)

# --------------------------------------------------
# 4. Create the horizontal bar chart
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.5))

bars = ax.barh(
    latest["country_label"],
    latest[metric]
)

# Allow values above 100 to remain visible
max_rate = latest[metric].max()
x_max = max(100.0, max_rate * 1.12 + 1)
ax.set_xlim(0, x_max)

# Add percentage labels at the end of each bar
label_offset = max(0.5, x_max * 0.005)

for bar, value in zip(bars, latest[metric]):
    ax.text(
        value + label_offset,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center",
        ha="left"
    )

ax.set_title(
    "G7 Countries: Population Fully Vaccinated Against COVID-19\n"
    "Latest Non-Missing Observation for Each Country"
)

ax.set_xlabel(
    "Fully vaccinated people per 100 population (%)"
)
ax.set_ylabel("Country (observation date)")

# Improve readability
ax.set_axisbelow(True)
ax.grid(axis="x", linestyle="--", alpha=0.35)

fig.text(
    0.01,
    0.01,
    "Source: Our World in Data CSV. Reporting dates may differ by country.",
    fontsize=9
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

plt.show()