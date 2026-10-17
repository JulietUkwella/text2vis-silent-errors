
import json
from urllib.request import urlopen

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load the COVID-19 dataset
# --------------------------------------------------
df = pd.read_csv(
    "owid-covid-data.csv",
    low_memory=False,
    parse_dates=["date"]
)

# Ensure numeric columns are numeric
for col in ["total_deaths", "population"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "iso_code"]).copy()

if df.empty:
    raise ValueError("The dataset contains no usable dated records.")


# --------------------------------------------------
# 2. Set a common cutoff date
# --------------------------------------------------
requested_cutoff = pd.Timestamp("2023-12-31")

dates_before_cutoff = df.loc[
    df["date"] <= requested_cutoff, "date"
]

if dates_before_cutoff.empty:
    # Fallback if the file contains only dates after
    # the requested cutoff.
    cutoff = df["date"].max()
    print(
        "No records exist on or before December 31, 2023. "
        f"Using the latest dataset date: {cutoff.date()}."
    )
else:
    cutoff = dates_before_cutoff.max()

print(f"Comparison cutoff: {cutoff.date()}")


# --------------------------------------------------
# 3. Retrieve World Bank income classifications
# --------------------------------------------------
url = (
    "https://api.worldbank.org/v2/country"
    "?format=json&per_page=400"
)

with urlopen(url, timeout=30) as response:
    wb_data = json.load(response)

if len(wb_data) < 2 or wb_data[1] is None:
    raise RuntimeError("Could not retrieve World Bank country data.")

income_rows = []

for country in wb_data[1]:
    iso3 = country.get("id")
    income = (country.get("incomeLevel") or {}).get("id")
    region = (country.get("region") or {}).get("id")

    # Retain actual countries/economies in the two groups.
    # Exclude regional and other aggregate entries.
    if iso3 and region != "NA" and income in ["LIC", "HIC"]:
        income_rows.append({
            "iso_code": iso3,
            "income_group": {
                "LIC": "Low income",
                "HIC": "High income"
            }[income]
        })

income_df = pd.DataFrame(income_rows).drop_duplicates("iso_code")

if income_df.empty:
    raise RuntimeError("No eligible World Bank income classifications found.")


# --------------------------------------------------
# 4. Get each country's latest reported deaths by cutoff
# --------------------------------------------------
eligible = df.loc[
    (df["date"] <= cutoff) & df["total_deaths"].notna()
].copy()

# Exclude non-country aggregates by matching to the
# World Bank's ISO-3 country codes in the later merge.
latest_deaths = (
    eligible
    .sort_values("date")
    .drop_duplicates("iso_code", keep="last")
    [["iso_code", "location", "date", "total_deaths"]]
)

# Get the latest available population on or before cutoff.
pop_rows = df.loc[
    (df["date"] <= cutoff) & df["population"].notna()
].copy()

latest_population = (
    pop_rows
    .sort_values("date")
    .drop_duplicates("iso_code", keep="last")
    [["iso_code", "population"]]
)

# Join deaths, population, and income group.
country_data = (
    latest_deaths
    .merge(latest_population, on="iso_code", how="inner")
    .merge(income_df, on="iso_code", how="inner")
)

# Missing values are not zero. Require a valid population
# and a non-negative cumulative death count.
country_data = country_data.loc[
    (country_data["population"] > 0)
    & (country_data["total_deaths"] >= 0)
].copy()

# Cumulative reported deaths per 100,000 people.
country_data["deaths_per_100k"] = (
    country_data["total_deaths"]
    / country_data["population"]
    * 100_000
)

if country_data.empty:
    raise ValueError(
        "No countries have usable death counts, population, "
        "and World Bank income classifications at the cutoff."
    )


# --------------------------------------------------
# 5. Compare median country-level death rates
# --------------------------------------------------
group_order = ["Low income", "High income"]

summary = (
    country_data
    .groupby("income_group")["deaths_per_100k"]
    .agg(median_rate="median", countries="count")
    .reindex(group_order)
)

if summary["median_rate"].isna().any():
    raise ValueError(
        "At least one income group has no eligible countries. "
        "Check the file and the income-group matches."
    )

print("\nMedian cumulative reported COVID-19 deaths")
print("per 100,000 people:")
print(summary.round(2).to_string())


# --------------------------------------------------
# 6. Plot the comparison
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    summary.index,
    summary["median_rate"]
)

ax.set_title(
    "COVID-19 death rates: low-income vs high-income countries\n"
    f"Latest available figures on or before {cutoff:%d %b %Y}"
)
ax.set_ylabel(
    "Median cumulative reported deaths per 100,000 people"
)
ax.set_xlabel("World Bank income group")

# Label each bar with its rate and number of countries.
for bar, group in zip(bars, group_order):
    rate = summary.loc[group, "median_rate"]
    n = int(summary.loc[group, "countries"])

    ax.annotate(
        f"{rate:,.1f}\n({n} countries)",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom"
    )

ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
ax.margins(y=0.15)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 7. Print the answer supported by this chart
# --------------------------------------------------
low_rate = summary.loc["Low income", "median_rate"]
high_rate = summary.loc["High income", "median_rate"]

if low_rate < high_rate:
    print(
        "\nThe median country-level reported COVID-19 death "
        "rate was LOWER in low-income countries."
    )
elif low_rate > high_rate:
    print(
        "\nThe median country-level reported COVID-19 death "
        "rate was HIGHER in low-income countries."
    )
else:
    print(
        "\nThe median country-level reported COVID-19 death "
        "rates were equal in the two income groups."
    )

print(
    "Interpret this as a comparison of reported deaths, "
    "not necessarily the true mortality burden."
)
