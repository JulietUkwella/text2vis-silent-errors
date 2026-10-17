import pandas as pd
import matplotlib.pyplot as plt
import requests

# 1. Load the Our World in Data COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv", low_memory=False)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["total_deaths_per_million"] = pd.to_numeric(
    df["total_deaths_per_million"], errors="coerce"
)

# 2. Retrieve World Bank country income classifications
url = "https://api.worldbank.org/v2/country?format=json&per_page=400"

response = requests.get(url, timeout=30)
response.raise_for_status()
payload = response.json()

if len(payload) < 2 or not isinstance(payload[1], list):
    raise ValueError("Could not retrieve World Bank country classifications.")

countries = payload[1]

income_records = []

for country in countries:
    income_code = country.get("incomeLevel", {}).get("id")
    iso_code = country.get("id")

    # Compare only low-income and high-income countries
    if income_code in ["LIC", "HIC"] and iso_code:
        income_records.append({
            "iso_code": iso_code,
            "income_group": {
                "LIC": "Low income",
                "HIC": "High income"
            }[income_code]
        })

income_df = pd.DataFrame(income_records).drop_duplicates(
    subset="iso_code"
)

# 3. Get each country's latest available cumulative death rate
df = df.dropna(subset=["iso_code", "date",
                       "total_deaths_per_million"])

df = df.sort_values("date")

latest = (
    df.groupby("iso_code", group_keys=False)
      .tail(1)
      .copy()
)

# 4. Match countries to their World Bank income groups
comparison = latest.merge(
    income_df,
    on="iso_code",
    how="inner"
)

# Middle-income countries and unmatched locations are excluded.
comparison = comparison[
    comparison["income_group"].isin(["Low income", "High income"])
]

# 5. Calculate the median deaths per million for each group
order = ["Low income", "High income"]

summary = (
    comparison.groupby("income_group")["total_deaths_per_million"]
    .agg(median="median", countries="count")
    .reindex(order)
)

print("Median cumulative reported COVID-19 deaths per million:")
print(summary)

# 6. Plot the results
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    order,
    summary["median"].values
)

ax.set_title(
    "COVID-19 Death Rates: Low-Income vs High-Income Countries"
)
ax.set_ylabel("Median cumulative deaths per million people")
ax.set_xlabel("World Bank income group")

# Display the median and number of countries above each bar
for bar, group in zip(bars, order):
    value = summary.loc[group, "median"]
    count = summary.loc[group, "countries"]

    if pd.notna(value):
        ax.annotate(
            f"{value:,.0f}\n({int(count)} countries)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom"
        )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

# 7. Print the conclusion based on the chart
low = summary.loc["Low income", "median"]
high = summary.loc["High income", "median"]

if pd.notna(low) and pd.notna(high):
    if low < high:
        print(
            "\nConclusion: Low-income countries had a lower "
            "median reported cumulative COVID-19 death rate."
        )
    elif low > high:
        print(
            "\nConclusion: Low-income countries had a higher "
            "median reported cumulative COVID-19 death rate."
        )
    else:
        print(
            "\nConclusion: Both income groups had the same "
            "median reported cumulative COVID-19 death rate."
        )
else:
    print("\nInsufficient data to compare both income groups.")