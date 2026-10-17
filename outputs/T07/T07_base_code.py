import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("owid-covid-data.csv", parse_dates=["date"])

# Filter for the United States
us = df[df["location"] == "United States"].copy()

# Ensure deaths are numeric
us["new_deaths"] = pd.to_numeric(us["new_deaths"], errors="coerce")

# Keep only 2021 and 2022
us = us[us["date"].dt.year.isin([2021, 2022])].copy()
us["year"] = us["date"].dt.year

# Calculate total reported deaths for each year
annual_deaths = (
    us.groupby("year")["new_deaths"]
    .sum(min_count=1)
    .reindex([2021, 2022])
)

# Print the totals and identify the year with more deaths
print("Reported COVID-19 deaths in the United States:")
for year, deaths in annual_deaths.items():
    print(f"{year}: {deaths:,.0f}")

if annual_deaths.notna().all():
    higher_year = annual_deaths.idxmax()
    print(f"\nThere were more reported COVID-19 deaths in {higher_year}.")

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    annual_deaths.index.astype(str),
    annual_deaths.values,
    color=["steelblue", "orange"],
    width=0.6
)

# Add labels above the bars
ax.bar_label(bars, labels=[
    f"{value:,.0f}" if pd.notna(value) else "No data"
    for value in annual_deaths.values
], padding=4)

ax.set_title("United States: COVID-19 Deaths in 2021 vs. 2022")
ax.set_xlabel("Year")
ax.set_ylabel("Total reported new deaths")
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()