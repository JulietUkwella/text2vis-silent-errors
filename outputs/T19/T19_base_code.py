import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv", parse_dates=["date"])

# 2. Keep individual countries and exclude regional aggregates
countries = df[
    df["continent"].notna()
    & ~df["iso_code"].str.startswith("OWID_", na=False)
].copy()

# 3. Keep rows with both GDP and COVID-19 death data
countries = countries.dropna(
    subset=["gdp_per_capita", "total_deaths_per_million"]
)

# Exclude invalid GDP values
countries = countries[countries["gdp_per_capita"] > 0]

# 4. Select the latest available observation for each country
latest = (
    countries.sort_values("date")
    .groupby(["iso_code", "location"], group_keys=False)
    .tail(1)
    .copy()
)

# 5. Calculate the correlation between log GDP and deaths per million
log_gdp = np.log10(latest["gdp_per_capita"])
correlation = log_gdp.corr(latest["total_deaths_per_million"])

# 6. Create the scatter plot
plt.figure(figsize=(11, 7))

plt.scatter(
    latest["gdp_per_capita"],
    latest["total_deaths_per_million"],
    alpha=0.7,
    s=45
)

# Use a logarithmic x-axis because GDP varies substantially
# between countries
plt.xscale("log")

plt.xlabel("GDP per capita (international dollars)")
plt.ylabel("Cumulative COVID-19 deaths per million people")
plt.title("Did Richer Countries Handle COVID-19 Better?")

plt.text(
    0.05,
    0.95,
    f"Correlation (log GDP vs. deaths per million): {correlation:.2f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()