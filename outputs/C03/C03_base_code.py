import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("owid-covid-data.csv")

# Keep country rows only, excluding regional and income-group aggregates.
# OWID aggregate rows generally have ISO codes beginning with OWID_.
countries = df[~df["iso_code"].str.startswith("OWID_", na=False)].copy()

# Select the latest available record for each country
countries["date"] = pd.to_datetime(countries["date"])
latest = (
    countries.sort_values("date")
    .groupby("location", as_index=False)
    .last()
)

# Keep rows with valid values for both variables
plot_data = latest.dropna(subset=["median_age", "total_deaths_per_million"])

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(
    plot_data["median_age"],
    plot_data["total_deaths_per_million"],
    alpha=0.7
)

plt.title("Median Age vs. Total COVID-19 Deaths per Million")
plt.xlabel("Median Age (years)")
plt.ylabel("Total COVID-19 Deaths per Million")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()