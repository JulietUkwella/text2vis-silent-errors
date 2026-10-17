import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# 2. Convert the date column to datetime format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 3. Select countries in South America
south_america = df[df["continent"] == "South America"].copy()

# 4. Sort by date and remove rows with missing death-rate data
south_america = south_america.sort_values("date")
south_america = south_america.dropna(
    subset=["total_deaths_per_million"]
)

# 5. Get the latest available cumulative deaths per million
latest = south_america.groupby("location").tail(1)

# 6. Sort countries from lowest to highest death rate
latest = latest.sort_values("total_deaths_per_million")

# 7. Create the horizontal bar chart
plt.figure(figsize=(12, 8))

plt.barh(
    latest["location"],
    latest["total_deaths_per_million"]
)

plt.title(
    "COVID-19 Impact Across South American Countries",
    fontsize=16
)
plt.xlabel("Cumulative COVID-19 Deaths per Million People", fontsize=12)
plt.ylabel("Country", fontsize=12)

plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()

# 8. Display the chart
plt.show()
