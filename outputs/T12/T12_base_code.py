import pandas as pd
import matplotlib.pyplot as plt

# Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Exclude aggregate regions and keep rows with vaccination data
countries = df[
    (~df["iso_code"].str.startswith("OWID_", na=False)) &
    (df["people_fully_vaccinated"].notna())
].copy()

# Get the latest available vaccination figure for each country
latest = (
    countries.sort_values("date")
    .groupby("location", as_index=False)
    .tail(1)
)

# Select the 10 countries with the most fully vaccinated people
top10 = latest.nlargest(10, "people_fully_vaccinated")

# Sort for a horizontal bar chart
top10 = top10.sort_values("people_fully_vaccinated")

# Plot the chart
plt.figure(figsize=(10, 6))
plt.barh(
    top10["location"],
    top10["people_fully_vaccinated"] / 1_000_000
)

plt.xlabel("People fully vaccinated (millions)")
plt.ylabel("Country")
plt.title("Top 10 Countries by Number of Fully Vaccinated People")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()