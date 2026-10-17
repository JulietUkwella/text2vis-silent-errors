import pandas as pd
import matplotlib.pyplot as plt

# Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Select the four countries
countries = ["France", "Germany", "Italy", "Spain"]
df = df[df["location"].isin(countries)]

# Plot total COVID-19 deaths per million over time
plt.figure(figsize=(12, 6))

for country in countries:
    country_data = df[df["location"] == country].sort_values("date")
    plt.plot(
        country_data["date"],
        country_data["total_deaths_per_million"],
        label=country
    )

# Add chart labels and formatting
plt.title("Total COVID-19 Deaths per Million Over Time")
plt.xlabel("Date")
plt.ylabel("Total Deaths per Million People")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Display the chart
plt.show()