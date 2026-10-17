import pandas as pd
import matplotlib.pyplot as plt

# Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# Filter the data for Italy
italy = df[df["location"] == "Italy"].copy()

# Convert dates to datetime format and sort chronologically
italy["date"] = pd.to_datetime(italy["date"])
italy = italy.sort_values("date")

# Plot total COVID-19 deaths over time
plt.figure(figsize=(12, 6))
plt.plot(italy["date"], italy["total_deaths"], color="red", linewidth=2)

# Add chart labels and title
plt.title("Total COVID-19 Deaths Over Time in Italy")
plt.xlabel("Date")
plt.ylabel("Total Deaths")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Display the chart
plt.show()
