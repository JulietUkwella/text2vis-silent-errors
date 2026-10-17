import pandas as pd
import matplotlib.pyplot as plt

# Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Keep rows with a known continent and recorded total deaths
df = df.dropna(subset=["continent", "total_deaths"])

# Get the latest recorded cumulative deaths for each continent
latest_deaths = (
    df.sort_values("date")
      .groupby("continent")
      .tail(1)
      .set_index("continent")["total_deaths"]
      .sort_values(ascending=False)
)

# Create a bar chart
plt.figure(figsize=(10, 6))
latest_deaths.plot(kind="bar", color="steelblue", edgecolor="black")

plt.title("Total Reported COVID-19 Deaths by Continent")
plt.xlabel("Continent")
plt.ylabel("Reported cumulative deaths")
plt.xticks(rotation=45, ha="right")
plt.ticklabel_format(axis="y", style="plain")
plt.tight_layout()

# Display the chart
plt.show()