import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("owid-covid-data.csv")

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

# Filter data for the United States
us = df[df["location"] == "United States"].copy()

# Remove rows where daily deaths are missing
us = us.dropna(subset=["new_deaths"])

# Find the day with the highest reported daily deaths
peak_idx = us["new_deaths"].idxmax()
peak_day = us.loc[peak_idx, "date"]
peak_deaths = us.loc[peak_idx, "new_deaths"]

# Print the answer
print(f"Peak day: {peak_day.strftime('%Y-%m-%d')}")
print(f"Reported new deaths: {peak_deaths:,.0f}")

# Plot daily COVID-19 deaths
plt.figure(figsize=(12, 6))
plt.plot(us["date"], us["new_deaths"], linewidth=1, label="Daily reported deaths")

# Highlight the peak
plt.scatter(peak_day, peak_deaths, color="red", zorder=3, label="Peak day")
plt.annotate(
    f"{peak_day.strftime('%Y-%m-%d')}\n{peak_deaths:,.0f} deaths",
    xy=(peak_day, peak_deaths),
    xytext=(10, -35),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)

plt.title("Daily Reported COVID-19 Deaths in the United States")
plt.xlabel("Date")
plt.ylabel("New deaths")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()