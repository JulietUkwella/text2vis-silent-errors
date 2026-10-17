import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the dataset
df = pd.read_csv("owid-covid-data.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Filter data for the United Kingdom
uk = df[df["location"] == "United Kingdom"].copy()
uk = uk.sort_values("date")

# Calculate the 7-day rolling average of daily deaths
uk["deaths_7day_avg"] = uk["new_deaths"].rolling(
    window=7, min_periods=1
).mean()

# Use vaccination coverage (% of population fully vaccinated)
uk["vaccination_pct"] = uk["people_fully_vaccinated_per_hundred"]

# Create the chart
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot daily deaths on the first y-axis
ax1.plot(
    uk["date"],
    uk["deaths_7day_avg"],
    color="firebrick",
    linewidth=2,
    label="Daily deaths (7-day average)"
)
ax1.set_xlabel("Date")
ax1.set_ylabel("Daily deaths (7-day average)", color="firebrick")
ax1.tick_params(axis="y", labelcolor="firebrick")
ax1.grid(True, alpha=0.3)

# Plot vaccination coverage on the second y-axis
ax2 = ax1.twinx()
ax2.plot(
    uk["date"],
    uk["vaccination_pct"],
    color="royalblue",
    linewidth=2,
    label="Fully vaccinated population (%)"
)
ax2.set_ylabel("Population fully vaccinated (%)", color="royalblue")
ax2.tick_params(axis="y", labelcolor="royalblue")
ax2.set_ylim(0, 100)

# Format the x-axis
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title(
    "COVID-19 Deaths and Vaccination Coverage in the United Kingdom"
)
fig.tight_layout()
plt.show()
