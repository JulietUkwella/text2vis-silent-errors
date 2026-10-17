import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv", parse_dates=["date"])

# Filter for countries in Africa
africa = df[df["continent"] == "Africa"].copy()

# Keep records with available vaccination and population data
africa = africa.dropna(
    subset=["people_fully_vaccinated", "population"]
)

# Aggregate fully vaccinated people and population by date
daily = (
    africa.groupby("date")
    .agg(
        fully_vaccinated=("people_fully_vaccinated", "sum"),
        population=("population", "sum")
    )
    .sort_index()
)

# Calculate the percentage fully vaccinated
daily["fully_vaccinated_share"] = (
    daily["fully_vaccinated"] / daily["population"] * 100
)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    daily.index,
    daily["fully_vaccinated_share"],
    linewidth=2,
    label="Fully vaccinated"
)

ax.set_title(
    "Share of the Population Fully Vaccinated Against COVID-19 in Africa"
)
ax.set_xlabel("Date")
ax.set_ylabel("Population fully vaccinated (%)")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Print the latest available estimate
latest = daily.dropna(subset=["fully_vaccinated_share"]).iloc[-1]

print(f"Latest date: {latest.name.strftime('%Y-%m-%d')}")
print(
    f"Share fully vaccinated: "
    f"{latest['fully_vaccinated_share']:.2f}%"
)
