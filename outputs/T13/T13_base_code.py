import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the COVID-19 dataset
df = pd.read_csv(
    "owid-covid-data.csv",
    parse_dates=["date"]
)

# 2. Filter data to European countries
europe = df[df["continent"] == "Europe"].copy()

# 3. Keep records with reported full vaccination rates
europe = europe.dropna(
    subset=["people_fully_vaccinated_per_hundred"]
)

# 4. Get the latest available vaccination rate for each country
latest = (
    europe.sort_values("date")
    .groupby("location", as_index=False)
    .tail(1)
    .sort_values("people_fully_vaccinated_per_hundred")
)

# 5. Create a horizontal bar chart
plt.figure(figsize=(12, max(8, len(latest) * 0.3)))

plt.barh(
    latest["location"],
    latest["people_fully_vaccinated_per_hundred"]
)

plt.xlabel("People fully vaccinated (% of population)")
plt.ylabel("Country")
plt.title("COVID-19 Full Vaccination Rates Across European Countries")

plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()