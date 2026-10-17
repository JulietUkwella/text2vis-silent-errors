
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Our World in Data COVID-19 dataset.
df = pd.read_csv(
    "owid-covid-data.csv",
    parse_dates=["date"]
)

# 2. Keep only observations for Italy.
italy = (
    df.loc[df["location"].eq("Italy"), ["date", "new_deaths"]]
    .sort_values("date")
    .copy()
)

if italy.empty:
    raise ValueError("No rows were found for Italy in the dataset.")

# 3. Check whether there is more than one row per date.
duplicate_dates = italy.loc[
    italy["date"].duplicated(keep=False), "date"
].drop_duplicates()

if not duplicate_dates.empty:
    print("Warning: duplicate dates found for Italy:")
    print(duplicate_dates.to_list())
    print("Investigate these rows before interpreting the chart.")

# 4. Report missing values and retain them as missing.
missing_count = italy["new_deaths"].isna().sum()
print(f"Number of Italy date rows: {len(italy)}")
print(f"Missing daily death counts: {missing_count}")

# 5. Compute a seven-calendar-day rolling average.
# min_periods=7 means the average is shown only when all
# seven calendar-day observations are available and non-missing.
italy["new_deaths_7day_avg"] = (
    italy.set_index("date")["new_deaths"]
    .rolling("7D", min_periods=7)
    .mean()
    .to_numpy()
)

# 6. Plot daily reported deaths and the rolling average.
fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    italy["date"],
    italy["new_deaths"],
    color="steelblue",
    linewidth=0.8,
    alpha=0.5,
    label="Daily reported deaths"
)

ax.plot(
    italy["date"],
    italy["new_deaths_7day_avg"],
    color="darkred",
    linewidth=2,
    label="7-day rolling average"
)

ax.axhline(0, color="black", linewidth=0.7)
ax.set_title("Daily Reported COVID-19 Deaths in Italy")
ax.set_xlabel("Date")
ax.set_ylabel("New deaths reported")
ax.grid(True, alpha=0.25)
ax.legend()
fig.tight_layout()

plt.show()
