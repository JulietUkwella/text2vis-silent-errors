import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv", low_memory=False)

# 2. Convert the relevant columns to the correct data types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["new_cases_smoothed"] = pd.to_numeric(
    df["new_cases_smoothed"], errors="coerce"
)

# 3. Select Germany's observations and the required columns
germany = df.loc[
    df["location"].eq("Germany"),
    ["date", "new_cases_smoothed"]
].dropna(subset=["date"])

# 4. Filter to the calendar year 2021 and sort chronologically
germany_2021 = (
    germany.loc[
        germany["date"].between("2021-01-01", "2021-12-31")
    ]
    .sort_values("date")
    .set_index("date")
)

# Check that the dataset contains usable observations for this period
if germany_2021.empty:
    raise ValueError(
        "No observations for Germany were found during 2021."
    )

if germany_2021["new_cases_smoothed"].notna().sum() == 0:
    raise ValueError(
        "No valid new_cases_smoothed values were found for Germany in 2021."
    )

# 5. Include every calendar day so missing dates appear as gaps
all_dates = pd.date_range(
    start="2021-01-01",
    end="2021-12-31",
    freq="D"
)

germany_2021 = germany_2021.reindex(all_dates)
germany_2021.index.name = "date"

# 6. Create the line chart
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    germany_2021.index,
    germany_2021["new_cases_smoothed"],
    linewidth=2,
    label="7-day average"
)

# 7. Add chart title, axis labels, and formatting
ax.set_title(
    "7-Day Average of New COVID-19 Cases in Germany (2021)",
    fontsize=14
)
ax.set_xlabel("Date")
ax.set_ylabel("Average new reported cases per day")

# Display one tick per month
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

ax.set_xlim(
    pd.Timestamp("2021-01-01"),
    pd.Timestamp("2021-12-31")
)

ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

# 8. Display the chart
plt.show()