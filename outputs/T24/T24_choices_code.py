
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv")

# 2. Convert dates and ensure the case measure is numeric
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["new_cases_smoothed"] = pd.to_numeric(
    df["new_cases_smoothed"], errors="coerce"
)

# 3. Select only United States observations in calendar year 2023
start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp("2023-12-31")

us = df.loc[
    (df["location"] == "United States")
    & (df["iso_code"] == "USA")
    & (df["date"] >= start_date)
    & (df["date"] <= end_date)
].copy()

# Check that the selection contains data
if us.empty:
    raise ValueError(
        "No United States records were found for 2023. "
        "Check the file, date values, and location labels."
    )

# 4. Check that there is only one row per date
if us["date"].duplicated().any():
    duplicate_dates = us.loc[
        us["date"].duplicated(keep=False), "date"
    ].dt.strftime("%Y-%m-%d").unique()

    raise ValueError(
        f"Duplicate United States dates found: {duplicate_dates}. "
        "Inspect these rows before plotting."
    )

# 5. Create a daily time series, sorted chronologically
cases = (
    us.set_index("date")["new_cases_smoothed"]
    .sort_index()
)

# Include every date in 2023 without filling missing values with zero
all_dates = pd.date_range(
    start=start_date,
    end=end_date,
    freq="D"
)
cases = cases.reindex(all_dates)

if cases.notna().sum() == 0:
    raise ValueError(
        "The new_cases_smoothed column has no usable observations "
        "for the United States in 2023."
    )

# 6. Plot the trend
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    cases.index,
    cases.values,
    linewidth=1.8,
    label="Seven-day smoothed new cases"
)

ax.set_title(
    "United States: Daily New COVID-19 Cases in 2023",
    fontsize=15
)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel(
    "New reported cases per day (7-day smoothed)",
    fontsize=12
)

# Show monthly ticks
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

ax.set_xlim(start_date, end_date)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()

plt.show()
