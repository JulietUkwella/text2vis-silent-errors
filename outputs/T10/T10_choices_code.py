import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
file_path = "owid-covid-data.csv"

df = pd.read_csv(file_path)

# Parse dates and convert death counts to numeric values.
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["new_deaths"] = pd.to_numeric(
    df["new_deaths"], errors="coerce"
)
df["new_deaths_smoothed"] = pd.to_numeric(
    df["new_deaths_smoothed"], errors="coerce"
)

# --------------------------------------------------
# 2. Select the national-level United States records
# --------------------------------------------------
us = df.loc[
    df["location"].eq("United States")
].copy()

if us.empty:
    raise ValueError(
        "No rows found for location = 'United States'. "
        "Check the location names in the CSV."
    )

# Exclude records that cannot be used to identify a peak.
us = us.dropna(subset=["date", "new_deaths"])

# The dataset is expected to have one row per location/date.
# Stop rather than silently double-counting unexpected duplicates.
if us["date"].duplicated().any():
    duplicates = us.loc[
        us["date"].duplicated(keep=False), "date"
    ].sort_values()

    raise ValueError(
        "Duplicate United States dates found. "
        f"Investigate these dates before proceeding:\n"
        f"{duplicates.to_list()}"
    )

# Sort chronologically for plotting.
us = us.sort_values("date").reset_index(drop=True)

if us.empty:
    raise ValueError(
        "No valid U.S. daily death counts are available."
    )

# --------------------------------------------------
# 3. Identify the maximum single-day reported death count
# --------------------------------------------------
max_deaths = us["new_deaths"].max()

# Keep all dates in case multiple days share the same maximum.
peak_rows = us.loc[
    us["new_deaths"].eq(max_deaths)
]

# Print the result.
print("Highest daily reported COVID-19 deaths in the U.S.")
print(f"Reported deaths: {max_deaths:,.0f}")
print("Peak date(s):")

for date in peak_rows["date"]:
    print(date.strftime("%Y-%m-%d"))

# --------------------------------------------------
# 4. Plot daily deaths and the smoothed trend
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))

# Raw daily reported deaths: this series determines the answer.
ax.plot(
    us["date"],
    us["new_deaths"],
    linewidth=0.8,
    alpha=0.55,
    label="Daily reported deaths"
)

# Seven-day smoothed series: context only.
ax.plot(
    us["date"],
    us["new_deaths_smoothed"],
    linewidth=2,
    label="Seven-day smoothed deaths"
)

# Highlight every date tied for the maximum raw daily count.
ax.scatter(
    peak_rows["date"],
    peak_rows["new_deaths"],
    color="red",
    s=80,
    zorder=5,
    label="Highest daily count"
)

# Annotate the first peak; all tied dates are printed above.
peak = peak_rows.iloc[0]

ax.annotate(
    f"Peak: {peak['date']:%Y-%m-%d}\n"
    f"{max_deaths:,.0f} reported deaths",
    xy=(peak["date"], peak["new_deaths"]),
    xytext=(12, 20),
    textcoords="offset points",
    arrowprops={"arrowstyle": "->"},
    fontsize=10
)

ax.set_title(
    "United States: Daily Reported COVID-19 Deaths"
)
ax.set_xlabel("Date")
ax.set_ylabel("Reported deaths per day")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()