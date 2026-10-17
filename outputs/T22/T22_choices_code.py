import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load and prepare the data
# --------------------------------------------------

file_path = "owid-covid-data.csv"

df = pd.read_csv(file_path)

# Convert dates and case-rate values to usable types.
df["date"] = pd.to_datetime(df["date"], errors="coerce")

metric = "new_cases_smoothed_per_million"
df[metric] = pd.to_numeric(df[metric], errors="coerce")

# Define the analysis period.
start_date = pd.Timestamp("2020-03-01")
end_date = pd.Timestamp("2020-03-31")

# Keep March 2020 and exclude aggregate rows that have no
# continent assigned.
march = df.loc[
    df["date"].between(start_date, end_date)
    & df["continent"].notna(),
    ["location", "date", metric]
].copy()

# Missing case-rate values are not treated as zero.
march = march.dropna(subset=["location", "date", metric])

# Express dates as days since the beginning of March.
# This provides the x-values for estimating each trend.
march["day"] = (march["date"] - start_date).dt.days


# --------------------------------------------------
# 2. Estimate how quickly incidence increased
# --------------------------------------------------

MIN_VALID_DAYS = 20
MIN_DATE_SPAN = 25  # At least 26 calendar days from first to last observation

results = []

for location, group in march.groupby("location"):

    group = group.sort_values("day")

    # Require adequate coverage across most of March.
    if group["day"].nunique() < MIN_VALID_DAYS:
        continue

    date_span = group["day"].max() - group["day"].min()

    if date_span < MIN_DATE_SPAN:
        continue

    # Fit a linear trend:
    # daily cases per million = intercept + slope * day
    x = group["day"].to_numpy(dtype=float)
    y = group[metric].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    results.append({
        "location": location,
        "slope": slope,
        "valid_days": len(group)
    })

# Put the country-level estimates into a DataFrame.
ranking = pd.DataFrame(results)

if ranking.empty:
    raise ValueError(
        "No locations met the minimum data requirements. "
        "Check the date range, column names, and missing values."
    )

# Retain only locations with an increasing trend.
ranking = ranking.loc[ranking["slope"] > 0]

# Select the 15 largest positive slopes.
top_15 = (
    ranking.nlargest(15, "slope")
    .sort_values("slope")
)

if top_15.empty:
    raise ValueError(
        "No eligible locations had a positive estimated trend."
    )


# --------------------------------------------------
# 3. Plot the results
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(
    top_15["location"],
    top_15["slope"]
)

ax.set_title(
    "Countries and Territories Where COVID-19 Incidence "
    "Rose Fastest in March 2020"
)

ax.set_xlabel(
    "Increase in daily new cases per million people "
    "per calendar day"
)

ax.set_ylabel("Location")

ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 4. Print the numerical ranking
# --------------------------------------------------

top_15 = top_15.sort_values("slope", ascending=False)

print(
    "\nTop locations by the rate of increase in daily "
    "COVID-19 cases per million during March 2020:\n"
)

print(
    top_15[["location", "slope", "valid_days"]]
    .to_string(
        index=False,
        formatters={"slope": "{:.4f}".format}
    )
)