import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
df = pd.read_csv(
    "owid-covid-data.csv",
    parse_dates=["date"]
)

# Make sure the variables used in the plot are numeric.
for col in ["median_age", "total_deaths_per_million"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------------------------------
# 2. Keep country-level rows
# --------------------------------------------------
# Standard three-letter ISO codes identify country rows.
# OWID_KOS is included as an exception for Kosovo.
iso = df["iso_code"].astype("string")

is_country = iso.str.fullmatch(r"[A-Z]{3}", na=False)
is_kosovo = iso.eq("OWID_KOS")

countries = df[is_country | is_kosovo].copy()

# --------------------------------------------------
# 3. Remove rows missing the required measurements
# --------------------------------------------------
usable = countries.dropna(
    subset=[
        "date",
        "median_age",
        "total_deaths_per_million"
    ]
)

# --------------------------------------------------
# 4. Keep the latest usable observation per country
# --------------------------------------------------
latest = (
    usable
    .sort_values("date")
    .groupby("iso_code", group_keys=False)
    .tail(1)
    .copy()
)

# Sort for a consistent output.
latest = latest.sort_values("location")

# Check that there is only one point per country.
assert latest["iso_code"].is_unique

print(f"Number of countries plotted: {len(latest)}")
print(f"Latest observation dates range from "
      f"{latest['date'].min().date()} to "
      f"{latest['date'].max().date()}")

# --------------------------------------------------
# 5. Create the scatter plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    latest["median_age"],
    latest["total_deaths_per_million"],
    alpha=0.7,
    s=45,
    edgecolors="white",
    linewidths=0.5
)

ax.set_title(
    "Median Age vs. Total COVID-19 Deaths per Million by Country"
)

ax.set_xlabel("Median age (years)")
ax.set_ylabel("Total COVID-19 deaths per million people")

ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()