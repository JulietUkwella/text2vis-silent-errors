import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
FILE = "owid-covid-data.csv"
METRIC = "excess_mortality_cumulative"
TOP_N = 15

df = pd.read_csv(FILE, low_memory=False)

# Check that the required columns are present.
required_columns = [
    "iso_code",
    "continent",
    "location",
    "date",
    METRIC,
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Parse dates and convert the chosen measure to numeric.
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")


# --------------------------------------------------
# 2. Keep country and territory rows, not aggregates
# --------------------------------------------------
# OWID aggregate codes generally do not have a standard
# three-letter ISO code or a continent value.
#
# This retains locations with a three-letter code and
# a non-missing continent. It can include territories
# and special jurisdictions, not only sovereign states.

is_country_or_territory = (
    df["continent"].notna()
    & df["iso_code"].astype("string").str.fullmatch(
        r"[A-Z]{3}", na=False
    )
)

country_data = df.loc[is_country_or_territory].copy()


# --------------------------------------------------
# 3. Remove rows without a usable measurement
# --------------------------------------------------
valid_data = country_data.dropna(
    subset=["date", METRIC]
).copy()

if valid_data.empty:
    raise ValueError(
        f"No valid observations found for {METRIC}."
    )


# --------------------------------------------------
# 4. Select each location's latest non-missing value
# --------------------------------------------------
# Sorting by date ensures that the final row for each
# ISO code contains its latest available measurement.

latest = (
    valid_data
    .sort_values(["iso_code", "date"])
    .groupby("iso_code", group_keys=False)
    .tail(1)
    .copy()
)


# --------------------------------------------------
# 5. Rank locations and select the top 15
# --------------------------------------------------
top = (
    latest
    .nlargest(TOP_N, METRIC)
    .sort_values(METRIC, ascending=True)
    .copy()
)

# Include the observation date to expose differences
# in reporting coverage between locations.
top["chart_label"] = (
    top["location"]
    + " (as of "
    + top["date"].dt.strftime("%Y-%m-%d")
    + ")"
)


# --------------------------------------------------
# 6. Create a horizontal bar chart
# --------------------------------------------------
fig, ax = plt.subplots(
    figsize=(12, max(6, 0.48 * len(top)))
)

bars = ax.barh(
    top["chart_label"],
    top[METRIC],
)

# A zero reference line distinguishes positive and
# negative cumulative excess mortality.
ax.axvline(0, linewidth=0.8)

ax.set_title(
    "Countries and territories with the highest "
    "cumulative excess mortality",
    fontsize=14,
    pad=24,
)

ax.set_xlabel(
    "Cumulative excess mortality (% above expected deaths)"
)

ax.set_ylabel("")

# The column stores percentage values as numbers such as
# 20 for 20%, rather than decimals such as 0.20.
ax.xaxis.set_major_formatter(
    PercentFormatter(xmax=100)
)

# Add a percentage label to the end of each bar.
largest_value = max(
    1,
    top[METRIC].abs().max()
)
label_offset = largest_value * 0.015

for bar, value in zip(bars, top[METRIC]):
    y = bar.get_y() + bar.get_height() / 2

    if value >= 0:
        x = value + label_offset
        alignment = "left"
    else:
        x = value - label_offset
        alignment = "right"

    ax.text(
        x,
        y,
        f"{value:.1f}%",
        va="center",
        ha=alignment,
        fontsize=9,
    )

ax.text(
    0,
    1.01,
    "Cumulative since 1 Jan 2020; latest available "
    "non-missing observation for each location",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
)

ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
ax.margins(x=0.20)

plt.tight_layout()
plt.show()