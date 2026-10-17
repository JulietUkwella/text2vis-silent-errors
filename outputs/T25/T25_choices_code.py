from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

FILE_PATH = Path("owid-covid-data.csv")
COVERAGE_THRESHOLD = 0.80  # Require at least 80% coverage

AGE_COL = "aged_65_older"
DEATH_COL = "total_deaths_per_million"

REQUIRED_COLS = {
    "iso_code",
    "continent",
    "location",
    "date",
    AGE_COL,
    DEATH_COL,
}


# --------------------------------------------------
# 2. Load and validate the data
# --------------------------------------------------

if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {FILE_PATH.resolve()}. "
        "Place the CSV in the current working directory "
        "or update FILE_PATH."
    )

df = pd.read_csv(FILE_PATH, low_memory=False)

missing_cols = REQUIRED_COLS.difference(df.columns)

if missing_cols:
    raise ValueError(
        f"The CSV is missing required columns: "
        f"{sorted(missing_cols)}"
    )

# Parse dates and ensure the two measures are numeric.
df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in [AGE_COL, DEATH_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# --------------------------------------------------
# 3. Keep country and territory rows
# --------------------------------------------------

# Aggregate rows, such as the world and continents,
# generally lack a country-level continent classification
# and use OWID_ codes.

country_rows = df.loc[
    df["continent"].notna()
    & df["iso_code"].notna()
    & ~df["iso_code"].astype(str).str.startswith("OWID_")
    & df["date"].notna()
].copy()


# --------------------------------------------------
# 4. Identify locations with usable data
# --------------------------------------------------

# A location is eligible if both measures exist on
# at least one date in the dataset.

complete = country_rows.dropna(
    subset=[AGE_COL, DEATH_COL]
)

eligible_count = complete["iso_code"].nunique()

if eligible_count < 2:
    raise ValueError(
        "Fewer than two countries or territories have "
        "both required measures. Check the input data."
    )

# Count locations with BOTH measures available on
# each date. Counting ISO codes avoids counting a
# location more than once on the same date.

coverage_by_date = (
    complete.groupby("date")["iso_code"]
    .nunique()
)

required_count = math.ceil(
    COVERAGE_THRESHOLD * eligible_count
)

# Choose the latest date meeting the coverage threshold.
suitable_dates = coverage_by_date[
    coverage_by_date >= required_count
]

if not suitable_dates.empty:
    reference_date = suitable_dates.index.max()

else:
    # If no date reaches 80% coverage, choose the date
    # with the greatest coverage. If several dates tie,
    # choose the latest and print a warning.

    max_coverage = coverage_by_date.max()

    reference_date = coverage_by_date[
        coverage_by_date == max_coverage
    ].index.max()

    print(
        "Warning: No date meets the requested "
        f"{COVERAGE_THRESHOLD:.0%} coverage threshold. "
        f"Using the latest date with maximum coverage "
        f"({max_coverage} of {eligible_count} eligible "
        "locations)."
    )


# --------------------------------------------------
# 5. Select one observation per location on that date
# --------------------------------------------------

plot_df = country_rows.loc[
    country_rows["date"] == reference_date
].dropna(
    subset=[AGE_COL, DEATH_COL]
).copy()

# The dataset is expected to have one row per location
# per date. Retain one row per ISO code as a safeguard
# against accidental duplicate country observations.

plot_df = plot_df.drop_duplicates(
    subset=["iso_code"],
    keep="last"
)

if len(plot_df) < 2:
    raise ValueError(
        "Fewer than two locations have both measures "
        "on the selected reference date."
    )

# Shorter variable names for plotting.
x = plot_df[AGE_COL]
y = plot_df[DEATH_COL]

# Pearson correlation measures the strength and direction
# of the linear association.
pearson_r = x.corr(y, method="pearson")


# --------------------------------------------------
# 6. Create the scatter plot and trend line
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    x,
    y,
    alpha=0.7,
    edgecolors="white",
    linewidths=0.4
)

# Add an ordinary least-squares linear trend line.
# pandas covariance / variance gives the fitted slope.

if x.nunique() > 1:
    slope = x.cov(y) / x.var()
    intercept = y.mean() - slope * x.mean()

    x_line = [x.min(), x.max()]
    y_line = [
        slope * x_line[0] + intercept,
        slope * x_line[1] + intercept,
    ]

    ax.plot(
        x_line,
        y_line,
        linewidth=2,
        label="Linear trend"
    )

    ax.legend()

# Report key details directly on the chart.
r_text = (
    f"{pearson_r:.2f}"
    if pd.notna(pearson_r)
    else "undefined"
)

annotation = (
    f"Reference date: {reference_date:%Y-%m-%d}\n"
    f"Locations shown: {len(plot_df)}\n"
    f"Pearson correlation (r): {r_text}"
)

ax.text(
    0.03,
    0.97,
    annotation,
    transform=ax.transAxes,
    ha="left",
    va="top",
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.85,
    }
)

ax.set_title(
    "Population aged 65+ and cumulative COVID-19 deaths"
)

ax.set_xlabel(
    "Population aged 65 and older (%)"
)

ax.set_ylabel(
    "Cumulative reported COVID-19 deaths per million"
)

ax.grid(True, alpha=0.25)

fig.tight_layout()

# Save a high-resolution copy and display it.
fig.savefig(
    "covid_deaths_vs_older_share.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 7. Print the analysis details
# --------------------------------------------------

print(f"Reference date: {reference_date:%Y-%m-%d}")
print(f"Eligible locations overall: {eligible_count}")
print(f"Locations included in the chart: {len(plot_df)}")
print(f"Pearson correlation: {r_text}")
print(
    "Saved chart to: "
    f"{Path('covid_deaths_vs_older_share.png').resolve()}"
)