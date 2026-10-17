import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CSV_PATH = "owid-covid-data.csv"

# None uses the latest date in the European rows of the CSV.
# For a fixed historical snapshot, enter a date such as
# "2022-12-31", provided that date falls within the data.
REFERENCE_DATE = None

# Exclude a country if its latest non-missing death
# observation is more than this many days before the
# reference date.
MAX_DEATH_DATA_AGE_DAYS = 90

OUTPUT_FILE = "europe_hospital_beds_vs_covid_deaths.png"


# --------------------------------------------------
# 2. Load and validate the data
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

required_columns = {
    "continent",
    "location",
    "date",
    "hospital_beds_per_thousand",
    "total_deaths_per_million",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"The CSV is missing these required columns: "
        f"{sorted(missing_columns)}"
    )

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Ensure the two analysis variables are numeric.
for column in [
    "hospital_beds_per_thousand",
    "total_deaths_per_million",
]:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# --------------------------------------------------
# 3. Select European locations and the reference date
# --------------------------------------------------

europe = df.loc[
    df["continent"].eq("Europe") & df["date"].notna()
].copy()

if europe.empty:
    raise ValueError(
        "No rows were found with continent == 'Europe'."
    )

earliest_date = europe["date"].min()
latest_date = europe["date"].max()

if REFERENCE_DATE is None:
    reference_date = latest_date
else:
    reference_date = pd.Timestamp(REFERENCE_DATE)

    if not earliest_date <= reference_date <= latest_date:
        raise ValueError(
            f"REFERENCE_DATE must be between "
            f"{earliest_date.date()} and {latest_date.date()}."
        )

# Do not use observations recorded after the reference date.
europe = europe.loc[
    europe["date"] <= reference_date
].copy()


# --------------------------------------------------
# 4. Extract the latest available hospital bed value
#    for each location
# --------------------------------------------------

bed_rows = europe.loc[
    europe["hospital_beds_per_thousand"].notna(),
    [
        "location",
        "date",
        "hospital_beds_per_thousand",
    ],
].copy()

beds = (
    bed_rows
    .sort_values(["location", "date"])
    .drop_duplicates(subset="location", keep="last")
    [["location", "hospital_beds_per_thousand"]]
)


# --------------------------------------------------
# 5. Extract the latest available death rate for each
#    location, then enforce the 90-day freshness limit
# --------------------------------------------------

death_rows = europe.loc[
    europe["total_deaths_per_million"].notna(),
    [
        "location",
        "date",
        "total_deaths_per_million",
    ],
].copy()

deaths = (
    death_rows
    .sort_values(["location", "date"])
    .drop_duplicates(subset="location", keep="last")
    .rename(columns={"date": "deaths_date"})
)

# How long before the reference date was each death rate observed?
deaths["days_before_reference"] = (
    reference_date - deaths["deaths_date"]
).dt.days

# Retain only reasonably recent death observations.
deaths = deaths.loc[
    deaths["days_before_reference"].between(
        0, MAX_DEATH_DATA_AGE_DAYS
    )
].copy()


# --------------------------------------------------
# 6. Combine the measures
# --------------------------------------------------

# An inner join retains locations with both required measures.
plot_data = beds.merge(
    deaths[
        [
            "location",
            "total_deaths_per_million",
            "deaths_date",
            "days_before_reference",
        ]
    ],
    on="location",
    how="inner",
    validate="one_to_one",
)

# Remove non-finite and physically invalid values.
plot_data = plot_data.replace(
    [np.inf, -np.inf], np.nan
)

plot_data = plot_data.dropna(
    subset=[
        "hospital_beds_per_thousand",
        "total_deaths_per_million",
    ]
)

plot_data = plot_data.loc[
    (plot_data["hospital_beds_per_thousand"] >= 0)
    & (plot_data["total_deaths_per_million"] >= 0)
].copy()

plot_data = plot_data.sort_values("location").reset_index(drop=True)


# --------------------------------------------------
# 7. Report the data included and excluded
# --------------------------------------------------

all_locations = set(
    europe["location"].dropna().unique()
)

included_locations = set(plot_data["location"])

excluded_locations = sorted(
    all_locations - included_locations
)

print(f"Reference date: {reference_date.date()}")
print(f"Countries/locations plotted: {len(plot_data)}")

if not plot_data.empty:
    print(
        "Dates of selected death observations: "
        f"{plot_data['deaths_date'].min().date()} to "
        f"{plot_data['deaths_date'].max().date()}"
    )

if excluded_locations:
    print(
        "\nLocations excluded because a required measure "
        "was missing or the death observation was too old:"
    )
    print(", ".join(excluded_locations))

if plot_data.empty:
    raise ValueError(
        "No locations have both required measures within "
        "the selected observation window. Check the reference "
        "date or increase MAX_DEATH_DATA_AGE_DAYS."
    )


# --------------------------------------------------
# 8. Create the scatter plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 8))

x = plot_data["hospital_beds_per_thousand"].to_numpy()
y = plot_data["total_deaths_per_million"].to_numpy()

ax.scatter(
    x,
    y,
    s=55,
    alpha=0.8,
    edgecolors="white",
    linewidths=0.6,
)

# Label every point with its location.
# Small alternating offsets help reduce some label collisions.
label_offsets = [
    (4, 4),
    (4, -8),
    (-4, 5),
    (-4, -8),
]

for i, row in plot_data.iterrows():
    dx, dy = label_offsets[i % len(label_offsets)]

    ax.annotate(
        row["location"],
        (
            row["hospital_beds_per_thousand"],
            row["total_deaths_per_million"],
        ),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7,
        ha="left" if dx >= 0 else "right",
        va="bottom" if dy >= 0 else "top",
    )


# --------------------------------------------------
# 9. Add a simple linear trend line
# --------------------------------------------------

# The line is descriptive only; it does not imply causation.
if (
    len(plot_data) >= 2
    and plot_data["hospital_beds_per_thousand"].nunique() > 1
):
    slope, intercept = np.polyfit(x, y, 1)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=1.5,
        label="Linear trend (descriptive)",
    )

    ax.legend(frameon=False)


# --------------------------------------------------
# 10. Format, save, and display the chart
# --------------------------------------------------

ax.set_title(
    "Hospital Bed Capacity vs. COVID-19 Deaths in Europe\n"
    f"Latest available observations through "
    f"{reference_date:%d %b %Y}",
    fontsize=14,
    pad=14,
)

ax.set_xlabel(
    "Hospital beds per 1,000 people",
    fontsize=11,
)

ax.set_ylabel(
    "Cumulative COVID-19 deaths per million people",
    fontsize=11,
)

ax.grid(True, linestyle=":", alpha=0.4)
ax.set_axisbelow(True)

fig.text(
    0.01,
    0.01,
    "Hospital bed values may reflect older source years. "
    "Death observation dates may differ by country. "
    "Association does not establish causation.",
    fontsize=8,
)

fig.tight_layout(rect=[0, 0.05, 1, 1])

fig.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight",
)

plt.show()

print(f"\nChart saved to: {OUTPUT_FILE}")