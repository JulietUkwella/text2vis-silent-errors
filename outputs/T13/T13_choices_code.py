from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the data and validate the required columns
# --------------------------------------------------

file_path = Path("owid-covid-data.csv")
rate_col = "people_fully_vaccinated_per_hundred"

df = pd.read_csv(file_path, low_memory=False)

required_cols = {
    "iso_code",
    "continent",
    "location",
    "date",
    rate_col,
}

missing_cols = required_cols - set(df.columns)

if missing_cols:
    raise ValueError(
        f"The CSV is missing required columns: {sorted(missing_cols)}"
    )

# Convert date and vaccination rate to appropriate data types.
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")


# --------------------------------------------------
# 2. Keep European countries, excluding aggregates
# --------------------------------------------------

# Standard three-letter ISO codes help exclude OWID aggregate
# rows such as continents and the world. Locations without
# standard ISO alpha-3 codes, such as OWID_KOS, are also excluded.
iso_codes = df["iso_code"].astype("string")

country_mask = (
    df["continent"].eq("Europe")
    & iso_codes.str.fullmatch(r"[A-Z]{3}", na=False)
)

europe = df.loc[
    country_mask,
    ["iso_code", "location", "date", rate_col],
].copy()

if europe.empty:
    raise ValueError("No European country records were found.")

all_country_codes = set(europe["iso_code"].dropna())

# Rows without a usable date cannot be used to identify the
# latest observation.
invalid_dates = europe["date"].isna().sum()

if invalid_dates:
    print(f"Excluding {invalid_dates} rows with invalid or missing dates.")

europe = europe.dropna(subset=["date"]).copy()


# --------------------------------------------------
# 3. Check for duplicate country-date records
# --------------------------------------------------

duplicates = europe.duplicated(
    subset=["iso_code", "date"],
    keep=False,
)

if duplicates.any():
    print("Duplicate country-date records detected:")
    print(
        europe.loc[
            duplicates,
            ["iso_code", "location", "date", rate_col],
        ]
        .sort_values(["iso_code", "date"])
        .to_string(index=False)
    )

    raise ValueError(
        "Resolve duplicate country-date records before plotting "
        "to avoid an arbitrary comparison."
    )


# --------------------------------------------------
# 4. Select each country's latest non-missing rate
# --------------------------------------------------

# Do not replace missing vaccination rates with zero.
# Select the latest available non-missing rate for each country.
latest = (
    europe.dropna(subset=[rate_col])
    .sort_values(["iso_code", "date"])
    .drop_duplicates(subset=["iso_code"], keep="last")
    .copy()
)

# Report countries that have no usable vaccination-rate record.
country_names = (
    europe.groupby("iso_code")["location"]
    .first()
    .to_dict()
)

missing_rate_codes = sorted(
    all_country_codes - set(latest["iso_code"])
)

if missing_rate_codes:
    missing_names = [
        country_names.get(code, code)
        for code in missing_rate_codes
    ]
    print(
        "Countries without a usable vaccination-rate observation:",
        ", ".join(missing_names),
    )


# --------------------------------------------------
# 5. Exclude out-of-range rates
# --------------------------------------------------

# The measure is a percentage of the population. Values outside
# 0–100 require investigation and are excluded from this chart.
invalid_rates = ~latest[rate_col].between(0, 100)

if invalid_rates.any():
    print("Excluding countries with out-of-range latest rates:")
    print(
        latest.loc[
            invalid_rates,
            ["location", "date", rate_col],
        ].to_string(index=False)
    )

latest = latest.loc[~invalid_rates].copy()

if latest.empty:
    raise ValueError(
        "No valid European vaccination-rate observations remain."
    )


# --------------------------------------------------
# 6. Sort countries and label observations by date
# --------------------------------------------------

latest["country_label"] = (
    latest["location"]
    + " ("
    + latest["date"].dt.strftime("%Y-%m-%d")
    + ")"
)

# For a horizontal bar chart, ascending order places the
# highest value at the top.
plot_data = latest.sort_values(
    rate_col,
    ascending=True,
)


# --------------------------------------------------
# 7. Draw the horizontal bar chart
# --------------------------------------------------

fig_height = max(8, 0.32 * len(plot_data))

fig, ax = plt.subplots(figsize=(12, fig_height))

bars = ax.barh(
    plot_data["country_label"],
    plot_data[rate_col],
)

# Add percentage labels to the bars.
for bar, value in zip(bars, plot_data[rate_col]):
    ax.text(
        value + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center",
        ha="left",
        fontsize=8,
    )

ax.set_xlim(0, 110)
ax.set_xticks(range(0, 101, 20))

ax.set_xlabel("Population fully vaccinated (%)")
ax.set_ylabel("Country (date of latest reported value)")

ax.set_title(
    "COVID-19 Full Vaccination Coverage in European Countries\n"
    "Latest Available Observation for Each Country"
)

ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()