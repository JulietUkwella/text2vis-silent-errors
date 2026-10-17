import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load and prepare the data
# --------------------------------------------------

FILE = "owid-covid-data.csv"

df = pd.read_csv(FILE, parse_dates=["date"])

# Ensure the policy measure is numeric.
df["stringency_index"] = pd.to_numeric(
    df["stringency_index"], errors="coerce"
)

# Select the requested period: all of 2023.
year_data = df.loc[
    (df["date"] >= "2023-01-01")
    & (df["date"] < "2024-01-01")
].copy()

# Keep country/territory locations rather than aggregate rows.
# In this dataset, aggregate locations generally have no continent.
country_data = year_data.loc[
    year_data["continent"].notna()
    & year_data["iso_code"].notna()
].copy()

# The dataset should have one row per location per date.
# Remove duplicate country-date records to avoid double-counting.
country_data = country_data.drop_duplicates(
    subset=["iso_code", "date"]
)

# --------------------------------------------------
# 2. Calculate the annual average and data coverage
# --------------------------------------------------

DAYS_IN_2023 = 365
MIN_DAYS_REQUIRED = 330  # Approximately 90% coverage

country_summary = (
    country_data
    .groupby(["iso_code", "location"], as_index=False)
    .agg(
        mean_stringency=("stringency_index", "mean"),
        days_with_score=("stringency_index", "count")
    )
)

# Calculate the proportion of the year with an observed score.
country_summary["coverage_pct"] = (
    country_summary["days_with_score"]
    / DAYS_IN_2023
    * 100
)

# Only include countries with sufficient data for the main chart.
comparable = country_summary.loc[
    country_summary["days_with_score"] >= MIN_DAYS_REQUIRED
].copy()

print("Country/territory locations with 2023 records:",
      country_data["iso_code"].nunique())

print("Locations meeting the coverage requirement:",
      len(comparable))

# --------------------------------------------------
# 3. Plot the comparison, if the data supports it
# --------------------------------------------------

if not comparable.empty:

    # Sort from lowest average strictness to highest.
    comparable = comparable.sort_values(
        "mean_stringency", ascending=True
    )

    # Adjust the figure height to fit the country labels.
    fig_height = max(6, 0.28 * len(comparable))

    fig, ax = plt.subplots(figsize=(11, fig_height))

    ax.barh(
        comparable["location"],
        comparable["mean_stringency"]
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Average daily stringency index (0–100)")
    ax.set_ylabel("Country or territory")
    ax.set_title(
        "Average COVID-19 Policy Stringency by Country, 2023\n"
        "(Only locations with at least 330 observed daily scores)"
    )

    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

else:
    print(
        "\nNo country or territory has at least "
        f"{MIN_DAYS_REQUIRED} observed stringency scores in 2023."
    )
    print(
        "A reliable full-year comparison cannot be produced "
        "from the available observations."
    )

    # --------------------------------------------------
    # 4. Show a diagnostic instead of a misleading ranking
    # --------------------------------------------------

    available = country_summary.loc[
        country_summary["days_with_score"] > 0
    ].copy()

    if not available.empty:

        # Show up to 20 locations with the most observed days.
        available = available.sort_values(
            "days_with_score", ascending=True
        ).tail(20)

        fig_height = max(4, 0.35 * len(available))
        fig, ax = plt.subplots(figsize=(10, fig_height))

        ax.barh(
            available["location"],
            available["days_with_score"]
        )

        ax.set_xlim(0, DAYS_IN_2023)
        ax.set_xlabel("Days with a reported stringency score (out of 365)")
        ax.set_ylabel("Country or territory")
        ax.set_title(
            "Availability of COVID-19 Stringency Data in 2023\n"
            "(This chart shows data coverage, not restriction strictness)"
        )

        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

    else:

        fig, ax = plt.subplots(figsize=(10, 3))

        ax.text(
            0.5,
            0.5,
            "No non-missing country-level stringency scores "
            "were found for 2023.",
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
            fontsize=12
        )

        ax.set_title(
            "COVID-19 Restriction Comparison: Insufficient 2023 Data"
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()