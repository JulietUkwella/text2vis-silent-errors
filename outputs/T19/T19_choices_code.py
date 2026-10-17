
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load and clean the data
# --------------------------------------------------

FILE = "owid-covid-data.csv"

df = pd.read_csv(FILE, low_memory=False)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Ensure the relevant columns are numeric.
for col in ["gdp_per_capita", "total_deaths_per_million"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Latest date represented anywhere in the file.
cutoff_date = df["date"].max()

if pd.isna(cutoff_date):
    raise ValueError("No valid dates were found in the CSV file.")

# Ignore rows with invalid dates or no country/territory
# classification. This removes the usual OWID aggregates.
country_rows = df.loc[
    df["date"].notna()
    & df["iso_code"].notna()
    & df["continent"].notna()
    & (df["date"] <= cutoff_date)
].copy()

# --------------------------------------------------
# 2. Get the latest available death rate per location
# --------------------------------------------------

deaths = (
    country_rows
    .dropna(subset=["total_deaths_per_million"])
    .sort_values(["iso_code", "date"])
    .drop_duplicates(subset="iso_code", keep="last")
    [
        [
            "iso_code",
            "location",
            "continent",
            "date",
            "total_deaths_per_million",
        ]
    ]
    .rename(columns={"date": "deaths_observation_date"})
)

# --------------------------------------------------
# 3. Get the latest available GDP per capita
# --------------------------------------------------

gdp = (
    country_rows
    .dropna(subset=["gdp_per_capita"])
    .sort_values(["iso_code", "date"])
    .drop_duplicates(subset="iso_code", keep="last")
    [["iso_code", "gdp_per_capita"]]
)

# Join GDP and deaths so each location appears only once.
plot_df = deaths.merge(gdp, on="iso_code", how="inner")

# Keep valid values suitable for the chosen axes.
plot_df = plot_df.loc[
    (plot_df["gdp_per_capita"] > 0)
    & (plot_df["total_deaths_per_million"] >= 0)
].copy()

if plot_df.empty:
    raise ValueError(
        "No locations have valid GDP and death-rate data."
    )

# --------------------------------------------------
# 4. Check how old the selected death observations are
# --------------------------------------------------

plot_df["days_before_cutoff"] = (
    cutoff_date - plot_df["deaths_observation_date"]
).dt.days

print(f"Latest date in CSV: {cutoff_date:%Y-%m-%d}")
print(f"Number of locations plotted: {len(plot_df)}")

stale_count = (
    plot_df["days_before_cutoff"] > 90
).sum()

print(
    "Locations whose selected death-rate observation is "
    f"more than 90 days before the cutoff: {stale_count}"
)

# Spearman correlation measures the direction and strength
# of the rank-based association between GDP and death rates.
rho = plot_df["gdp_per_capita"].corr(
    plot_df["total_deaths_per_million"],
    method="spearman",
)

print(f"Spearman rank correlation: {rho:.3f}")

# --------------------------------------------------
# 5. Plot GDP per capita against reported COVID deaths
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 7))

for continent, group in plot_df.groupby("continent", sort=True):
    ax.scatter(
        group["gdp_per_capita"],
        group["total_deaths_per_million"],
        s=35,
        alpha=0.75,
        label=continent,
    )

# GDP varies across a wide range, so use a log scale.
ax.set_xscale("log")

ax.set_xlabel("GDP per capita (OWID dataset units; log scale)")
ax.set_ylabel("Cumulative reported COVID-19 deaths per million")

ax.set_title(
    "GDP per Capita vs. Reported COVID-19 Deaths\n"
    f"Latest available observations through {cutoff_date:%d %b %Y}"
)

ax.set_ylim(bottom=0)
ax.grid(True, which="both", alpha=0.25)

ax.legend(
    title="Continent",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
)

fig.tight_layout()
plt.show()
