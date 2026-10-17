import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
FILE = "owid-covid-data.csv"

df = pd.read_csv(FILE, low_memory=False)

required_columns = [
    "iso_code",
    "continent",
    "location",
    "date",
    "diabetes_prevalence",
    "total_deaths_per_million",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Convert dates and the two analytical variables to usable types.
df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in ["diabetes_prevalence", "total_deaths_per_million"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# --------------------------------------------------
# 2. Keep country- and territory-level observations
# --------------------------------------------------
# Aggregate rows, such as the world and continents, generally
# have no continent value of their own.
country_rows = df.loc[
    df["continent"].notna()
    & df["iso_code"].notna()
    & df["date"].notna()
].copy()


# --------------------------------------------------
# 3. Select the latest valid diabetes estimate
#    for each country/territory
# --------------------------------------------------
valid_diabetes = country_rows.loc[
    country_rows["diabetes_prevalence"].between(0, 100)
].copy()

latest_diabetes = (
    valid_diabetes
    .sort_values("date")
    .groupby("iso_code", as_index=False)
    .tail(1)[
        ["iso_code", "diabetes_prevalence"]
    ]
)


# --------------------------------------------------
# 4. Select the latest valid cumulative death rate
#    for each country/territory
# --------------------------------------------------
valid_deaths = country_rows.loc[
    country_rows["total_deaths_per_million"].ge(0)
].copy()

latest_deaths = (
    valid_deaths
    .sort_values("date")
    .groupby("iso_code", as_index=False)
    .tail(1)[
        [
            "iso_code",
            "location",
            "date",
            "total_deaths_per_million",
        ]
    ]
)


# --------------------------------------------------
# 5. Combine both measures into one row per country
# --------------------------------------------------
plot_df = latest_deaths.merge(
    latest_diabetes,
    on="iso_code",
    how="inner",
)

plot_df = plot_df.dropna(
    subset=[
        "diabetes_prevalence",
        "total_deaths_per_million",
    ]
)

if plot_df.empty:
    raise ValueError(
        "No country/territory observations have valid values "
        "for both diabetes prevalence and deaths per million."
    )


# --------------------------------------------------
# 6. Calculate the Spearman rank correlation
# --------------------------------------------------
rho = plot_df["diabetes_prevalence"].corr(
    plot_df["total_deaths_per_million"],
    method="spearman",
)

rho_text = f"{rho:.2f}" if pd.notna(rho) else "not defined"


# --------------------------------------------------
# 7. Create the scatter plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    plot_df["diabetes_prevalence"],
    plot_df["total_deaths_per_million"],
    alpha=0.7,
    edgecolors="none",
)

# Keep zero-death observations visible while reducing the
# visual dominance of countries with exceptionally high rates.
ax.set_yscale("symlog", linthresh=10)

ax.set_xlabel(
    "Diabetes prevalence (% of people aged 20–79)"
)

ax.set_ylabel(
    "Cumulative reported COVID-19 deaths per million "
    "(symmetric logarithmic scale)"
)

ax.set_title(
    "Diabetes prevalence vs. cumulative COVID-19 mortality\n"
    f"{len(plot_df)} countries/territories | "
    f"Spearman rank correlation = {rho_text}"
)

ax.grid(True, which="both", alpha=0.25)

fig.tight_layout()
plt.show()


# --------------------------------------------------
# 8. Print information needed to interpret the chart
# --------------------------------------------------
print(f"Countries/territories plotted: {len(plot_df)}")
print(f"Spearman rank correlation: {rho_text}")

print(
    "Selected death-observation dates range from "
    f"{plot_df['date'].min().date()} to "
    f"{plot_df['date'].max().date()}."
)