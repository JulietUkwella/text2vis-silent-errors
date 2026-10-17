import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

file_path = "owid-covid-data.csv"

df = pd.read_csv(file_path, parse_dates=["date"])

# Ensure the death-rate column contains numeric values
metric = "total_deaths_per_million"
df[metric] = pd.to_numeric(df[metric], errors="coerce")


# --------------------------------------------------
# 2. Select the 12 sovereign countries of South America
# --------------------------------------------------

south_america_countries = [
    "Argentina",
    "Bolivia",
    "Brazil",
    "Chile",
    "Colombia",
    "Ecuador",
    "Guyana",
    "Paraguay",
    "Peru",
    "Suriname",
    "Uruguay",
    "Venezuela",
]

# Use the explicit country list to exclude aggregate rows
# such as "World" or regional summaries.
sa = df.loc[
    df["location"].isin(south_america_countries),
    ["location", "date", metric]
].copy()

if sa.empty:
    raise ValueError(
        "No matching South American country rows were found. "
        "Check the file path and the country names in the dataset."
    )

# Keep rows with both a valid date and a usable death-rate value.
valid = sa.dropna(subset=["date", metric]).copy()

countries_with_data = [
    country
    for country in south_america_countries
    if country in valid["location"].unique()
]

countries_without_data = sorted(
    set(south_america_countries) - set(countries_with_data)
)

if countries_without_data:
    print(
        "Countries with no usable death-rate observations:",
        ", ".join(countries_without_data)
    )

if not countries_with_data:
    raise ValueError(
        "No usable total_deaths_per_million values were found."
    )


# --------------------------------------------------
# 3. Find a common comparison date
# --------------------------------------------------

# Rows are dates; columns are countries.
# The value in each cell is the cumulative death rate
# for that country on that date.
wide = valid.pivot_table(
    index="date",
    columns="location",
    values=metric,
    aggfunc="last"
).reindex(columns=countries_with_data)

# Keep dates for which every included country has a value.
complete_dates = wide.dropna(subset=countries_with_data)

if not complete_dates.empty:
    # Latest date with an observation for every included country
    comparison_date = complete_dates.index.max()

    values = wide.loc[comparison_date, countries_with_data]

    latest = (
        values.rename(metric)
        .rename_axis("location")
        .reset_index()
    )

    latest["data_date"] = comparison_date
    latest["label"] = latest["location"]

    chart_title = (
        "Reported COVID-19 Mortality in South America\n"
        f"Cumulative deaths per million people, as of "
        f"{comparison_date:%Y-%m-%d}"
    )

else:
    # Fallback: use the latest valid observation for each country.
    # Dates are displayed in the country labels because they
    # may differ between countries.
    latest = (
        valid.sort_values("date")
        .groupby("location", as_index=False)
        .tail(1)
        [["location", "date", metric]]
        .copy()
    )

    latest["data_date"] = latest["date"]

    latest["label"] = latest.apply(
        lambda row: (
            f"{row['location']} "
            f"({row['data_date']:%Y-%m-%d})"
        ),
        axis=1
    )

    chart_title = (
        "Reported COVID-19 Mortality in South America\n"
        "Cumulative deaths per million people, "
        "latest available observation per country"
    )


# --------------------------------------------------
# 4. Sort countries by deaths per million
# --------------------------------------------------

latest = latest.sort_values(
    metric,
    ascending=True
)


# --------------------------------------------------
# 5. Create the horizontal bar chart
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    latest["label"],
    latest[metric]
)

# Add the value to the end of each bar.
ax.bar_label(
    bars,
    fmt="%.1f",
    padding=4
)

ax.set_title(chart_title, fontsize=13)
ax.set_xlabel("Cumulative reported COVID-19 deaths per million people")
ax.set_ylabel("Country")

# Light gridlines make comparisons easier.
ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)
ax.set_axisbelow(True)

# Leave room for the numeric labels at the end of the bars.
ax.margins(x=0.12)

fig.tight_layout()
plt.show()