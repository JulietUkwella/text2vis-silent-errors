import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
file_path = "owid-covid-data.csv"
df = pd.read_csv(file_path)

# Convert dates to datetime values.
# Unparseable dates become missing values.
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Ensure the measure is numeric.
# Missing or invalid values become NaN.
df["total_deaths_per_million"] = pd.to_numeric(
    df["total_deaths_per_million"],
    errors="coerce"
)

# --------------------------------------------------
# 2. Select the four countries
# --------------------------------------------------
countries = ["France", "Germany", "Italy", "Spain"]

plot_df = df.loc[
    df["location"].isin(countries),
    ["location", "date", "total_deaths_per_million"]
].copy()

# Exclude rows with invalid dates.
# Do not remove missing death-rate values: keeping them as NaN
# allows the plot to show gaps rather than inventing data.
plot_df = plot_df.dropna(subset=["date"])

if plot_df.empty:
    raise ValueError(
        "No matching country records found. "
        "Check the CSV file and the location names."
    )

# --------------------------------------------------
# 3. Check for duplicate country-date records
# --------------------------------------------------
duplicates = plot_df.duplicated(
    subset=["location", "date"],
    keep=False
)

if duplicates.any():
    duplicate_rows = plot_df.loc[duplicates].sort_values(
        ["location", "date"]
    )
    raise ValueError(
        "Duplicate country-date records found. "
        "Inspect these rows before plotting:\n"
        f"{duplicate_rows.to_string(index=False)}"
    )

# --------------------------------------------------
# 4. Plot cumulative deaths per million over time
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

for country in countries:
    country_df = (
        plot_df.loc[plot_df["location"] == country]
        .sort_values("date")
    )

    if country_df.empty:
        print(f"Warning: No records found for {country}.")
        continue

    ax.plot(
        country_df["date"],
        country_df["total_deaths_per_million"],
        label=country,
        linewidth=2
    )

# --------------------------------------------------
# 5. Format the chart
# --------------------------------------------------
ax.set_title(
    "Cumulative COVID-19 Deaths per Million People",
    fontsize=15
)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Total deaths per million people", fontsize=12)

# Use a shared scale starting at zero.
ax.set_ylim(bottom=0)

# Format the date axis for readability.
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
)

ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(title="Country")
fig.tight_layout()

# Display the chart.
plt.show()