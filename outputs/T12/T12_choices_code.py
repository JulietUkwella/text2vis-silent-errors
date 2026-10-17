import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

file_path = "owid-covid-data.csv"
df = pd.read_csv(file_path)

# Check that the necessary columns are present.
required_columns = {
    "iso_code",
    "location",
    "date",
    "people_vaccinated"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"The CSV file is missing these columns: {missing_columns}"
    )

# --------------------------------------------------
# 2. Clean the relevant columns
# --------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["people_vaccinated"] = pd.to_numeric(
    df["people_vaccinated"],
    errors="coerce"
)

# Normalize ISO codes for consistent filtering.
df["_iso_code"] = df["iso_code"].fillna("").str.upper()

# Include entries with standard three-letter ISO codes.
# Kosovo is included separately because OWID uses OWID_KOS.
is_country_or_territory = (
    df["_iso_code"].str.fullmatch(r"[A-Z]{3}", na=False)
    | df["_iso_code"].eq("OWID_KOS")
)

# Exclude rows without a usable date or vaccination count.
valid_rows = (
    is_country_or_territory
    & df["date"].notna()
    & df["people_vaccinated"].notna()
)

df_clean = df.loc[
    valid_rows,
    ["_iso_code", "location", "date", "people_vaccinated"]
].copy()

if df_clean.empty:
    raise ValueError(
        "No valid country/territory vaccination observations were found."
    )

# --------------------------------------------------
# 3. Find the latest available observation per entry
# --------------------------------------------------

# The values are cumulative, so do not sum across dates.
# Select the latest date with a non-missing vaccination count
# for each country or territory.

latest = (
    df_clean
    .sort_values(["_iso_code", "date"])
    .drop_duplicates(subset="_iso_code", keep="last")
)

# --------------------------------------------------
# 4. Rank countries and select the top 10
# --------------------------------------------------

top_10 = (
    latest
    .nlargest(10, "people_vaccinated")
    .copy()
)

# Convert counts to millions for plotting.
top_10["vaccinated_millions"] = (
    top_10["people_vaccinated"] / 1_000_000
)

# Include each entry's latest reporting date in its label.
top_10["chart_label"] = (
    top_10["location"].fillna(top_10["_iso_code"])
    + " ("
    + top_10["date"].dt.strftime("%Y-%m-%d")
    + ")"
)

# --------------------------------------------------
# 5. Create the horizontal bar chart
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    top_10["chart_label"],
    top_10["vaccinated_millions"]
)

# Show the largest value at the top.
ax.invert_yaxis()

ax.set_title(
    "Countries and Territories with the Most COVID-19 Vaccinations",
    fontsize=14,
    pad=15
)

ax.set_xlabel(
    "People who received at least one dose (millions)"
)

ax.set_ylabel("Country or territory (latest reporting date)")

# Format the horizontal axis using millions of people.
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, position: f"{x:,.0f}")
)

# Format bar-end labels using millions or billions.
def format_people(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    return f"{value / 1_000_000:.0f}M"

labels = [
    format_people(value)
    for value in top_10["people_vaccinated"]
]

ax.bar_label(
    bars,
    labels=labels,
    padding=4
)

# Leave space on the right for the bar-end labels.
max_value = top_10["vaccinated_millions"].max()

ax.set_xlim(
    0,
    max_value * 1.18 if max_value > 0 else 1
)

ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()