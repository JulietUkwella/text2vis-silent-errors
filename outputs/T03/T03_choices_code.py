import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load and prepare the data
# --------------------------------------------------
df = pd.read_csv("owid-covid-data.csv")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

metric = "total_deaths_per_million"
df[metric] = pd.to_numeric(df[metric], errors="coerce")

# --------------------------------------------------
# 2. Select African countries, excluding aggregates
# --------------------------------------------------
continent = df["continent"].fillna("").astype(str).str.strip()
iso_code = df["iso_code"].fillna("").astype(str).str.strip()

africa = df.loc[
    continent.eq("Africa")
    & iso_code.ne("")
    & ~iso_code.str.startswith("OWID_")
].copy()

# --------------------------------------------------
# 3. Find the latest valid observation for each country
# --------------------------------------------------
# Exclude rows where the date or chosen measure is missing.
valid = africa.dropna(subset=["date", metric])

# Sort by date and keep one latest record per country.
latest = (
    valid.sort_values("date")
         .drop_duplicates(subset="iso_code", keep="last")
)

if latest.empty:
    raise ValueError(
        "No African country records with valid death-rate data were found."
    )

# --------------------------------------------------
# 4. Rank countries and select the top 10
# --------------------------------------------------
top10 = (
    latest.nlargest(10, metric)
          .sort_values(metric, ascending=True)
          .copy()
)

# Include the observation date so readers can see
# when each country's reported value was last updated.
top10["country_label"] = top10.apply(
    lambda row: (
        f"{row['location']} "
        f"({row['date']:%Y-%m-%d})"
    ),
    axis=1
)

# --------------------------------------------------
# 5. Create the horizontal bar chart
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    top10["country_label"],
    top10[metric]
)

ax.set_title(
    "African Countries with the Highest Reported "
    "COVID-19 Death Rates\n"
    "Latest non-missing observation per country",
    fontsize=14
)

ax.set_xlabel(
    "Cumulative reported COVID-19 deaths per million people"
)
ax.set_ylabel("Country (observation date)")

# Add a little space to the right for value labels.
max_value = max(top10[metric].max(), 1)
ax.set_xlim(0, max_value * 1.18)

# Label each bar with its numerical value.
for bar, value in zip(bars, top10[metric]):
    ax.text(
        bar.get_width() + max_value * 0.015,
        bar.get