import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# 2. Convert columns to the correct data types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["tests_per_case"] = pd.to_numeric(
    df["tests_per_case"], errors="coerce"
)

# 3. Keep country-level records and valid testing values
df = df[
    df["continent"].notna()
    & df["tests_per_case"].notna()
    & (df["tests_per_case"] > 0)
].copy()

# Remove infinite values, if any
df = df[
    df["tests_per_case"].apply(lambda x: pd.notna(x) and x != float("inf"))
]

# 4. Find each country's latest available testing value
latest = (
    df.sort_values("date")
      .groupby("location")
      .tail(1)
)

# 5. Select the 15 countries with the highest tests-per-case values
top_countries = (
    latest.nlargest(15, "tests_per_case")
          .sort_values("tests_per_case", ascending=True)
)

# 6. Create the horizontal bar chart
fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    top_countries["location"],
    top_countries["tests_per_case"]
)

# Display the values at the ends of the bars
ax.bar_label(bars, fmt="%.1f", padding=4)

ax.set_title(
    "Countries with the Highest COVID-19 Tests per Confirmed Case\n"
    "(Latest Available Reported Values)"
)
ax.set_xlabel("Tests per confirmed case (higher = more testing relative to cases)")
ax.set_ylabel("Country")
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
