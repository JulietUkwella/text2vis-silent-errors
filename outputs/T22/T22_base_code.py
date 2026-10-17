import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv", parse_dates=["date"])

# 2. Select country-level data for March 2020
march = df.loc[
    df["date"].between("2020-03-01", "2020-03-31")
    & df["continent"].notna()
    & ~df["iso_code"].fillna("").str.startswith("OWID_"),
    ["iso_code", "location", "date", "total_cases"]
].copy()

march["total_cases"] = pd.to_numeric(
    march["total_cases"], errors="coerce"
)

march = march.dropna(
    subset=["iso_code", "total_cases"]
).sort_values("date")

# 3. Get the first and last available case counts for each country
stats = (
    march.groupby("iso_code")
    .agg(
        country=("location", "last"),
        start_date=("date", "first"),
        start_cases=("total_cases", "first"),
        end_date=("date", "last"),
        end_cases=("total_cases", "last")
    )
    .reset_index()
)

# Calculate the number of days covered
stats["days"] = (
    stats["end_date"] - stats["start_date"]
).dt.days

# Retain countries with a meaningful starting count and
# observations spanning at least 25 days
stats = stats[
    (stats["start_cases"] >= 100)
    & (stats["days"] >= 25)
    & (stats["end_cases"] > stats["start_cases"])
].copy()

# 4. Calculate compound average daily growth rate (%)
stats["daily_growth_pct"] = (
    (stats["end_cases"] / stats["start_cases"])
    ** (1 / stats["days"]) - 1
) * 100

# Select the 10 countries with the fastest growth
top10 = stats.nlargest(10, "daily_growth_pct")
top10 = top10.sort_values("daily_growth_pct")

# 5. Print the ranking
print("Countries with the fastest relative case growth in March 2020:")
print(
    top10.sort_values("daily_growth_pct", ascending=False)[
        ["country", "start_cases", "end_cases", "daily_growth_pct"]
    ].to_string(
        index=False,
        formatters={"daily_growth_pct": "{:.2f}%".format}
    )
)

# 6. Create a horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    top10["country"],
    top10["daily_growth_pct"]
)

ax.set_title(
    "Countries with the Fastest COVID-19 Case Growth\n"
    "March 2020",
    fontsize=14
)
ax.set_xlabel("Average compound daily growth in confirmed cases (%)")
ax.set_ylabel("Country")
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# Display growth rates at the ends of the bars
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.1,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}%",
        va="center"
    )

plt.tight_layout()
plt.show()