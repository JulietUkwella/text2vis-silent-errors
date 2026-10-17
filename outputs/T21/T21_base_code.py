import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# 2. Convert dates and death counts to appropriate data types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["total_deaths"] = pd.to_numeric(df["total_deaths"], errors="coerce")

# 3. Keep country-level records and exclude aggregate regions
#    such as World and continents.
df = df[df["iso_code"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()

# 4. Keep records with known cumulative death counts
df = df.dropna(subset=["date", "total_deaths"])

# 5. Get each country's latest available cumulative death count
latest = (
    df.sort_values("date")
      .groupby("location", as_index=False)
      .tail(1)
      .copy()
)

# 6. Rank countries by total COVID-19 deaths
top10 = latest.nlargest(10, "total_deaths").sort_values(
    "total_deaths", ascending=True
)

# 7. Print the country with the highest total
deadliest = top10.iloc[-1]

print("Country with the highest reported COVID-19 deaths:")
print(f"Country: {deadliest['location']}")
print(f"Total deaths: {deadliest['total_deaths']:,.0f}")
print(f"Latest data date: {deadliest['date'].date()}")

# 8. Create a horizontal bar chart
plt.figure(figsize=(11, 7))

bars = plt.barh(
    top10["location"],
    top10["total_deaths"],
    color="steelblue"
)

plt.title(
    "Countries with the Highest Cumulative COVID-19 Deaths",
    fontsize=15
)
plt.xlabel("Total reported deaths", fontsize=12)
plt.ylabel("Country", fontsize=12)

# Format the x-axis with thousands separators
plt.gca().xaxis.set_major_formatter(
    ticker.StrMethodFormatter("{x:,.0f}")
)

# Add death counts beside each bar
for bar in bars:
    width = bar.get_width()
    plt.text(
        width,
        bar.get_y() + bar.get_height() / 2,
        f" {width:,.0f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()
plt.show()