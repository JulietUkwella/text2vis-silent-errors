import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv")

# 2. Convert date and death columns to the correct data types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["total_deaths"] = pd.to_numeric(
    df["total_deaths"], errors="coerce"
)

# 3. Exclude aggregate regions and keep individual countries
# OWID aggregate rows typically have ISO codes beginning with OWID_
countries = df[
    df["iso_code"].notna()
    & ~df["iso_code"].str.startswith("OWID_", na=False)
].copy()

# 4. Sort records chronologically and get each country's
# latest reported cumulative death count
countries = countries.sort_values("date")

latest_deaths = (
    countries.dropna(subset=["total_deaths"])
    .groupby(["iso_code", "location"], as_index=False)
    .tail(1)
    [["iso_code", "location", "total_deaths"]]
)

# 5. Select the top 10 countries by total confirmed deaths
top_10 = latest_deaths.nlargest(10, "total_deaths").sort_values(
    "total_deaths", ascending=True
)

# 6. Print the ranking
print("Top 10 countries by total confirmed COVID-19 deaths:")
print(
    top_10[["location", "total_deaths"]]
    .sort_values("total_deaths", ascending=False)
    .to_string(
        index=False,
        formatters={"total_deaths": "{:,.0f}".format}
    )
)

# 7. Create a horizontal bar chart
plt.figure(figsize=(12, 7))

bars = plt.barh(
    top_10["location"],
    top_10["total_deaths"],
)

# Add death counts to the ends of the bars
for bar in bars:
    value = bar.get_width()
    plt.text(
        value,
        bar.get_y() + bar.get_height() / 2,
        f" {value:,.0f}",
        va="center",
        fontsize=9,
    )

plt.title(
    "Top 10 Countries by Total Confirmed COVID-19 Deaths",
    fontsize=15,
    pad=15,
)
plt.xlabel("Total confirmed deaths")
plt.ylabel("Country")
plt.gca().xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
)
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()

# Save the chart and display it
plt.savefig("top_10_covid_deaths.png", dpi=300, bbox_inches="tight")
plt.show()
