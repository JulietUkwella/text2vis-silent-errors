import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 2. Keep African countries and exclude OWID regional aggregates
africa = df[
    (df["continent"] == "Africa") &
    (~df["iso_code"].str.startswith("OWID_", na=False))
].copy()

# 3. Get the latest available reported death rate for each country
deaths = africa.dropna(subset=["total_deaths_per_million"])
latest_deaths = (
    deaths.sort_values("date")
    .drop_duplicates(subset="location", keep="last")
)

# 4. Select the 10 countries with the highest deaths per million
top10 = latest_deaths.nlargest(10, "total_deaths_per_million")
top10 = top10.sort_values("total_deaths_per_million")

# 5. Plot a horizontal bar chart
plt.figure(figsize=(11, 7))

plt.barh(
    top10["location"],
    top10["total_deaths_per_million"],
    color="steelblue"
)

plt.xlabel("Reported cumulative COVID-19 deaths per million people")
plt.ylabel("Country")
plt.title("10 Most Affected African Countries by Reported COVID-19 Death Rate")
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.show()

# 6. Print the ranking and latest available data dates
print("Top 10 African countries by reported COVID-19 deaths per million:")
print(
    top10[
        ["location", "date", "total_deaths_per_million"]
    ].sort_values("total_deaths_per_million", ascending=False)
    .to_string(index=False)
)
