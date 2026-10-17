import pandas as pd
import matplotlib.pyplot as plt

# Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# Keep country rows, excluding continents and other aggregate locations
df = df[~df["iso_code"].str.startswith("OWID_", na=False)]

# Find the latest recorded total_cases value for each country
df["date"] = pd.to_datetime(df["date"])
latest_cases = (
    df.sort_values("date")
      .groupby("location", as_index=False)
      .last()[["location", "total_cases"]]
)

# Remove rows with missing case counts
latest_cases = latest_cases.dropna(subset=["total_cases"])

# Select the 10 countries with the most total cases
top_10 = latest_cases.nlargest(10, "total_cases")
top_10 = top_10.sort_values("total_cases", ascending=True)

# Create a horizontal bar chart
plt.figure(figsize=(12, 7))
plt.barh(top_10["location"], top_10["total_cases"])

plt.xlabel("Total confirmed COVID-19 cases")
plt.ylabel("Country")
plt.title("Top 10 Countries by Total COVID-19 Cases")
plt.gca().xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
)

plt.tight_layout()
plt.show()
