import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv", parse_dates=["date"])

# Ensure the analysis columns are numeric
columns = ["aged_65_older", "total_deaths_per_million"]
for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. Exclude OWID aggregate rows, such as World and continents
countries = df[
    ~df["iso_code"].fillna("").str.startswith("OWID_")
].copy()

# Keep rows with both variables available
countries = countries.dropna(
    subset=["iso_code", "aged_65_older", "total_deaths_per_million"]
)

# 3. Select the latest available record for each country
latest = (
    countries.sort_values("date")
    .groupby("iso_code")
    .tail(1)
    .copy()
)

# Extract variables for the chart
x = latest["aged_65_older"]
y = latest["total_deaths_per_million"]

# 4. Calculate the correlation coefficient
correlation = x.corr(y)

# Calculate a linear trendline
slope, intercept = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept

# 5. Create the scatter plot
plt.figure(figsize=(10, 6))

plt.scatter(x, y, alpha=0.7, edgecolor="black", linewidth=0.4)
plt.plot(x_line, y_line, linewidth=2, color="red")

plt.xlabel("Population aged 65 and older (% of population)")
plt.ylabel("Cumulative COVID-19 deaths per million")
plt.title("Share of Older People vs. COVID-19 Death Rates")

plt.annotate(
    f"Pearson correlation: {correlation:.2f}\n"
    f"Countries included: {len(latest)}",
    xy=(0.03, 0.97),
    xycoords="axes fraction",
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Correlation coefficient: {correlation:.3f}")
print(f"Number of countries analyzed: {len(latest)}")