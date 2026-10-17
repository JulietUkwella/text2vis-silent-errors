import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Our World in Data COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# 2. Convert columns to the appropriate data types
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["diabetes_prevalence"] = pd.to_numeric(
    df["diabetes_prevalence"], errors="coerce"
)
df["total_deaths_per_million"] = pd.to_numeric(
    df["total_deaths_per_million"], errors="coerce"
)

# 3. Exclude aggregate locations such as the World and continents
df = df[
    df["iso_code"].notna()
    & ~df["iso_code"].astype(str).str.startswith("OWID_")
].dropna(subset=["date"])

# 4. Get the latest available non-missing values for each location
df = df.sort_values("date")

countries = (
    df.groupby(["iso_code", "location"], as_index=False)
    .agg(
        diabetes_prevalence=("diabetes_prevalence", "last"),
        deaths_per_million=("total_deaths_per_million", "last")
    )
    .dropna(subset=["diabetes_prevalence", "deaths_per_million"])
)

# 5. Calculate the correlation between the two variables
x = countries["diabetes_prevalence"]
y = countries["deaths_per_million"]

correlation = x.corr(y)

# 6. Create the scatter plot
plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    alpha=0.6,
    edgecolors="none",
    label="Countries"
)

# Add a linear trend line
if len(countries) >= 2 and x.nunique() > 1:
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    plt.plot(
        x_line,
        y_line,
        linewidth=2,
        label="Linear trend"
    )

# Add chart labels and title
plt.title("Diabetes Prevalence vs. COVID-19 Deaths by Country")
plt.xlabel("Diabetes prevalence (%)")
plt.ylabel("Cumulative COVID-19 deaths per million people")

plt.annotate(
    f"Pearson correlation: {correlation:.2f}\n"
    f"Locations included: {len(countries)}",
    xy=(0.03, 0.97),
    xycoords="axes fraction",
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()