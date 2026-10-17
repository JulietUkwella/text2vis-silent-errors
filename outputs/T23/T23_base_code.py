import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# Convert date to datetime format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 2. Keep European countries only.
# The 3-letter ISO code filter excludes regional and global aggregates.
europe = df[
    (df["continent"] == "Europe")
    & (df["iso_code"].str.fullmatch(r"[A-Z]{3}", na=False))
].copy()

# 3. Sort by date so we can select the latest non-missing values
europe = europe.sort_values("date")

# Get the latest available COVID death rate and hospital bed capacity
country_data = (
    europe.groupby(["iso_code", "location"], as_index=False)
    .agg(
        deaths_per_million=("total_deaths_per_million", "last"),
        beds_per_1000=("hospital_beds_per_thousand", "last")
    )
)

# Remove countries missing either measurement
plot_df = country_data.dropna(
    subset=["deaths_per_million", "beds_per_1000"]
).copy()

# 4. Create the scatter plot
fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(
    plot_df["beds_per_1000"],
    plot_df["deaths_per_million"],
    alpha=0.75,
    s=60
)

# Add median reference lines
ax.axvline(
    plot_df["beds_per_1000"].median(),
    linestyle="--",
    alpha=0.6,
    label="Median hospital bed capacity"
)

ax.axhline(
    plot_df["deaths_per_million"].median(),
    linestyle=":",
    alpha=0.8,
    label="Median COVID-19 death rate"
)

# 5. Label countries with the highest death rates
# and the lowest hospital bed capacity
labels = pd.concat([
    plot_df.nlargest(8, "deaths_per_million"),
    plot_df.nsmallest(5, "beds_per_1000")
]).drop_duplicates(subset=["iso_code"])

for _, row in labels.iterrows():
    ax.annotate(
        row["location"],
        xy=(row["beds_per_1000"], row["deaths_per_million"]),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=8
    )

# 6. Add titles, labels, and formatting
ax.set_title(
    "Hospital Bed Capacity vs. COVID-19 Deaths in Europe",
    fontsize=15
)
ax.set_xlabel("Hospital beds per 1,000 people", fontsize=11)
ax.set_ylabel("Cumulative COVID-19 deaths per million people", fontsize=11)

ax.grid(True, alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()