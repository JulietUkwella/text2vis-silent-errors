import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv")

# Convert dates and vaccination/population fields to suitable types
df["date"] = pd.to_datetime(df["date"], errors="coerce")

for col in [
    "people_fully_vaccinated_per_hundred",
    "people_fully_vaccinated",
    "population",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. Prefer OWID's regional aggregate for Africa
africa = df.loc[
    df["location"].eq("Africa"),
    ["date", "people_fully_vaccinated_per_hundred"],
].copy()

africa = africa.rename(
    columns={
        "people_fully_vaccinated_per_hundred": "share_pct"
    }
)

africa = africa.dropna(subset=["date", "share_pct"])

if not africa.empty:
    series = africa[["date", "share_pct"]].copy()
    method_note = (
        "Source: OWID Africa aggregate; "
        "share calculated per 100 people."
    )

else:
    # 3. Fallback: use individual African locations
    countries = df.loc[df["continent"].eq("Africa")].copy()

    # Keep only records with a vaccination count and population
    valid = countries.dropna(
        subset=["date", "people_fully_vaccinated", "population"]
    )

    valid = valid.loc[valid["population"] > 0]

    # Aggregate counts and population by date.
    # This is population-weighted, not an average of country percentages.
    daily = (
        valid.groupby("date", as_index=False)
        .agg(
            fully_vaccinated=("people_fully_vaccinated", "sum"),
            population_covered=("population", "sum"),
        )
    )

    daily = daily.loc[daily["population_covered"] > 0].copy()

    daily["share_pct"] = (
        daily["fully_vaccinated"]
        / daily["population_covered"]
        * 100
    )

    series = daily[["date", "share_pct"]].copy()

    method_note = (
        "Fallback estimate: population-weighted share among "
        "African locations reporting both vaccination counts "
        "and population; coverage may vary by date."
    )

# 4. Clean and sort the time series
series = (
    series.dropna(subset=["date", "share_pct"])
    .sort_values("date")
    .drop_duplicates(subset=["date"], keep="last")
)

if series.empty:
    raise ValueError(
        "No usable African vaccination data was found in the CSV."
    )

# 5. Get the latest available estimate
latest = series.iloc[-1]

print(
    f"Latest available observation: {latest['date']:%Y-%m-%d}"
)
print(
    f"Share of people fully vaccinated: "
    f"{latest['share_pct']:.1f}%"
)

if (series["share_pct"] > 100).any():
    print(
        "Warning: Some percentages exceed 100%. "
        "Check reporting and population inconsistencies."
    )

# 6. Plot the results
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    series["date"],
    series["share_pct"],
    linewidth=2,
)

# Emphasize the latest observation
ax.scatter(
    latest["date"],
    latest["share_pct"],
    zorder=3,
)

ax.annotate(
    f"{latest['share_pct']:.1f}%\n"
    f"{latest['date']:%Y-%m-%d}",
    xy=(latest["date"], latest["share_pct"]),
    xytext=(8, 8),
    textcoords="offset points",
)

ax.set_title(
    "Share of People in Africa Fully Vaccinated Against COVID-19"
)
ax.set_xlabel("Date")
ax.set_ylabel("People fully vaccinated (% of population)")
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)

# Explain which calculation was used
fig.text(
    0.01,
    0.01,
    method_note,
    ha="left",
    fontsize=8,
    wrap=True,
)

fig.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()