import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("owid-covid-data.csv")

# 2. Prepare the date and testing-rate columns
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["total_tests_per_thousand"] = pd.to_numeric(
    df["total_tests_per_thousand"], errors="coerce"
)

# 3. Keep country-level locations and the fields needed for analysis.
# A non-missing continent generally excludes OWID global and regional aggregates.
country_data = df.loc[
    df["continent"].notna(),
    [
        "location",
        "continent",
        "date",
        "total_tests_per_thousand",
        "tests_units",
    ],
].copy()

# 4. Exclude observations with missing dates or testing rates.
# Negative cumulative rates are not meaningful for this comparison.
country_data = country_data.dropna(
    subset=["date", "total_tests_per_thousand"]
)
country_data = country_data[
    country_data["total_tests_per_thousand"] >= 0
]

if country_data.empty:
    raise ValueError(
        "No usable country-level observations were found for "
        "'total_tests_per_thousand'."
    )

# 5. Select each location's latest non-missing reported testing rate.
latest = (
    country_data
    .sort_values(["location", "date"])
    .drop_duplicates(subset="location", keep="last")
)

# 6. Rank locations and select the top 20.
TOP_N = 20

top = latest.nlargest(TOP_N, "total_tests_per_thousand").copy()

# Print the ranking, reporting date, and test unit for interpretation.
print("\nTop locations by reported cumulative testing rate:")
print(
    top[
        [
            "location",
            "total_tests_per_thousand",
            "date",
            "tests_units",
        ]
    ]
    .sort_values("total_tests_per_thousand", ascending=False)
    .to_string(
        index=False,
        formatters={
            "total_tests_per_thousand": lambda x: f"{x:,.1f}",
            "date": lambda x: x.strftime("%Y-%m-%d"),
        },
    )
)

# 7. Sort from lowest to highest so the highest bar appears at the top.
plot_data = top.sort_values("total_tests_per_thousand", ascending=True)

# 8. Draw a horizontal bar chart.
fig, ax = plt.subplots(
    figsize=(12, max(6, 0.38 * len(plot_data)))
)

bars = ax.barh(
    plot_data["location"],
    plot_data["total_tests_per_thousand"],
)

# Show both the testing rate and its latest observation date.
max_rate = plot_data["total_tests_per_thousand"].max()
x_padding = max(max_rate * 0.015, 0.1)

for bar, (_, row) in zip(bars, plot_data.iterrows()):
    rate = row["total_tests_per_thousand"]
    date_label = row["date"].strftime("%Y-%m-%d")

    ax.text(
        bar.get_width() + x_padding,
        bar.get_y() + bar.get_height() / 2,
        f"{rate:,.1f}  ({date_label})",
        va="center",
        fontsize=8,
    )

# 9. Add descriptive labels and a note about interpretation.
ax.set_title(
    "Countries with the Highest Reported Cumulative COVID-19 Testing",
    pad=14,
)
ax.set_xlabel("Reported cumulative tests per 1,000 people")
ax.set_ylabel("")

ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# Leave room for the values and dates printed beside the bars.
ax.set_xlim(0, max_rate * 1.5 if max_rate > 0 else 1)

fig.text(
    0.01,
    0.01,
    "Each bar uses the latest non-missing observation for that location; "
    "reporting dates and test definitions vary. "
    "OWID stopped adding COVID-19 testing observations after June 23, 2022.",
    fontsize=8,
    wrap=True,
)

plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()