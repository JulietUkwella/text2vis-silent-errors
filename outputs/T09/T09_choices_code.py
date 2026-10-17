import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
file_path = "owid-covid-data.csv"
df = pd.read_csv(file_path)

# 2. Select only the United Kingdom's records
uk = df.loc[
    df["location"] == "United Kingdom",
    ["date", "new_deaths_smoothed"]
].copy()

# 3. Convert columns to suitable data types
uk["date"] = pd.to_datetime(uk["date"], errors="coerce")
uk["new_deaths_smoothed"] = pd.to_numeric(
    uk["new_deaths_smoothed"], errors="coerce"
)

# 4. Remove missing observations and sort chronologically
uk = (
    uk.dropna(subset=["date", "new_deaths_smoothed"])
      .sort_values("date")
)

if uk.empty:
    raise ValueError(
        "No valid United Kingdom death data was found in the CSV."
    )

# 5. Set the date of the first UK vaccination
rollout_date = pd.Timestamp("2020-12-08")

# 6. Create the chart
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    uk["date"],
    uk["new_deaths_smoothed"],
    linewidth=1.8,
    label="Daily deaths (7-day smoothed)"
)

# Mark the start of the vaccination rollout
ax.axvline(
    rollout_date,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="UK vaccination rollout begins"
)

# Add a label near the top of the chart
ax.text(
    rollout_date + pd.Timedelta(days=20),
    0.96,
    "First vaccine: 8 Dec 2020",
    transform=ax.get_xaxis_transform(),
    color="red",
    verticalalignment="top"
)

# 7. Add titles and labels
ax.set_title(
    "Did COVID-19 Deaths Fall After Vaccination Began in the UK?"
)
ax.set_xlabel("Date")
ax.set_ylabel("Daily COVID-19 deaths (7-day average)")

ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()