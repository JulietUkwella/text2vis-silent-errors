import pandas as pd
import matplotlib.pyplot as plt

# Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# Convert the date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Filter for Germany during 2021
germany_2021 = df[
    (df["location"] == "Germany") &
    (df["date"] >= "2021-01-01") &
    (df["date"] < "2022-01-01")
].copy()

# Sort data by date
germany_2021 = germany_2021.sort_values("date")

# Plot the 7-day average of new cases
plt.figure(figsize=(12, 6))
plt.plot(
    germany_2021["date"],
    germany_2021["new_cases_smoothed"],
    label="7-day average of new cases"
)

# Add chart labels and formatting
plt.title("7-Day Average of New COVID-19 Cases in Germany (2021)")
plt.xlabel("Date")
plt.ylabel("New cases (7-day average)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Display the chart
plt.show()
