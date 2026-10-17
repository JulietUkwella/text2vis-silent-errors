import pandas as pd
import matplotlib.pyplot as plt

# Load the COVID-19 dataset
df = pd.read_csv("owid-covid-data.csv")

# Convert the date column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Filter data for the United States during 2023
us_2023 = df[
    (df["location"] == "United States") &
    (df["date"].dt.year == 2023)
].copy()

# Sort the data by date
us_2023 = us_2023.sort_values("date")

# Create the chart
plt.figure(figsize=(12, 6))
plt.plot(
    us_2023["date"],
    us_2023["new_cases"],
    label="Daily new COVID-19 cases",
    color="blue"
)

# Add chart labels and title
plt.title("Daily New COVID-19 Cases in the United States (2023)")
plt.xlabel("Date")
plt.ylabel("Number of New Cases")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Display the chart
plt.show()