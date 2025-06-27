# unemployment_analysis.py

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
# Replace 'unemployment.csv' with your actual file path
df = pd.read_csv("unemployment.csv")

# Display basic info
print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Optional: Fill or drop missing values
df = df.dropna()

# Basic stats
print("\nSummary Statistics:")
print(df.describe())

# Plot unemployment over time (example assumes 'Date' and 'Unemployment Rate' columns)
df['Date'] = pd.to_datetime(df['Date'])  # convert to datetime
df.sort_values('Date', inplace=True)

plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Unemployment Rate'], marker='o', linestyle='-')
plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Heatmap of correlation
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Boxplot for seasonal pattern (assuming 'Month' exists)
if 'Month' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Month', y='Unemployment Rate', data=df)
    plt.title("Monthly Unemployment Rate Distribution")
    plt.show()

# Analyze Covid-19 impact (example filter)
if 'Year' in df.columns:
    covid_df = df[df['Year'] >= 2020]
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=covid_df, x='Date', y='Unemployment Rate')
    plt.title("Covid-19 Impact on Unemployment")
    plt.show()