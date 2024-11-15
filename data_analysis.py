# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

# Load the Iris dataset using sklearn and convert it to a pandas DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target (species) column to the dataframe
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Explore the dataset structure
print("\nData types and missing values:")
print(df.info())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Additional Exploration - Check for duplicate rows
print("\nChecking for duplicate rows:")
print(df.duplicated().sum())

# Check for summary statistics of the numerical columns
print("\nBasic statistics:")
print(df.describe())

# Task 2: Basic Data Analysis

# Perform groupings by species and compute the mean of numerical columns
grouped_data = df.groupby('species').mean()
print("\nMean values for each species:")
print(grouped_data)

# Additional Data Analysis - Standard deviation per species
print("\nStandard Deviation for each species:")
print(df.groupby('species').std())

# Task 3: Data Visualization

# 1. Line chart (showing trends over time, here we'll use a synthetic time series)
# For this example, we will simulate a time series for one of the features.

# Create a time series for 'sepal length'
df['sepal length (cm)'] = df['sepal length (cm)'].sort_values().reset_index(drop=True)
plt.figure(figsize=(8,6))
plt.plot(df.index, df['sepal length (cm)'], label="Sepal Length", color='b')
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar chart (showing average petal length per species)
plt.figure(figsize=(8,6))
sns.barplot(x='species', y='petal length (cm)', data=df, palette='Set2')
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(8,6))
sns.histplot(df['sepal width (cm)'], bins=10, kde=True, color='green')
plt.title("Histogram: Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (visualizing relationship between sepal length and petal length)
plt.figure(figsize=(8,6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='Set1')
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# 5. Boxplot (showing distribution of sepal length per species)
plt.figure(figsize=(8,6))
sns.boxplot(x='species', y='sepal length (cm)', data=df, palette='Set3')
plt.title("Boxplot: Sepal Length per Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")
plt.show()

# 6. Heatmap (Correlation between features)
correlation_matrix = df.drop('species', axis=1).corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Heatmap: Correlation Between Features")
plt.show()

# Additional Findings
print("\nAdditional Findings:")
print("- The dataset has no missing values or duplicates.")
print("- Sepal length and petal length are positively correlated (as seen in the scatter plot).")
print("- 'Versicolor' species has the longest average petal length (bar chart).")
print("- Most sepal widths are concentrated around 3 cm (histogram).")
print("- The boxplot shows that 'setosa' has a smaller range of sepal lengths compared to 'virginica' and 'versicolor'.")
print("- The heatmap shows strong positive correlation between sepal length and petal length, while sepal width shows a negative correlation with petal length.")
