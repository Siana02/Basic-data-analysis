# Basic-data-analysis

# DATA ANALYSIS AND VISUALIZATION WITH PYTHON 

This project demonstrates how to load, explore, analyze, and visualize a dataset using the pandas, matplotlib, and seaborn libraries in Python. In this example, we use the **Iris dataset** from scikit-learn for classification purposes. The code covers loading the data, basic analysis, and various types of visualizations.

## Features:
- **Data Loading & Exploration**: 
  - Load the Iris dataset and explore its structure (data types, missing values, and summary statistics).
  - Perform basic data exploration (checking for duplicates and missing values).

- **Basic Data Analysis**: 
  - Compute basic statistical measures (mean, standard deviation) for numerical features.
  - Group data by species to compute and compare statistical summaries.

- **Data Visualization**:
  - **Line Chart**: Show trends in 'sepal length' over time (synthetic).
  - **Bar Chart**: Visualize the average petal length per species.
  - **Histogram**: Show the distribution of sepal width across all data points.
  - **Scatter Plot**: Visualize the relationship between sepal length and petal length, with species differentiation.
  - **Boxplot**: Show the distribution of sepal length across species.
  - **Heatmap**: Visualize correlations between features in the dataset.

## Key Observations:
- The dataset contains no missing values or duplicates.
- Sepal length and petal length are positively correlated.
- 'Versicolor' has the longest average petal length, while 'Setosa' has a smaller range of sepal lengths.
- The heatmap indicates strong positive correlations between sepal length and petal length.

## Requirements:
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run:
1. Install the required libraries: `pip install pandas numpy matplotlib seaborn scikit-learn`
2. Run the script or Jupyter notebook to see the data analysis and visualizations.
