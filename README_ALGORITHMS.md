# Data Analysis Algorithms Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Data Analysis (DA)](#data-analysis-da)
3. [Data Exploration (DE)](#data-exploration-de)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Statistical Correlation](#statistical-correlation)
6. [Linear Regression](#linear-regression)
7. [Quantitative and Qualitative Parameters](#quantitative-and-qualitative-parameters)
8. [Outliers](#outliers)
9. [Quartiles and Interquartile Range (IQR)](#quartiles-and-interquartile-range-iqr)
10. [Heatmaps](#heatmaps)
11. [Data Processing](#data-processing)
12. [Usage Examples](#usage-examples)

## Introduction

This document describes the algorithms and data analysis techniques implemented in the project. The library contains two main modules:

- `data_analysis.py` - contains the `DataAnalyzer` class for analyzing and visualizing data
- `data_processor.py` - contains the `DataProcessor` class for processing and transforming data

## Data Analysis (DA)

Data Analysis is the process of examining, cleaning, transforming, and modeling data to discover useful information, draw conclusions, and support decision-making.

### Implemented Functions:

- Loading data from CSV files
- Basic descriptive statistics
- Data visualization
- Correlation analysis between variables
- Linear regression modeling

## Data Exploration (DE)

Data Exploration is the first step in data analysis, involving familiarization with the data, its structure, and basic properties.

### Implemented Functions:

- Examining data structure (dimensions, columns, data types)
- Detecting missing values
- Basic statistics for numeric and categorical columns
- Generating text reports with exploration results

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis is an approach to analyzing datasets that uses visualization techniques to discover patterns, identify anomalies, and test hypotheses.

### Implemented Functions:

- Histograms for numeric variables
- Box plots for detecting outliers
- Scatter plots for pairs of variables
- Correlation heatmaps

## Statistical Correlation

Statistical correlation is a measure of the relationship between two variables. The most commonly used correlation methods are Pearson, Spearman, and Kendall correlation.

### Implemented Functions:

- Calculating Pearson correlation matrix
- Calculating Spearman correlation matrix
- Calculating Kendall correlation matrix
- Visualizing correlation matrices as heatmaps

## Linear Regression

Linear regression is a statistical modeling technique used to predict the value of a dependent variable based on one or more independent variables.

### Implemented Functions:

- Building a linear regression model
- Splitting data into training and testing sets
- Evaluating the model using various metrics:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² (coefficient of determination)
- Visualizing the regression line and data

## Quantitative and Qualitative Parameters

Data can be divided into quantitative (numeric) and qualitative (categorical) parameters.

### Handling Quantitative Parameters:

- Descriptive statistics: mean, median, standard deviation, etc.
- Data scaling (standardization, normalization)
- Creating polynomial and interaction features

### Handling Qualitative Parameters:

- One-hot encoding
- Label encoding
- Statistics for categorical data (count, frequency)

## Outliers

Outliers are observations that significantly differ from other observations in the data.

### Implemented Detection Methods:

- IQR (Interquartile Range) method
- Z-score method
- Visualizing outliers on box plots

## Quartiles and Interquartile Range (IQR)

Quartiles divide the dataset into four equal parts. The Interquartile Range (IQR) is the difference between the third (Q3) and the first (Q1) quartile.

### Implemented Functions:

- Calculating quartiles (Q1, Q2/median, Q3)
- Calculating interquartile range (IQR)
- Quartile analysis for all numeric variables
- Visualizing quartiles on box plots

## Heatmaps

Heatmaps are graphical representations of data where values are represented by colors. They are particularly useful for visualizing correlation matrices.

### Implemented Functions:

- Creating heatmaps for data matrices
- Customizing colors and labels
- Visualizing correlation matrices as heatmaps

## Data Processing

The `data_processor.py` module contains functions for processing and cleaning data before analysis.

### Implemented Functions:

- Removing duplicates
- Handling missing values
- Scaling numeric features
- Encoding categorical features
- Removing outliers
- Feature selection
- Creating polynomial and interaction features
- Discretizing numeric variables

## Usage Examples

### Data Analysis

```python
from data_analysis import DataAnalyzer

# Initialize analyzer
analyzer = DataAnalyzer(data_path="data/example.csv")

# Load data
df = analyzer.load_data()

# Data exploration
analyzer.data_exploration()

# EDA visualizations
analyzer.eda_visualizations()

# Correlation analysis
correlation_matrix = analyzer.correlation_analysis(method='pearson')

# Linear regression
regression_results = analyzer.linear_regression('x1', 'y')

# Outlier analysis
outliers_info = analyzer.detect_outliers(method='iqr')

# Quartile analysis
quartile_info = analyzer.quartile_analysis()
```

### Data Processing

```python
from data_processor import DataProcessor

# Initialize processor
processor = DataProcessor()

# Load data
data = processor.load_data("data/example.csv")

# Process data
processor.remove_duplicates()
processor.handle_missing_values(strategy='mean')
processor.remove_outliers(method='iqr')
processor.scale_features(method='standard')
processor.encode_categorical(method='onehot')
processor.create_polynomial_features(degree=2)
processor.create_interaction_features()

# Save processed data
processor.save_data("data/processed.csv")
```

---

Documentation prepared for the Web Base Development project. 