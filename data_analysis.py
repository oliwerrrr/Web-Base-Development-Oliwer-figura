#!/usr/bin/env python3
"""
Data Analysis Module

This module contains the DataAnalyzer class for analyzing and visualizing data.
The class provides methods for data loading, exploration, visualization, correlation analysis, 
regression modeling, outlier detection, and quartile analysis.

Usage:
    analyzer = DataAnalyzer(data_path="data/example.csv")
    df = analyzer.load_data()
    analyzer.data_exploration()
    analyzer.eda_visualizations()
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    A class for analyzing data, including exploration, visualization, and statistical analysis.
    
    Attributes:
        data_path (str): Path to the data file
        df (DataFrame): Pandas DataFrame containing the data
        results (dict): Dictionary to store analysis results
        figures_path (str): Path to save figures
    """
    
    def __init__(self, data_path=None, df=None, figures_path="figures"):
        """
        Initialize the DataAnalyzer with either a data path or a DataFrame.
        
        Args:
            data_path (str, optional): Path to the data file
            df (DataFrame, optional): Pandas DataFrame
            figures_path (str, optional): Path to save figures
        """
        self.data_path = data_path
        self.df = df
        self.results = {}
        self.figures_path = figures_path
        
        # Create figures directory if it doesn't exist
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)
            logger.info(f"Created directory: {figures_path}")
    
    def load_data(self, data_path=None):
        """
        Load data from a CSV file.
        
        Args:
            data_path (str, optional): Path to the data file
            
        Returns:
            DataFrame: Pandas DataFrame containing the loaded data
        """
        if data_path:
            self.data_path = data_path
        
        if self.data_path:
            try:
                self.df = pd.read_csv(self.data_path)
                logger.info(f"Data loaded from {self.data_path}, shape: {self.df.shape}")
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                return None
        else:
            logger.error("No data path provided")
            return None
        
        return self.df
    
    def data_exploration(self, save_report=True, report_path="exploration_report.txt"):
        """
        Perform data exploration and generate a report.
        
        Args:
            save_report (bool, optional): Whether to save the report to a file
            report_path (str, optional): Path to save the report
            
        Returns:
            dict: Dictionary containing exploration results
        """
        if self.df is None:
            logger.error("No data loaded. Please load data first.")
            return None
        
        results = {}
        
        # Dataset shape
        results["shape"] = self.df.shape
        
        # Data types
        results["dtypes"] = self.df.dtypes.astype(str).to_dict()
        
        # Missing values
        missing_values = self.df.isnull().sum().to_dict()
        results["missing_values"] = missing_values
        results["missing_percentage"] = {col: (missing/self.df.shape[0])*100 for col, missing in missing_values.items()}
        
        # Descriptive statistics
        results["descriptive_stats"] = self.df.describe().to_dict()
        
        # Categorical columns statistics
        cat_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        results["categorical_columns"] = cat_columns
        results["categorical_stats"] = {}
        
        for col in cat_columns:
            results["categorical_stats"][col] = {
                "unique_values": self.df[col].nunique(),
                "value_counts": self.df[col].value_counts().to_dict()
            }
        
        # Numeric columns
        num_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        results["numeric_columns"] = num_columns
        
        # Save report if requested
        if save_report:
            self._save_exploration_report(results, report_path)
        
        # Store results
        self.results["exploration"] = results
        
        return results
    
    def _save_exploration_report(self, results, report_path):
        """
        Save the exploration report to a file.
        
        Args:
            results (dict): Dictionary containing exploration results
            report_path (str): Path to save the report
        """
        try:
            with open(report_path, 'w') as f:
                f.write("# Data Exploration Report\n\n")
                
                f.write("## Dataset Information\n")
                f.write(f"Number of rows: {results['shape'][0]}\n")
                f.write(f"Number of columns: {results['shape'][1]}\n\n")
                
                f.write("## Data Types\n")
                for col, dtype in results["dtypes"].items():
                    f.write(f"{col}: {dtype}\n")
                f.write("\n")
                
                f.write("## Missing Values\n")
                for col, missing in results["missing_values"].items():
                    percentage = results["missing_percentage"][col]
                    f.write(f"{col}: {missing} ({percentage:.2f}%)\n")
                f.write("\n")
                
                f.write("## Descriptive Statistics\n")
                # Write descriptive statistics in a tabular format
                stats_df = pd.DataFrame(results["descriptive_stats"])
                f.write(stats_df.to_string())
                f.write("\n\n")
                
                f.write("## Categorical Columns\n")
                for col in results["categorical_columns"]:
                    f.write(f"### {col}\n")
                    f.write(f"Unique values: {results['categorical_stats'][col]['unique_values']}\n")
                    f.write("Value counts:\n")
                    for val, count in results["categorical_stats"][col]["value_counts"].items():
                        f.write(f"  {val}: {count}\n")
                    f.write("\n")
            
            logger.info(f"Exploration report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving exploration report: {e}")
    
    def eda_visualizations(self, save_figures=True):
        """
        Generate Exploratory Data Analysis visualizations.
        
        Args:
            save_figures (bool, optional): Whether to save the figures
            
        Returns:
            dict: Dictionary containing paths to the saved figures
        """
        if self.df is None:
            logger.error("No data loaded. Please load data first.")
            return None
        
        figures = {}
        
        # Numeric columns
        num_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Histograms for numeric columns
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(num_columns[:9], 1):  # Limit to 9 columns for readability
            plt.subplot(3, 3, i)
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
        
        if save_figures:
            hist_path = os.path.join(self.figures_path, "histograms.png")
            plt.savefig(hist_path)
            figures["histograms"] = hist_path
        
        plt.close()
        
        # Box plots for numeric columns
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(num_columns[:9], 1):
            plt.subplot(3, 3, i)
            sns.boxplot(y=self.df[col])
            plt.title(f"Box Plot of {col}")
            plt.tight_layout()
        
        if save_figures:
            box_path = os.path.join(self.figures_path, "boxplots.png")
            plt.savefig(box_path)
            figures["boxplots"] = box_path
        
        plt.close()
        
        # Correlation heatmap
        if len(num_columns) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.df[num_columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix")
            
            if save_figures:
                corr_path = os.path.join(self.figures_path, "correlation_heatmap.png")
                plt.savefig(corr_path)
                figures["correlation_heatmap"] = corr_path
            
            plt.close()
        
        # Scatter plots for pairs of variables (limited to first 5 for readability)
        if len(num_columns) > 1:
            for i, col1 in enumerate(num_columns[:4]):
                for col2 in num_columns[i+1:5]:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(x=self.df[col1], y=self.df[col2])
                    plt.title(f"Scatter Plot: {col1} vs {col2}")
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    
                    if save_figures:
                        scatter_path = os.path.join(self.figures_path, f"scatter_{col1}_vs_{col2}.png")
                        plt.savefig(scatter_path)
                        figures[f"scatter_{col1}_vs_{col2}"] = scatter_path
                    
                    plt.close()
        
        # Store results
        self.results["eda_visualizations"] = figures
        
        logger.info(f"EDA visualizations completed. {len(figures)} figures generated.")
        return figures
    
    def correlation_analysis(self, method='pearson', threshold=0.0, save_results=True, results_path="correlation_results.json"):
        """
        Perform correlation analysis on numeric columns.
        
        Args:
            method (str, optional): Correlation method ('pearson', 'spearman', or 'kendall')
            threshold (float, optional): Threshold for correlation coefficient filtering
            save_results (bool, optional): Whether to save the results to a file
            results_path (str, optional): Path to save the results
            
        Returns:
            dict: Dictionary containing correlation analysis results
        """
        if self.df is None:
            logger.error("No data loaded. Please load data first.")
            return None
        
        # Get numeric columns
        num_columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(num_columns) < 2:
            logger.warning("Not enough numeric columns for correlation analysis.")
            return None
        
        # Calculate correlation matrix
        corr_matrix = self.df[num_columns].corr(method=method)
        
        # Convert to dictionary for easier serialization
        corr_dict = corr_matrix.to_dict()
        
        # Filter correlations by threshold
        filtered_correlations = []
        for col1 in num_columns:
            for col2 in num_columns:
                if col1 != col2 and abs(corr_matrix.loc[col1, col2]) >= threshold:
                    filtered_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': corr_matrix.loc[col1, col2]
                    })
        
        # Sort by absolute correlation value
        filtered_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Prepare results
        results = {
            'method': method,
            'threshold': threshold,
            'correlation_matrix': corr_dict,
            'filtered_correlations': filtered_correlations
        }
        
        # Save results if requested
        if save_results:
            try:
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Correlation results saved to {results_path}")
            except Exception as e:
                logger.error(f"Error saving correlation results: {e}")
        
        # Store results
        self.results["correlation_analysis"] = results
        
        return results
    
    def linear_regression(self, x_col, y_col, test_size=0.3, save=True):
        """
        Perform linear regression analysis between two variables.
        
        Args:
            x_col (str): Independent variable column name
            y_col (str): Dependent variable column name
            test_size (float): Proportion of data to use for testing (0.0-1.0)
            save (bool): Whether to save results to files
            
        Returns:
            dict: Dictionary with regression results (model, metrics)
        """
        if self.df is None:
            logger.error("No data loaded. Please load data first.")
            return None
        
        if x_col not in self.df.columns or y_col not in self.df.columns:
            logger.error(f"One or both columns ({x_col}, {y_col}) not found in data!")
            return None
        
        # Prepare data
        df = self.df[[x_col, y_col]].dropna()
        X = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'mae': mean_absolute_error(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2': r2_score(y_test, y_test_pred)
        }
        
        # Store results
        results = {
            'model': model,
            'coefficient': model.coef_[0],
            'intercept': model.intercept_,
            'equation': f"{y_col} = {model.coef_[0]:.6f} * {x_col} + {model.intercept_:.6f}",
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        # Create visualization
        if save:
            # Create directory for regression
            reg_dir = os.path.join(self.figures_path, "regression")
            if not os.path.exists(reg_dir):
                os.makedirs(reg_dir)
            
            # Plot regression line
            plt.figure(figsize=(10, 6))
            plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
            plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Testing data')
            
            # Plot regression line
            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)
            plt.plot(x_range, y_pred, color='red', linewidth=2, label='Regression line')
            
            plt.title(f"Linear Regression: {x_col} vs {y_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.legend()
            
            plt.savefig(os.path.join(reg_dir, f"regression_{x_col}_vs_{y_col}.png"))
            plt.close()
            
            # Save metrics to file
            with open(os.path.join(reg_dir, f"regression_results_{x_col}_vs_{y_col}.txt"), "w") as f:
                f.write(f"# Linear Regression: {x_col} vs {y_col}\n\n")
                f.write(f"Equation: {results['equation']}\n\n")
                
                f.write("Training Metrics:\n")
                for metric, value in train_metrics.items():
                    f.write(f"- {metric}: {value:.6f}\n")
                
                f.write("\nTesting Metrics:\n")
                for metric, value in test_metrics.items():
                    f.write(f"- {metric}: {value:.6f}\n")
        
        return results
    
    def detect_outliers(self, cols=None, method='iqr', threshold=1.5, save=True):
        """
        Detect outliers in numeric columns.
        
        Args:
            cols (list): List of columns to analyze. If None, uses all numeric columns.
            method (str): Method for outlier detection ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection (usually 1.5 for IQR, 3 for Z-score)
            save (bool): Whether to save results to files
            
        Returns:
            dict: Dictionary with outlier information
        """
        if self.df is None:
            logger.error("No data loaded. Please load data first.")
            return None
        
        # Use numeric columns if none specified
        if cols is None:
            cols = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
        
        if not cols:
            logger.warning("No numeric columns selected for outlier detection.")
            return None
        
        outliers_info = {}
        
        for col in cols:
            # Skip columns with missing data
            if self.df[col].isnull().all():
                continue
            
            if method == 'iqr':
                # IQR (Interquartile Range) method
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = self.df[
                    (self.df[col] < lower_bound) | 
                    (self.df[col] > upper_bound)
                ]
                
                outliers_info[col] = {
                    'method': 'IQR',
                    'threshold': threshold,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outliers_count': len(outliers),
                    'outliers_percentage': (len(outliers) / len(self.df)) * 100,
                    'outliers_indices': outliers.index.tolist()
                }
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers_mask = z_scores > threshold
                outliers = self.df[col].iloc[outliers_mask]
                
                outliers_info[col] = {
                    'method': 'Z-score',
                    'threshold': threshold,
                    'outliers_count': len(outliers),
                    'outliers_percentage': (len(outliers) / len(self.df[col].dropna())) * 100,
                    'outliers_indices': outliers.index.tolist()
                }
        
        if save:
            # Create directory for outlier analyses
            outliers_dir = os.path.join(self.figures_path, "outliers")
            if not os.path.exists(outliers_dir):
                os.makedirs(outliers_dir)
            
            # Save outlier information
            with open(os.path.join(outliers_dir, f"outliers_{method}.txt"), "w") as f:
                f.write(f"# Outlier Detection - Method: {method}\n\n")
                
                for col, info in outliers_info.items():
                    f.write(f"## Column: {col}\n")
                    f.write(f"Method: {info['method']}\n")
                    
                    if method == 'iqr':
                        f.write(f"Q1: {info['Q1']:.6f}\n")
                        f.write(f"Q3: {info['Q3']:.6f}\n")
                        f.write(f"IQR: {info['IQR']:.6f}\n")
                        f.write(f"Lower Bound: {info['lower_bound']:.6f}\n")
                        f.write(f"Upper Bound: {info['upper_bound']:.6f}\n")
                    else:
                        f.write(f"Z-score Threshold: {info['threshold']:.6f}\n")
                    
                    f.write(f"Outlier Count: {info['outliers_count']}\n")
                    f.write(f"Outlier Percentage: {info['outliers_percentage']:.2f}%\n\n")
            
            # Visualization - boxplot with marked outliers
            for col in cols:
                if col in outliers_info:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x=self.df[col])
                    plt.title(f"Boxplot with Outliers - {col}")
                    plt.xlabel(col)
                    plt.savefig(os.path.join(outliers_dir, f"boxplot_outliers_{col}.png"))
                    plt.close()
        
        return outliers_info
    
    def quartile_analysis(self, cols=None, save=True):
        """
        Perform quartile analysis on numeric columns.
        
        Args:
            cols (list): List of columns to analyze. If None, uses all numeric columns.
            save (bool): Whether to save results to files
            
        Returns:
            dict: Dictionary with quartile information
        """
        if self.df is None:
            logger.error("No data loaded. Please load data first.")
            return None
        
        # Use numeric columns if none specified
        if cols is None:
            cols = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
        
        if not cols:
            logger.warning("No numeric columns selected for quartile analysis.")
            return None
        
        quartile_info = {}
        
        for col in cols:
            # Skip columns with missing data
            if self.df[col].isnull().all():
                continue
            
            # Calculate quartiles
            min_val = self.df[col].min()
            Q1 = self.df[col].quantile(0.25)
            median = self.df[col].quantile(0.5)
            Q3 = self.df[col].quantile(0.75)
            max_val = self.df[col].max()
            IQR = Q3 - Q1
            
            quartile_info[col] = {
                'min': min_val,
                'Q1': Q1,
                'median': median,
                'Q3': Q3,
                'max': max_val,
                'IQR': IQR,
                'range': max_val - min_val
            }
        
        if save:
            # Create directory for quartile analyses
            quartile_dir = os.path.join(self.figures_path, "quartiles")
            if not os.path.exists(quartile_dir):
                os.makedirs(quartile_dir)
            
            # Save quartile information
            with open(os.path.join(quartile_dir, "quartile_analysis.txt"), "w") as f:
                f.write("# Quartile Analysis\n\n")
                
                for col, info in quartile_info.items():
                    f.write(f"## Column: {col}\n")
                    f.write(f"Minimum: {info['min']:.6f}\n")
                    f.write(f"Q1 (25%): {info['Q1']:.6f}\n")
                    f.write(f"Median (50%): {info['median']:.6f}\n")
                    f.write(f"Q3 (75%): {info['Q3']:.6f}\n")
                    f.write(f"Maximum: {info['max']:.6f}\n")
                    f.write(f"IQR (Q3-Q1): {info['IQR']:.6f}\n")
                    f.write(f"Range (max-min): {info['range']:.6f}\n\n")
            
            # Visualization - boxplot
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=self.df[cols])
            plt.title("Boxplot for Numeric Columns - Quartile Analysis")
            plt.xticks(rotation=90)
            plt.savefig(os.path.join(quartile_dir, "boxplot_quartiles.png"))
            plt.close()
        
        return quartile_info
    
    def create_heatmap(self, data_matrix, columns=None, rows=None, title="Heat Map", 
                       cmap="coolwarm", save=True, filename="heatmap.png"):
        """
        Create a customized heatmap.
        
        Args:
            data_matrix (array-like): The data for the heatmap (2D array or DataFrame)
            columns (list): Column labels (optional)
            rows (list): Row labels (optional)
            title (str): Title for the heatmap
            cmap (str): Color map name
            save (bool): Whether to save the figure
            filename (str): Filename to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        plt.figure(figsize=(12, 10))
        
        if isinstance(data_matrix, pd.DataFrame):
            data = data_matrix.values
            if columns is None:
                columns = data_matrix.columns
            if rows is None:
                rows = data_matrix.index
        else:
            data = np.array(data_matrix)
        
        ax = sns.heatmap(data, annot=True, cmap=cmap, fmt=".2f", 
                     xticklabels=columns, yticklabels=rows)
        plt.title(title)
        
        if save:
            heat_dir = os.path.join(self.figures_path, "heatmaps")
            if not os.path.exists(heat_dir):
                os.makedirs(heat_dir)
            plt.savefig(os.path.join(heat_dir, filename))
        
        return plt.gcf()


def main():
    """Przykładowe użycie modułu analizy danych."""
    # Przykład użycia z domyślnym plikiem CSV
    data_path = "dane/przyklad.csv"
    
    # Sprawdź czy plik istnieje, jeśli nie, wygeneruj przykładowe dane
    if not os.path.exists(data_path):
        print(f"Kurwa, plik {data_path} nie istnieje. Generuję przykładowe dane.")
        
        # Utwórz katalog na dane, jeśli nie istnieje
        if not os.path.exists("dane"):
            os.makedirs("dane")
        
        # Generowanie przykładowych danych
        np.random.seed(42)
        n_samples = 100
        
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(5, 2, n_samples)
        y = 3 * x1 + 2 * x2 + np.random.normal(0, 1, n_samples)
        
        categorical = np.random.choice(['A', 'B', 'C'], n_samples)
        
        df = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'y': y,
            'category': categorical
        })
        
        df.to_csv(data_path, index=False)
        print(f"Kurwa, zapisano przykładowe dane do {data_path}")
    
    # Inicjalizacja analizatora
    analyzer = DataAnalyzer(data_path=data_path)
    
    # Wczytanie danych
    df = analyzer.load_data()
    
    if df is not None:
        # Eksploracja danych
        analyzer.data_exploration()
        
        # Wizualizacje EDA
        analyzer.eda_visualizations()
        
        # Analiza korelacji
        analyzer.correlation_analysis()
        
        # Regresja liniowa
        analyzer.linear_regression('x1', 'y')
        
        # Wykrywanie wartości odstających
        analyzer.detect_outliers()
        
        # Analiza kwartylowa
        analyzer.quartile_analysis()
        
        print("Kurwa, analiza zakończona! Wyniki zapisano w katalogu:", analyzer.figures_path)


if __name__ == "__main__":
    main() 