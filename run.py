#!/usr/bin/env python3
"""
Script for running the visualization and analysis application.
"""

import os
import sys
import argparse
import webbrowser
from PyQt5.QtWidgets import QApplication

# Import project modules
import visualization_viewer_qt  # Visualization viewer
try:
    import data_analysis  # Data analysis module
    import data_processor  # Data processing module
    import internet_traffic_analysis  # Internet traffic analysis module
    HAS_DATA_MODULES = True
except ImportError:
    HAS_DATA_MODULES = False

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Runs the visualization and data analysis application.")
    
    # Arguments for visualization viewer
    parser.add_argument('--viewer', action='store_true', help='Run the visualization viewer')
    parser.add_argument('--dir', type=str, default='results_test', help='Directory with visualization files (default: results_test)')
    
    # Arguments for data analysis
    if HAS_DATA_MODULES:
        parser.add_argument('--analyze', action='store_true', help='Run sample data analysis')
        parser.add_argument('--data-file', type=str, help='Path to CSV file with data for analysis')
        parser.add_argument('--correlation', action='store_true', help='Perform correlation analysis')
        parser.add_argument('--regression', action='store_true', help='Perform linear regression')
        parser.add_argument('--outliers', action='store_true', help='Perform outlier analysis')
        parser.add_argument('--process', action='store_true', help='Process data before analysis')
        
        # Argument for internet traffic analysis
        parser.add_argument('--traffic', action='store_true', help='Run internet traffic analysis')
        parser.add_argument('--data-dir', type=str, default='data', help='Directory with internet traffic data')
        parser.add_argument('--output-dir', type=str, default='internet_traffic_results_2023', help='Directory for analysis results')
    
    args = parser.parse_args()
    
    # Run internet traffic analysis if --traffic was selected
    if HAS_DATA_MODULES and getattr(args, 'traffic', False):
        print(f"Running internet traffic analysis from directory {args.data_dir}")
        
        # Check if data directory exists, if not, display error
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory {args.data_dir} does not exist!")
            return
        
        # Create directory for results if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run traffic analysis
        analyzer = internet_traffic_analysis.InternetTrafficAnalyzer(
            data_dir=args.data_dir, 
            output_dir=args.output_dir
        )
        results = analyzer.run_analysis()
        
        # Display path to results directory
        print(f"\nAnalysis completed! Results saved in directory: {args.output_dir}")
        
        # Open results directory in file browser
        try:
            if sys.platform == 'darwin':  # macOS
                os.system(f'open "{os.path.abspath(args.output_dir)}"')
            elif sys.platform == 'win32':  # Windows
                os.system(f'explorer "{os.path.abspath(args.output_dir)}"')
            else:  # Linux
                os.system(f'xdg-open "{os.path.abspath(args.output_dir)}"')
        except Exception as e:
            print(f"Failed to open results directory: {e}")
        
        # Ask if you want to open the visualization viewer
        print("\nDo you want to open the visualization viewer with the results? (y/n)")
        choice = input().lower()
        
        if choice == 'y' or choice == 'yes':
            print(f"Starting visualization viewer from directory {args.output_dir}")
            app = QApplication(sys.argv)
            viewer = visualization_viewer_qt.VisualizationViewer(args.output_dir)
            viewer.show()
            sys.exit(app.exec_())
        
        return
    
    # Run visualization viewer if --viewer was selected or no arguments were provided
    if args.viewer or (not args.viewer and not getattr(args, 'analyze', False) and not getattr(args, 'traffic', False)):
        print(f"Starting viewer with directory {args.dir}")
        
        # Run PyQt5 application
        app = QApplication(sys.argv)
        viewer = visualization_viewer_qt.VisualizationViewer(args.dir)
        viewer.show()
        sys.exit(app.exec_())
    
    # Run data analysis if --analyze was selected
    if HAS_DATA_MODULES and getattr(args, 'analyze', False):
        data_path = args.data_file if args.data_file else "data/example.csv"
        
        # Check if data directory exists, if not, create it
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        print(f"Starting data analysis from file {data_path}")
        
        # Data processing
        if args.process:
            processor = data_processor.DataProcessor()
            
            # If file doesn't exist, main() from data_processor.py will generate sample data
            if not os.path.exists(data_path):
                data_processor.main()
            else:
                data = processor.load_data(data_path)
                
                if data is not None:
                    # Sample processing
                    processor.remove_duplicates()
                    processor.handle_missing_values()
                    processor.remove_outliers(method='iqr')
                    processor.scale_features(method='standard')
                    processor.encode_categorical(method='onehot')
                    
                    # Show transformation log
                    print("\nTransformations performed:")
                    for i, transform in enumerate(processor.get_transformation_log(), 1):
                        print(f"{i}. {transform}")
                    
                    # Save processed data
                    processed_path = "data/processed.csv"
                    processor.save_data(processed_path)
                    data_path = processed_path
        
        # Data analysis
        analyzer = data_analysis.DataAnalyzer(data_path=data_path)
        df = analyzer.load_data()
        
        if df is not None:
            # Data exploration
            info = analyzer.data_exploration()
            
            # EDA visualizations
            analyzer.eda_visualizations()
            
            # Correlation analysis
            if getattr(args, 'correlation', False) or not any([args.correlation, args.regression, args.outliers]):
                print("\nPerforming correlation analysis...")
                correlation_matrix = analyzer.correlation_analysis()
            
            # Linear regression
            if getattr(args, 'regression', False) or not any([args.correlation, args.regression, args.outliers]):
                print("\nPerforming linear regression analysis...")
                # Select columns for regression - first and second numeric columns
                numeric_cols = list(df.select_dtypes(include=['number']).columns)
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    regression_results = analyzer.linear_regression(x_col, y_col)
            
            # Outlier analysis
            if getattr(args, 'outliers', False) or not any([args.correlation, args.regression, args.outliers]):
                print("\nPerforming outlier analysis...")
                outliers_info = analyzer.detect_outliers()
            
            # Display path to results directory
            print(f"\nAnalysis completed! Results saved in directory: {analyzer.output_dir}")
            
            # Open results directory in file browser
            try:
                if sys.platform == 'darwin':  # macOS
                    os.system(f'open "{os.path.abspath(analyzer.output_dir)}"')
                elif sys.platform == 'win32':  # Windows
                    os.system(f'explorer "{os.path.abspath(analyzer.output_dir)}"')
                else:  # Linux
                    os.system(f'xdg-open "{os.path.abspath(analyzer.output_dir)}"')
            except Exception as e:
                print(f"Failed to open results directory: {e}")


if __name__ == "__main__":
    main() 