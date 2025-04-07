#!/usr/bin/env python3
"""
Skrypt uruchamiający aplikację przeglądania i analizy wizualizacji.
"""

import os
import sys
import argparse
import webbrowser
from PyQt5.QtWidgets import QApplication

# Import modułów z projektu
import visualization_viewer_qt  # Przeglądarka wizualizacji
try:
    import data_analysis  # Moduł analizy danych
    import data_processor  # Moduł przetwarzania danych
    HAS_DATA_MODULES = True
except ImportError:
    HAS_DATA_MODULES = False

def main():
    """Główna funkcja uruchamiająca aplikację."""
    parser = argparse.ArgumentParser(description="Uruchamia aplikację przeglądania wizualizacji i analizy danych.")
    
    # Argumenty dla przeglądarki wizualizacji
    parser.add_argument('--viewer', action='store_true', help='Uruchamia przeglądarkę wizualizacji')
    parser.add_argument('--dir', type=str, default='wyniki_test', help='Katalog z plikami wizualizacji (domyślnie: wyniki_test)')
    
    # Argumenty dla analizy danych
    if HAS_DATA_MODULES:
        parser.add_argument('--analyze', action='store_true', help='Uruchamia przykładową analizę danych')
        parser.add_argument('--data-file', type=str, help='Ścieżka do pliku CSV z danymi do analizy')
        parser.add_argument('--correlation', action='store_true', help='Wykonuje analizę korelacji')
        parser.add_argument('--regression', action='store_true', help='Wykonuje regresję liniową')
        parser.add_argument('--outliers', action='store_true', help='Wykonuje analizę wartości odstających')
        parser.add_argument('--process', action='store_true', help='Przetwarza dane przed analizą')
    
    args = parser.parse_args()
    
    # Uruchom przeglądarkę wizualizacji, jeśli wybrano --viewer lub nie podano argumentów
    if args.viewer or (not args.viewer and not getattr(args, 'analyze', False)):
        print(f"Uruchamiam przeglądarkę z katalogiem {args.dir}")
        
        # Uruchom aplikację PyQt5
        app = QApplication(sys.argv)
        viewer = visualization_viewer_qt.VisualizationViewer(args.dir)
        viewer.show()
        sys.exit(app.exec_())
    
    # Uruchom analizę danych, jeśli wybrano --analyze
    if HAS_DATA_MODULES and getattr(args, 'analyze', False):
        data_path = args.data_file if args.data_file else "dane/przyklad.csv"
        
        # Sprawdź czy istnieje katalog dane, jeśli nie, utwórz go
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        print(f"Uruchamiam analizę danych z pliku {data_path}")
        
        # Przetwarzanie danych
        if args.process:
            processor = data_processor.DataProcessor()
            
            # Jeśli plik nie istnieje, main() z data_processor.py wygeneruje przykładowe dane
            if not os.path.exists(data_path):
                data_processor.main()
            else:
                data = processor.load_data(data_path)
                
                if data is not None:
                    # Przykładowe przetwarzanie
                    processor.remove_duplicates()
                    processor.handle_missing_values()
                    processor.remove_outliers(method='iqr')
                    processor.scale_features(method='standard')
                    processor.encode_categorical(method='onehot')
                    
                    # Pokaż log transformacji
                    print("\nWykonane transformacje:")
                    for i, transform in enumerate(processor.get_transformation_log(), 1):
                        print(f"{i}. {transform}")
                    
                    # Zapisz przetworzone dane
                    processed_path = "dane/przetworzone.csv"
                    processor.save_data(processed_path)
                    data_path = processed_path
        
        # Analiza danych
        analyzer = data_analysis.DataAnalyzer(data_path=data_path)
        df = analyzer.load_data()
        
        if df is not None:
            # Eksploracja danych
            info = analyzer.data_exploration()
            
            # Wizualizacje EDA
            analyzer.eda_visualizations()
            
            # Analiza korelacji
            if getattr(args, 'correlation', False) or not any([args.correlation, args.regression, args.outliers]):
                print("\nWykonuję analizę korelacji...")
                correlation_matrix = analyzer.correlation_analysis()
            
            # Regresja liniowa
            if getattr(args, 'regression', False) or not any([args.correlation, args.regression, args.outliers]):
                print("\nWykonuję analizę regresji liniowej...")
                # Wybierz kolumny do regresji - pierwsza i druga kolumna numeryczna
                numeric_cols = list(df.select_dtypes(include=['number']).columns)
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    regression_results = analyzer.linear_regression(x_col, y_col)
            
            # Analiza wartości odstających
            if getattr(args, 'outliers', False) or not any([args.correlation, args.regression, args.outliers]):
                print("\nWykonuję analizę wartości odstających...")
                outliers_info = analyzer.detect_outliers()
            
            # Wyświetl ścieżkę do katalogu z wynikami
            print(f"\nAnaliza zakończona! Wyniki zapisano w katalogu: {analyzer.output_dir}")
            
            # Otwórz katalog z wynikami w przeglądarce plików
            try:
                if sys.platform == 'darwin':  # macOS
                    os.system(f'open "{os.path.abspath(analyzer.output_dir)}"')
                elif sys.platform == 'win32':  # Windows
                    os.system(f'explorer "{os.path.abspath(analyzer.output_dir)}"')
                else:  # Linux
                    os.system(f'xdg-open "{os.path.abspath(analyzer.output_dir)}"')
            except Exception as e:
                print(f"Nie udało się otworzyć katalogu z wynikami: {e}")


if __name__ == "__main__":
    main() 