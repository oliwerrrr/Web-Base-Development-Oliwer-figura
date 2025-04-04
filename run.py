#!/usr/bin/env python3
"""
Skrypt uruchomieniowy dla przeglądarki wizualizacji.
Służy jako wygodny wrapper do uruchamiania głównego programu.
"""
import os
import sys
import subprocess
import argparse
import time


def check_dependencies():
    """Sprawdza czy wszystkie wymagane zależności są zainstalowane."""
    try:
        import PyQt5
        import PIL
        import numpy
        return True
    except ImportError as e:
        print(f"Kurwa, brakuje zależności: {e}")
        print("Instaluję brakujące zależności...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return False


def generate_samples():
    """Generuje przykładowe dane jeśli nie istnieją."""
    if not os.path.exists("wyniki_test"):
        print("Kurwa, nie ma przykładowych danych! Generuję...")
        subprocess.run([sys.executable, "test_visualization_viewer.py", "--generate-samples"])
        time.sleep(1)  # Daj chwilę na zakończenie generowania
        print("Dane wygenerowane.")
    return "wyniki_test"


def main():
    """Główna funkcja uruchomieniowa."""
    parser = argparse.ArgumentParser(description="Przeglądarka wizualizacji algorytmów")
    parser.add_argument("dir", nargs="?", help="Katalog z wynikami wizualizacji")
    parser.add_argument("--gen-samples", action="store_true", help="Wygeneruj przykładowe dane")
    parser.add_argument("--test", action="store_true", help="Uruchom testy")
    
    args = parser.parse_args()
    
    # Sprawdź zależności
    dependencies_ok = check_dependencies()
    if not dependencies_ok:
        print("Kurwa, zainstalowałem zależności. Uruchom ten skrypt ponownie.")
        return
    
    # Uruchom testy jeśli podano flagę
    if args.test:
        print("Kurwa, odpalam testy...")
        subprocess.run([sys.executable, "test_visualization_viewer.py"])
        return
    
    # Wygeneruj przykładowe dane jeśli podano flagę lub nie podano katalogu
    if args.gen_samples or not args.dir:
        results_dir = generate_samples()
    else:
        results_dir = args.dir
    
    # Uruchom przeglądarkę
    print(f"Kurwa, odpalam przeglądarkę z katalogiem: {results_dir}")
    subprocess.run([sys.executable, "visualization_viewer_qt.py", results_dir])


if __name__ == "__main__":
    main() 