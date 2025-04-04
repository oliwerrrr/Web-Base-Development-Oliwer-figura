# Przeglądarka Wizualizacji

Program do przeglądania i analizy wizualizacji algorytmów oparty na bibliotece PyQt5.

## Wymagania systemowe

- Python 3.6 lub nowszy
- Biblioteki: PyQt5, Pillow, numpy

## Instalacja

```bash
# Klonowanie repozytorium (opcjonalne)
git clone <url-repozytorium>
cd <katalog-repozytorium>

# Instalacja zależności
pip install -r requirements.txt
```

## Uruchamianie

```bash
# Uruchomienie przeglądarki z domyślnym katalogiem wyników
python visualization_viewer_qt.py

# Uruchomienie przeglądarki z własnym katalogiem wyników
python visualization_viewer_qt.py /ścieżka/do/katalogu/z/wynikami

# Alternatywnie, użyj skryptu uruchomieniowego
python run.py [katalog_z_wynikami]
```

## Generowanie przykładowych danych

Możesz wygenerować przykładowe wizualizacje do testów:

```bash
python test_visualization_viewer.py --generate-samples

# Alternatywnie
python run.py --gen-samples
```

To stworzy katalog `wyniki_test` z kilkoma podfolderami zawierającymi różne typy wizualizacji:
- histogramy
- wykresy
- mapy ciepła

## Funkcje

Aplikacja oferuje następujące funkcje:
- Przeglądanie wizualizacji w drzewiastej strukturze katalogów
- Filtrowanie wizualizacji po nazwie
- Nawigacja między obrazami za pomocą przycisków "Poprzedni" i "Następny"
- Wyświetlanie informacji o obrazie (rozdzielczość, ścieżka)
- Otwieranie obrazu w zewnętrznym programie
- Intuicyjny interfejs z możliwością zmiany rozmiaru

## Dokumentacja algorytmów i wizualizacji

**KURWA, WAŻNE!** Szczegółowe informacje o zaimplementowanych algorytmach i ich wizualizacjach znajdziesz w pliku [README_ALGORITHMS.md](README_ALGORITHMS.md). Dokument zawiera:
- Szczegółowy opis każdego typu wizualizacji
- Wyjaśnienie każdego algorytmu i jego zastosowań
- Instrukcje interpretacji wizualizacji
- Przykłady tworzenia własnych wizualizacji

**Zdecydowanie przeczytaj ten dokument, żeby zrozumieć co kurwa widzisz w wynikach wizualizacji!**

## Testy

Aby uruchomić testy automatyczne:

```bash
python test_visualization_viewer.py

# Alternatywnie
python run.py --test
```

## Struktura projektu

- `visualization_viewer_qt.py` - główny program z interfejsem PyQt5
- `test_visualization_viewer.py` - testy jednostkowe i funkcja generująca przykładowe wizualizacje
- `run.py` - wygodny skrypt uruchomieniowy
- `requirements.txt` - lista wymaganych bibliotek
- `README.md` - główna dokumentacja projektu
- `README_ALGORITHMS.md` - szczegółowa dokumentacja algorytmów i wizualizacji
- `wyniki_test/` - katalog z przykładowymi wizualizacjami (generowany automatycznie)
  - `histogramy/` - przykładowe histogramy (rozkład normalny, Poissona)
  - `wykresy/` - przykładowe wykresy funkcji (sinus, funkcja kwadratowa)
  - `mapy_ciepla/` - przykładowe mapy ciepła (losowa, sin-cos)

## Rozwiązywanie problemów

1. **Problemy z PyQt5**
   - Upewnij się, że masz zainstalowaną bibliotekę PyQt5: `pip install PyQt5`
   - Na systemach Linux może być konieczne zainstalowanie dodatkowych pakietów: `sudo apt-get install python3-pyqt5`

2. **Brak wyświetlania obrazów**
   - Sprawdź czy masz uprawnienia do odczytu plików
   - Sprawdź czy format plików jest obsługiwany (obecnie wspierane są pliki PNG)

3. **Błędy z biblioteką Pillow**
   - Zaktualizuj Pillow: `pip install --upgrade Pillow`

4. **Skrypt run.py nie działa**
   - Sprawdź czy ma uprawnienia do wykonania: `chmod +x run.py`
   - Upewnij się, że masz zainstalowane wszystkie zależności

## Licencja

Ten projekt jest udostępniany na licencji [MIT](LICENSE).

## Kontakt

W razie problemów lub pytań, utwórz Issue lub napisz bezpośrednio do autora. 