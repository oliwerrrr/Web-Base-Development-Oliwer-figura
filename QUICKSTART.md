# Szybki start - Przeglądarka Wizualizacji

## Kurwa, jak to odpalić?

1. **Instalacja zależności**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generowanie przykładowych danych**:
   ```bash
   python run.py --gen-samples
   ```

3. **Uruchomienie aplikacji**:
   ```bash
   python run.py
   ```

## Skróty klawiaturowe

- **Następny obraz**: `→` lub przycisk "Następny"
- **Poprzedni obraz**: `←` lub przycisk "Poprzedni"
- **Otwórz w programie zewnętrznym**: Przycisk "Otwórz w programie"
- **Wyjście**: `Esc` lub zamknij okno

## Krótki przegląd wizualizacji

### Histogramy
- **histogram_normalny.png**: Rozkład normalny (dzwonowaty)
- **histogram_poisson.png**: Rozkład zdarzeń rzadkich

### Wykresy
- **wykres_sin.png**: Funkcja sinus (falująca)
- **wykres_kwadratowy.png**: Funkcja kwadratowa (parabola)

### Mapy ciepła
- **mapa_ciepla_losowa.png**: Losowe dane
- **mapa_ciepla_sincos.png**: Wzór matematyczny sin*cos

## Co dalej?

1. **Szczegółowa dokumentacja**: Zobacz [README_ALGORITHMS.md](README_ALGORITHMS.md)
2. **Testy aplikacji**: Uruchom `python run.py --test`
3. **Własne dane**: Utwórz własne wizualizacje i otwórz katalog przez `python run.py twój_katalog`

## Kurwa, pomocy?

Jeśli coś nie działa:
1. Sprawdź czy masz zainstalowane wszystkie zależności
2. Upewnij się, że katalog z danymi istnieje
3. Sprawdź uprawnienia do odczytu plików
4. Zajrzyj do pełnej dokumentacji w README.md 