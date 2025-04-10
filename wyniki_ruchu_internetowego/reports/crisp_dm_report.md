# Raport analizy wydajności ruchu internetowego wg metodologii CRISP-DM

## 1. Zrozumienie biznesu/problemu (Business Understanding)

### Cele biznesowe

- Zrozumieć wydajność ruchu internetowego, aby zoptymalizować doświadczenia użytkowników
- Identyfikacja czynników wpływających na prędkość pobierania i wysyłania danych
- Ustalenie zależności między różnymi parametrami sieci

### Kryteria sukcesu

- Identyfikacja kluczowych wskaźników wydajności dla ruchu internetowego
- Znalezienie istotnych korelacji między parametrami sieci
- Stworzenie modeli wyjaśniających wpływ różnych czynników na wydajność

## 2. Zrozumienie danych (Data Understanding)

### Opis danych

Analiza wykorzystuje następujące zestawy danych związane z ruchem internetowym:

**Dane dotyczące pobierania:**
- httpgetmt: 724511 wierszy, 14 kolumn
- httpgetmt6: 489 wierszy, 14 kolumn
- dlping: 1246832 wierszy, 9 kolumn
- webget: 9522842 wierszy, 27 kolumn

**Dane dotyczące wysyłania:**
- httppostmt: 722118 wierszy, 14 kolumn
- httppostmt6: 1159 wierszy, 14 kolumn
- ulping: 1260944 wierszy, 9 kolumn

**Dodatkowe dane sieci:**
- dns: 50000 wierszy, 8 kolumn
- ping: 50000 wierszy, 9 kolumn
- traceroute: 50000 wierszy, 13 kolumn
- udplatency: 50000 wierszy, 9 kolumn
- udpjitter: 50000 wierszy, 15 kolumn
- udpcloss: 50000 wierszy, 6 kolumn

### Eksploracja danych

Główne parametry w danych dotyczących pobierania:
- Kolumny w httpgetmt: unit_id, dtime, target, address, fetch_time, bytes_total, bytes_sec, bytes_sec_interval, warmup_time, warmup_bytes, sequence, threads, successes, failures

Główne parametry w danych dotyczących wysyłania:
- Kolumny w httppostmt: unit_id, dtime, target, address, fetch_time, bytes_total, bytes_sec, bytes_sec_interval, warmup_time, warmup_bytes, sequence, threads, successes, failures

### Jakość danych

- W danych httpgetmt brakuje 0 wartości (0.00% wszystkich danych)
- W danych httppostmt brakuje 0 wartości (0.00% wszystkich danych)

## 3. Przygotowanie danych (Data Preparation)

### Wybór danych

- Skoncentrowano się na danych HTTP GET i HTTP POST jako kluczowych wskaźnikach wydajności
- Uwzględniono dodatkowe parametry sieci (ping, jitter) do analizy korelacji

### Czyszczenie danych

- Usunięto wartości odstające (poniżej 5 i powyżej 95 percentyla) dla prędkości pobierania i wysyłania
- Usunięto wiersze z brakującymi wartościami dla kluczowych parametrów
- Agregowano dane według dnia dla analiz czasowych

### Transformacja danych

- Konwersja kolumn czasowych na format datetime
- Agregacja danych według unit_id i dnia dla analizy korelacji

## 4. Modelowanie (Modeling)

### Techniki modelowania

- Wykorzystano regresję liniową do określenia wpływu opóźnienia na prędkość pobierania
- Zastosowano analizę korelacji do identyfikacji zależności między parametrami

### Ocena modeli

- R² score dla modelu wpływu opóźnienia: 0.1142
- MAE dla modelu wpływu opóźnienia: 5195904.7066

## 5. Ewaluacja (Evaluation)

### Wyniki

- Średnia prędkość pobierania: 31,001,327.37 bytes/sec
- Średnia prędkość wysyłania: 9,777,732.40 bytes/sec
- Stosunek prędkości pobierania do wysyłania: 3.17

- Korelacja między opóźnieniem a prędkością pobierania: -0.3354
- Każdy dodatkowy 1ms opóźnienia zmniejsza prędkość pobierania o 27.44 bytes/sec

### Ocena realizacji celów biznesowych

- Zidentyfikowano kluczowe wskaźniki wydajności ruchu internetowego
- Określono zależność między opóźnieniem a prędkością pobierania
- Porównano wydajność pobierania i wysyłania danych

## 6. Wdrożenie (Deployment)

### Plan wdrożenia

- Stworzono kompleksowy raport z wizualizacjami
- Zaimplementowano algorytmy analizy w modułach Pythona
- Przygotowano wyniki w formie gotowej do prezentacji

### Monitoring

- Wyniki analizy można wizualizować za pomocą istniejącej aplikacji run.py
- Zaleca się regularne powtarzanie analizy dla nowych danych

## Podsumowanie

Analiza wydajności ruchu internetowego wykazała istotne różnice między prędkością pobierania i wysyłania danych. Zidentyfikowano wpływ opóźnienia sieci na prędkość pobierania. Wyniki analizy mogą posłużyć do optymalizacji konfiguracji sieci i poprawy doświadczeń użytkowników.
