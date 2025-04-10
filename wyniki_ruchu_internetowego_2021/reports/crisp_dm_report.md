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
- httpget: 45780 wierszy, 15 kolumn
- httpgetmt: 1199755 wierszy, 15 kolumn
- httpgetmt6: 3482 wierszy, 15 kolumn
- dlping: 2530196 wierszy, 10 kolumn
- webget: 11396961 wierszy, 28 kolumn

**Dane dotyczące wysyłania:**
- httppost: 45629 wierszy, 15 kolumn
- httppostmt: 1194194 wierszy, 15 kolumn
- httppostmt6: 3469 wierszy, 15 kolumn
- ulping: 2534862 wierszy, 10 kolumn

**Dodatkowe dane sieci:**
- dns: 50000 wierszy, 9 kolumn
- ping: 50000 wierszy, 10 kolumn
- udplatency: 50000 wierszy, 10 kolumn
- udpjitter: 50000 wierszy, 16 kolumn
- udpcloss: 50000 wierszy, 7 kolumn

### Eksploracja danych

Główne parametry w danych dotyczących pobierania:
- Kolumny w httpgetmt: unit_id, dtime, target, address, fetch_time, bytes_total, bytes_sec, bytes_sec_interval, warmup_time, warmup_bytes, sequence, threads, successes, failures, location_id

Główne parametry w danych dotyczących wysyłania:
- Kolumny w httppostmt: unit_id, dtime, target, address, fetch_time, bytes_total, bytes_sec, bytes_sec_interval, warmup_time, warmup_bytes, sequence, threads, successes, failures, location_id

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

- R² score dla modelu wpływu opóźnienia: 0.0250
- MAE dla modelu wpływu opóźnienia: 7126082.5791

## 5. Ewaluacja (Evaluation)

### Wyniki

- Średnia prędkość pobierania: 12,362,686.32 bytes/sec
- Średnia prędkość wysyłania: 2,937,953.23 bytes/sec
- Stosunek prędkości pobierania do wysyłania: 4.21

- Korelacja między opóźnieniem a prędkością pobierania: -0.1794
- Każdy dodatkowy 1ms opóźnienia zmniejsza prędkość pobierania o 23.77 bytes/sec

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
