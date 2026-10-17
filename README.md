# Windvorhersage für Basel

MLOps-Projektarbeit im CAS AI Operations 2026: Vorhersage der Windgeschwindigkeit eine Stunde im Voraus mit einer Feature-, Trainings- und Inferenz-Pipeline (FTI), Hopsworks Feature Store und Model Registry.

## Daten und Modell

Die historischen und aktuellen Wettermodellwerte stammen von [Open-Meteo](https://open-meteo.com/). Windgeschwindigkeiten werden in km/h verarbeitet, Zeitpunkte in UTC. Die Vorhersage wird in Schweizer Ortszeit angezeigt.

| Spalte | Bedeutung | Verwendung |
|---|---|---|
| `wind_current` | Windgeschwindigkeit | Training: historisch aus Hopsworks; Inferenz: direkt aus der Live-API |
| `wind_mean_6h` | Durchschnitt der betrachteten Stunde und der fünf vorherigen Stunden | Modelleingabe bei Training und Inferenz, jeweils aus Hopsworks |
| `wind_next_hour` | Wind eine Stunde später | Zielwert beim Training; wird bei der Inferenz vorhergesagt |
| `time` | Ereigniszeit in UTC | Zeitliche Trainingsaufteilung und Auswahl aktueller Features |
| `location` | Standortkennung `basel` | Teil des Primärschlüssels in Hopsworks |

Ein `RandomForestRegressor` mit 100 Bäumen, maximaler Tiefe 5 und `random_state=42` verwendet die beiden Windfeatures. Die vollständigen Beispiele werden chronologisch aufgeteilt: 80 % Training, 20 % Test. Modell-MAE und Basis-MAE werden in Hopsworks gespeichert. Die Basisannahme lautet: Der Wind bleibt eine weitere Stunde gleich.

Bisherige Versuche zeigten nur einen geringen Vorteil gegenüber der Basisannahme; der Testfehler belegt keine allgemeine Vorhersagequalität. Die Daten sind modellbasiert, und der aktuelle Wind kann einen neueren Zeitpunkt haben als der stündliche Durchschnitt.

## Die drei Pipelines

- **Feature:** `src/feature_pipeline.py` berechnet und speichert Features in Hopsworks. `--mode historical` verarbeitet den konfigurierten Zeitraum samt Labels; `--mode current` aktualisiert nur die aktuelle Stundenzeile mit zunächst leerem Label. Beide Modi verwenden dieselbe Feature-Berechnung. Primärschlüssel sind `location` und `time`, Ereigniszeit ist `time`.
- **Training:** `src/training_pipeline.py` erstellt oder lädt die Feature View, lädt vollständige Trainingsbeispiele, trainiert und bewertet das Modell und speichert es in der Model Registry.
- **Inferenz:** `src/inference_pipeline.py` lädt das Modell mit der höchsten Versionsnummer und den Durchschnitt für die letzte volle Stunde über die Feature View. Zusammen mit dem aktuellen API-Wind wird die Vorhersage berechnet.

## Einrichtung

Benötigt werden Python 3.11 oder 3.12, uv und ein Hopsworks-Projekt mit API-Schlüssel. Die Skripte können lokal oder in GitHub Codespaces ausgeführt werden; ein eigenes Docker-Setup ist nicht erforderlich.

Im Projektordner die Abhängigkeiten installieren:

```bash
uv sync --locked
```

Eine Datei `.env` im Projektordner anlegen:

```dotenv
HOPSWORKS_API_KEY=DEIN_API_SCHLUESSEL
HOPSWORKS_PROJECT=DEIN_HOPSWORKS_PROJEKT
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
```

Host und Projekt an den eigenen Zugang anpassen. `.env` ist von Git ausgeschlossen; Zugangsdaten dürfen nicht veröffentlicht werden.

## Konfiguration

`configs/config.json` enthält Standort, API-Adressen, Anzeigezeitzone, historischen Zeitraum und die Hopsworks-Namen und Versionen. Konfiguriert sind 2023–2025, Feature Group Version 3 und Feature View Version 4. Für einen getrennten Test können neue Group-, View- und Modellnamen verwendet werden.

Start- und Enddatum gelten als ganze UTC-Tage. Die ersten fünf Stunden ohne vollständigen Durchschnitt und die letzte Stunde ohne Label entfallen; bei vollständigen Daten bleiben 26’298 Beispiele. Das Training verwendet alle vollständigen Beispiele der Feature View, auch bereits früher gespeicherte Zeilen.

`forecast_horizon_hours` muss auf **1** bleiben, da das Label fest die nächste Stunde beschreibt.

## Ausführen

Befehle im Projektordner ausführen und jeweils den erfolgreichen Abschluss abwarten.

**Einmaliger historischer Aufbau und Training:**

```bash
uv run --no-sync --env-file .env python src/feature_pipeline.py --mode historical
uv run --no-sync --env-file .env python src/training_pipeline.py
```

**Aktuelle Features und Vorhersage:**

```bash
uv run --no-sync --env-file .env python src/feature_pipeline.py --mode current
uv run --no-sync --env-file .env python src/inference_pipeline.py
```

Innerhalb derselben Stunde kann der gespeicherte Durchschnitt wiederverwendet werden. Bei einem Stundenwechsel zuerst den aktuellen Feature-Lauf wiederholen. Historische Daten und Training müssen dafür nicht erneut ausgeführt werden. Die Ausführung erfolgt manuell; ein Scheduler ist nicht eingerichtet.

API-Abfragen haben maximal zwei zusätzliche Wiederholungsversuche bei vorübergehenden Fehlern, etwa HTTP 503. Bleibt der Fehler bestehen, bricht der Lauf ab. Bei Hopsworks-Uploads die Speicherbestätigung abwarten; eine Meldung zum abgeschlossenen Lesen bestätigt noch keinen Upload.

## Dateien

- `src/`: drei Pipelines und gemeinsame API-Hilfe `weather_api.py`
- `configs/config.json`: Konfiguration
- `notebooks/01_exploration.ipynb`: schrittweise Erklärung und optionales Nachtragen von Labels
- `pyproject.toml`, `uv.lock`: Abhängigkeiten und festgelegte Versionen
- `models/`: lokale Modellartefakte, nicht in Git
