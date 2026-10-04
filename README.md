# Windvorhersage für Basel

MLOps-Projektarbeit im CAS AI Operations 2026. Das Projekt demonstriert eine Feature-, Trainings- und Inferenz-Pipeline mit dem Hopsworks Feature Store. Ziel ist die Vorhersage der Windgeschwindigkeit in Basel eine Stunde im Voraus. Im Mittelpunkt steht der nachvollziehbare Ablauf, nicht eine möglichst hohe Vorhersagequalität.

## Daten, Features und Modell

Die Wetterdaten stammen von Open-Meteo:

- Historische Stundenwerte: https://archive-api.open-meteo.com/v1/archive
- Aktuelle Wettermodellwerte: https://api.open-meteo.com/v1/forecast

Die Werte sind modellbasierte Wetterdaten, keine direkten Messungen einer lokalen Wetterstation. Historische und aktuelle Daten sind nicht notwendigerweise aus demselben Modellstand. Alle Windgeschwindigkeiten werden in km/h verarbeitet.

| Spalte | Bedeutung |
|---|---|
| `wind_current` | Windgeschwindigkeit zum jeweiligen Zeitpunkt; bei der Inferenz direkt aus der aktuellen API-Antwort |
| `wind_mean_6h` | Aggregiertes Feature: Mittelwert aus der betrachteten Stunde und den fünf vorherigen Stunden |
| `wind_next_hour` | Trainingslabel: Windgeschwindigkeit eine Stunde später |
| `time` | Ereigniszeit, in Hopsworks als UTC gespeichert |
| `location` | Standortkennung, hier `basel` |

Das Modell ist ein `RandomForestRegressor` mit 100 Bäumen, maximaler Tiefe 5 und `random_state=42`. Nach zeitlicher Sortierung dienen die ersten 80 % der vollständigen Beispiele zum Training und die übrigen 20 % zum Testen. Die Zeitspalte ist eine Hilfsspalte und keine Modelleingabe. Bewertet wird der mittlere absolute Fehler (MAE) in km/h. Als Vergleich dient die Annahme, dass der Wind eine weitere Stunde unverändert bleibt.

## Die drei Pipelines

### 1. Feature-Pipeline

`src/wind_basel/feature_pipeline.py` lädt den konfigurierten historischen Zeitraum. Ein zusätzlicher Vortag liefert die Werte für den ersten Durchschnitt; ein zusätzlicher Folgetag liefert den letzten Zielwert. Nach der Feature-Berechnung wird auf den gewünschten Zeitraum begrenzt, unvollständige Beispiele werden entfernt und die Zeiten werden nach UTC umgerechnet.

Die vollständigen Trainingsbeispiele werden in einer Feature Group gespeichert. Primärschlüssel sind `location` und `time`, Ereigniszeit ist `time`. Der Online Feature Store ist nicht aktiviert.

Zusätzlich lädt die Pipeline aktuelle Stundenwerte, berechnet den Durchschnitt für die letzte volle Stunde und speichert eine aktuelle Feature-Zeile mit zunächst leerem Label. Wenn diese Stunde bereits einen Zielwert besitzt, überspringt der aktuelle Upload die gesamte Zeile. Vorhandene historische Zeilen werden anhand ihres Schlüssels aktualisiert, neue ergänzt.

### 2. Trainings-Pipeline

`src/wind_basel/training_pipeline.py` lädt die Feature Group und holt oder erstellt die konfigurierte Feature View. Diese wählt beide Modelleingaben, das Label und die Zeit als Trainingshilfsspalte aus. Die Daten werden über die Feature View geladen. Nach Ausschluss unvollständiger Beispiele erfolgen die zeitliche Aufteilung, das Training und der Vergleich mit der Basisannahme.

Das Modell wird mit Joblib unter `models/wind_basel/model.joblib` gespeichert und anschliessend in der Hopsworks Model Registry registriert, einschliesslich MAE und Verknüpfung zur Feature View. Jeder erfolgreiche Trainingslauf legt eine neue Modellversion an. Der verwendete Registry-Upload kann die lokale Modelldatei verschieben; die Inferenz lädt das Modell deshalb aus der Registry.

### 3. Inferenz-Pipeline

`src/wind_basel/inference_pipeline.py` lädt die konfigurierte Feature View und das registrierte Modell mit der höchsten Versionsnummer. Die gewählte Version wird ausgegeben; dies ist nicht automatisch das Modell mit dem besten Fehlerwert.

Die Pipeline ruft den aktuellen Wind direkt bei Open-Meteo ab und lädt über die Feature View die Feature-Zeile für die letzte volle Stunde. Sie erwartet genau eine Zeile. Den Durchschnitt daraus kombiniert sie mit dem aktuellen API-Wind, prüft auf fehlende Eingaben und berechnet die Vorhersage. Der Zielzeitpunkt wird in Basler Ortszeit angezeigt.

## Einrichtung

Voraussetzungen sind ein Hopsworks-Account mit Projekt und API-Schlüssel, Internetzugang sowie Python und uv. Lokal wurde Python 3.12 verwendet; `pyproject.toml` erlaubt Python >=3.11,<3.13. Die Abhängigkeiten stehen in `pyproject.toml`, die aufgelösten Versionen in `uv.lock`.

Nach dem Klonen des Repositorys im Projektordner:

```bash
uv sync --locked
```

Eine lokale Datei `.env` neben `pyproject.toml` anlegen:

```dotenv
HOPSWORKS_API_KEY=DEIN_API_SCHLUESSEL
HOPSWORKS_PROJECT=DEIN_HOPSWORKS_PROJEKT
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
```

Projekt und Host müssen zum eigenen Hopsworks-Zugang passen. Die `.env`-Datei ist durch `.gitignore` ausgeschlossen und darf nicht veröffentlicht werden. Die Startbefehle unten laden sie ausdrücklich über uv.

## Konfiguration

`configs/basel.json` enthält Standort, Koordinaten, Zeitzone, historischen Zeitraum sowie Namen und Versionen von Feature Group und Feature View und den Modellnamen. Der aktuelle Übungszeitraum umfasst den 1. bis 7. Juni 2025: bei vollständigen Daten 168 stündliche Trainingsbeispiele.

Die bestehende Feature Group verwendet Version 1, die Feature View Version 2. Die Nummern sind unabhängig voneinander. Eine neue Feature View könnte auch bei Version 1 beginnen. Alle Pipelines müssen dieselbe Konfiguration verwenden. Eine bestehende View wird durch `get_or_create` nicht mit einer geänderten Definition überschrieben.

**Der Vorhersagehorizont muss derzeit auf 1 stehen:** Die Label-Erzeugung verwendet fest die nächste Stundenzeile. Eine Änderung allein in der Konfiguration würde nur die Beschriftung der Vorhersage ändern, nicht das trainierte Ziel.

## Ausführen

Alle Befehle im Projektordner ausführen. Für den ersten vollständigen Lauf gilt diese Reihenfolge; jeweils den erfolgreichen Abschluss abwarten:

```bash
uv run --no-sync --env-file .env python src/wind_basel/feature_pipeline.py
uv run --no-sync --env-file .env python src/wind_basel/training_pipeline.py
uv run --no-sync --env-file .env python src/wind_basel/inference_pipeline.py
```

Für weitere Vorhersagen genügt bei vorhandenem Modell zuerst die Feature-Pipeline und danach die Inferenz-Pipeline. Erneutes Training ist nicht für jede Vorhersage nötig. Die Feature-Pipeline muss eine Zeile für dieselbe volle Stunde bereitgestellt haben, die die Inferenz anfragt. Bei einem Stundenwechsel gegebenenfalls die Feature-Pipeline erneut ausführen.

Erwartete Ausgaben sind die Anzahl historischer Beispiele und Upload-Bestätigungen, beim Training die Fehlerwerte und die Speicherbestätigung, bei der Inferenz Modellversion, Eingabewerte und eine Windvorhersage mit Zeitpunkt. Uploads können etwas dauern. Die Meldung zum abgeschlossenen Lesen allein bedeutet noch nicht, dass der anschliessende Upload beendet ist.

Die direkten Dateiaufrufe umgehen ein lokal beobachtetes Problem beim Import des installierten Projektpakets. `--no-sync` verwendet die bereits eingerichtete Umgebung; nach Änderungen an Abhängigkeiten zuerst erneut synchronisieren.

## Stand und Einschränkungen

- Die drei Pipelines wurden in der lokalen Projektumgebung nacheinander ausgeführt. Training und Inferenz wurden vom Anwender als erfolgreich bestätigt. Dies ersetzt keinen Test auf einem zweiten Rechner oder mit einem neuen Hopsworks-Projekt.
- Die kleine historische Übungswoche deckt keine Jahreszeiten ab. Der Testfehler ist kein belastbarer Nachweis allgemeiner Vorhersagequalität.
- Das Training verwendet stündliche historische Werte. Der aktuelle API-Wind kann einen Viertelstundenzeitpunkt haben, während der Durchschnitt an der letzten vollen Stunde endet. Diese Abweichung ist eine Vereinfachung des Prototyps.
- Die Trainings-Pipeline verwendet alle vollständigen Beispiele der Feature View, nicht nur den konfigurierten historischen Zeitraum. Auch das im Notebook einmal nachgetragene Beispiel kann deshalb enthalten sein.
- Historische Zeitwerte werden zunächst in lokaler Zeit verarbeitet. Zeiträume über eine Sommer-/Winterzeitumstellung sind mit diesem Ablauf nicht robust abgedeckt. Der gewählte Juni-Zeitraum enthält keine Umstellung.
- Die Inferenz ist auf den einzelnen Standort Basel ausgelegt. Ein gemeinsamer Datenbestand für mehrere Standorte würde einen zusätzlichen Standortfilter erfordern.
- Bei jedem Feature-Lauf wird auch der feste historische Zeitraum erneut abgerufen und hochgeladen. Ein gleitendes Fenster, automatische Ausführung und laufendes Nachtragen späterer Zielwerte sind nicht implementiert.
- Das Nachtragen eines Zielwerts wurde im Notebook als optionaler Ablauf ausprobiert. Es gehört nicht zu den drei Skripten und ist für den Mindestumfang der Aufgabe nicht erforderlich.
- Die Prüfung vor dem aktuellen Upload schützt vorhandene Labels bei nacheinander ausgeführten Läufen. Parallele schreibende Läufe sind nicht abgesichert.
- Lokal wurde mit Hopsworks-Client 5.0.9 und Backend 5.1.0 eine Kompatibilitätswarnung angezeigt. Die beschriebenen Läufe funktionierten trotzdem; eine allgemeine Kompatibilität ist damit nicht garantiert.
- Unter macOS war die Verweisdatei der editierbaren Paketinstallation wiederholt als versteckt markiert. Dadurch scheiterte der Modulaufruf. Die Ursache ist ungeklärt; die oben dokumentierten direkten Dateiaufrufe funktionierten.

## Repository-Struktur

- `src/wind_basel/`: die drei ausführbaren Pipelines
- `configs/basel.json`: gemeinsame Anwendungseinstellungen
- `notebooks/01_exploration.ipynb`: schrittweise Erkundung und Erklärung, inklusive optionalem Nachtragen
- `pyproject.toml` und `uv.lock`: Python-Anforderungen und Abhängigkeiten
- `models/`: lokale Modellartefakte, nicht in Git
- `docs/`: vorgesehene Ablage für ergänzende Dokumentation

Für die Abgabe wird der Link zum GitHub-Repository mit den drei Pipelines und dieser README benötigt. API-Schlüssel gehören weder in den Code noch in Notebook-Ausgaben oder Git.
