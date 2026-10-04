import json
import os
from pathlib import Path

import hopsworks
import joblib
import pandas as pd
import requests


def main():
    # Projektordner anhand des Speicherorts dieser Python-Datei finden.
    project_root = Path(__file__).resolve().parents[2]

    # Einstellungen für Basel laden.
    config_path = project_root / "configs" / "basel.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    project = hopsworks.login(
        host=os.environ["HOPSWORKS_HOST"],
        project=os.environ["HOPSWORKS_PROJECT"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
    )

    fs = project.get_feature_store()
    print("Mit dem Feature Store verbunden.")

    wind_fg = fs.get_feature_group(
        name=config["feature_group_name"],
        version=config["feature_group_version"],
    )

    # Bestehende Feature View aus der Trainings-Pipeline holen.
    feature_view = fs.get_feature_view(
        name=config["feature_view_name"],
        version=config["feature_view_version"],
    )

    print("Feature View für die Inferenz geladen.")


    # Neueste registrierte Version dieses Modells auswählen.
    mr = project.get_model_registry()
    models = mr.get_models(name=config["model_name"])

    if not models:
        raise ValueError("Für diesen Namen ist noch kein Modell registriert.")

    registered_model = max(models, key=lambda model: int(model.version))

    # Modelldatei herunterladen und laden.
    downloaded_model_dir = registered_model.download()
    inference_model = joblib.load(
        Path(downloaded_model_dir) / "model.joblib"
    )

    print("Geladene Modellversion:", registered_model.version)


    current_response = requests.get(
        config["open_meteo_forecast_url"],
        params={
            "latitude": config["latitude"],
            "longitude": config["longitude"],
            "current": "wind_speed_10m",
            "hourly": "wind_speed_10m",
            "past_days": 1,
            "forecast_days": 1,
            "wind_speed_unit": "kmh",
            "timezone": "UTC",
        },
        timeout=30,
    )
    current_response.raise_for_status()
    current_data = current_response.json()

    # Zeitpunkt des aktuellen Windwerts eindeutig als UTC lesen.
    current_time = pd.to_datetime(
        current_data["current"]["time"], utc=True
    )

    # Features für die letzte volle Stunde aus Hopsworks lesen.
    aggregate_time = current_time.floor("h")

    stored_features = feature_view.get_batch_data(
        start_time=aggregate_time.to_pydatetime(),
        end_time=(aggregate_time + pd.Timedelta(hours=1)).to_pydatetime(),
    )

    if len(stored_features) != 1:
        raise ValueError(
            "Genau eine Feature-Zeile für die aktuelle Stunde erwartet. "
            "Bitte zuerst die Feature-Pipeline ausführen."
        )

    print(stored_features)

    # Dieselben Eingabespalten und dieselbe Reihenfolge wie beim Training.
    prediction_input = pd.DataFrame({
        # Echtzeit-Feature direkt aus der API.
        "wind_current": [
            current_data["current"]["wind_speed_10m"]
        ],
        # Aggregiertes Feature aus Hopsworks.
        "wind_mean_6h": [
            stored_features["wind_mean_6h"].iloc[0]
        ],
    })

    if prediction_input.isna().any().any():
        raise ValueError("Mindestens ein Eingabewert fehlt.")

    print(prediction_input)

    # Gespeichertes Modell auf unsere eine Eingabezeile anwenden.
    predicted_wind = float(inference_model.predict(prediction_input)[0])

    # Zielzeitpunkt berechnen und für Basel anzeigen.
    prediction_time = current_time + pd.Timedelta(
        hours=config["forecast_horizon_hours"]
    )
    local_prediction_time = prediction_time.tz_convert(config["timezone"])

    print(
        f"Vorhersage für {local_prediction_time:%d.%m.%Y %H:%M %Z}: "
        f"{predicted_wind:.2f} km/h"
    )


if __name__ == "__main__":
    main()