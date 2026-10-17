import json
import os
from pathlib import Path

import hopsworks
import joblib
import pandas as pd
from weather_api import get_weather


def main():
    # Projektordner anhand des Speicherorts dieser Python-Datei finden.
    project_root = Path(__file__).resolve().parents[1]

    # Einstellungen für Basel laden.
    config_path = project_root / "configs" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    print("Inferenz-Pipeline")
    print("Lade Modell und aktuelle Features …")
    project = hopsworks.login(
        host=os.environ["HOPSWORKS_HOST"],
        project=os.environ["HOPSWORKS_PROJECT"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
    )

    fs = project.get_feature_store()

    # Bestehende Feature View aus der Trainings-Pipeline holen.
    feature_view = fs.get_feature_view(
        name=config["feature_view_name"],
        version=config["feature_view_version"],
    )



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

    print("Modellversion:", registered_model.version)


    current_data = get_weather(
        config["open_meteo_forecast_url"],
        params={
            "latitude": config["latitude"],
            "longitude": config["longitude"],
            "current": "wind_speed_10m",
            "wind_speed_unit": "kmh",
            "timezone": "UTC",
        },
    )

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
            "Bitte zuerst die Feature-Pipeline mit --mode current ausführen."
        )


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


    print(f"Aktueller Wind: {prediction_input['wind_current'].iloc[0]:.2f} km/h")
    print(f"Sechsstunden-Durchschnitt: {prediction_input['wind_mean_6h'].iloc[0]:.2f} km/h")

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