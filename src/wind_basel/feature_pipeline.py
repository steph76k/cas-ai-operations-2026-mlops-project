import json
import os
from pathlib import Path

import hopsworks
import pandas as pd
import requests


def load_historical_data(config):
    url = config["open_meteo_archive_url"]

    # Vortag für den Sechsstunden-Durchschnitt mitladen.
    fetch_start = (
        pd.Timestamp(config["historical_start_date"])
        - pd.Timedelta(days=1)
    )

    # Folgetag für den Zielwert der letzten Stunde mitladen.
    fetch_end = (
        pd.Timestamp(config["historical_end_date"])
        + pd.Timedelta(days=1)
    )

    params = {
        "latitude": config["latitude"],
        "longitude": config["longitude"],
        "start_date": fetch_start.strftime("%Y-%m-%d"),
        "end_date": fetch_end.strftime("%Y-%m-%d"),
        "hourly": "wind_speed_10m",
        "wind_speed_unit": "kmh",
        "timezone": config["timezone"],
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    weather_data = response.json()

    wind_df = pd.DataFrame(weather_data["hourly"])
    wind_df["time"] = pd.to_datetime(wind_df["time"])

    return wind_df


def main():
    # Projektordner anhand des Speicherorts dieser Python-Datei finden.
    project_root = Path(__file__).resolve().parents[2]

    # Einstellungen für Basel laden.
    config_path = project_root / "configs" / "basel.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    historical_df = load_historical_data(config)
    print(historical_df.head())

    # Historische Werte zeitlich ordnen.
    historical_df = historical_df.sort_values("time").copy()

    # Prüfen, dass aufeinanderfolgende Zeilen eine Stunde Abstand haben.
    if not historical_df["time"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("Die historischen Daten enthalten Zeitlücken oder doppelte Stunden.")

    # Aktueller Wind und Durchschnitt dieser Stunde plus fünf vorheriger.
    historical_df["wind_current"] = historical_df["wind_speed_10m"]
    historical_df["wind_mean_6h"] = (
        historical_df["wind_current"]
        .rolling(window=6, min_periods=6)
        .mean()
    )

    # Wind eine Stunde später als Zielwert.
    historical_df["wind_next_hour"] = historical_df["wind_current"].shift(-1)

    # Gewünschten Zeitraum auswählen, inklusive des gesamten Endtags.
    start_time = pd.Timestamp(config["historical_start_date"])
    end_time_exclusive = (
        pd.Timestamp(config["historical_end_date"])
        + pd.Timedelta(days=1)
    )

    historical_features = historical_df.loc[
        (historical_df["time"] >= start_time)
        & (historical_df["time"] < end_time_exclusive),
        ["time", "wind_current", "wind_mean_6h", "wind_next_hour"],
    ].copy()

    # Nur vollständige Trainingsbeispiele behalten.
    historical_features = historical_features.dropna(
        subset=["wind_current", "wind_mean_6h", "wind_next_hour"]
    )

    print("Historische Trainingsbeispiele:", len(historical_features))

    # Standort ergänzen.
    historical_features["location"] = config["location"]

    # Lokale Zeit des historischen Abrufs nach UTC umrechnen.
    historical_features["time"] = (
        historical_features["time"]
        .dt.tz_localize(config["timezone"])
        .dt.tz_convert("UTC")
    )

    print(historical_features.head())

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

    # Stundenwerte als Tabelle aufbereiten.
    recent_wind_df = pd.DataFrame(current_data["hourly"])
    recent_wind_df["time"] = pd.to_datetime(
        recent_wind_df["time"], utc=True
    )

    # Nur Zeitpunkte bis zum aktuellen Wert behalten.
    recent_wind_df = (
        recent_wind_df.loc[recent_wind_df["time"] <= current_time]
        .sort_values("time")
        .copy()
    )

    print(recent_wind_df.tail(6))


    # Letzte volle Stunde bis zum aktuellen Datenzeitpunkt.
    aggregate_time = current_time.floor("h")

    # Diese Stunde und die fünf vorherigen Stunden auswählen.
    expected_times = pd.date_range(
        end=aggregate_time,
        periods=6,
        freq="h",
    )

    last_six_hours = (
        recent_wind_df
        .set_index("time")
        .reindex(expected_times)
    )

    # Nur mit sechs vorhandenen Stundenwerten rechnen.
    if last_six_hours["wind_speed_10m"].isna().any():
        raise ValueError("Für den Durchschnitt fehlen Stundenwerte.")

    wind_mean_6h = float(last_six_hours["wind_speed_10m"].mean())

    print(f"Sechsstunden-Durchschnitt: {wind_mean_6h:.2f} km/h")
    print("Stand des Durchschnitts:", aggregate_time)


    live_feature_df = pd.DataFrame({
        "time": [aggregate_time],
        "wind_current": [
            float(last_six_hours.loc[aggregate_time, "wind_speed_10m"])
        ],
        "wind_mean_6h": [wind_mean_6h],
        "wind_next_hour": [float("nan")],
        "location": [config["location"]],
    })

    print(live_feature_df)

    project = hopsworks.login(
        host=os.environ["HOPSWORKS_HOST"],
        project=os.environ["HOPSWORKS_PROJECT"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
    )

    fs = project.get_feature_store()
    print("Mit dem Feature Store verbunden.")

    wind_fg = fs.get_or_create_feature_group(
        name=config["feature_group_name"],
        version=config["feature_group_version"],
        description=f"Stündliche Winddaten für {config['location']}",
        primary_key=["location", "time"],
        event_time="time",
        online_enabled=False,
    )

    # Historische Trainingsbeispiele mit vollständigen Zielwerten speichern.
    wind_fg.insert(historical_features, wait=True)
    print("Historische Trainingsbeispiele in Hopsworks gespeichert.")

    # Prüfen, ob diese Stunde bereits einen Zielwert hat.
    existing_df = wind_fg.read()
    existing_df["time"] = pd.to_datetime(existing_df["time"], utc=True)

    existing_row = existing_df.loc[
        (existing_df["location"] == config["location"])
        & (existing_df["time"] == aggregate_time)
    ]

    if existing_row["wind_next_hour"].notna().any():
        print("Die Stunde ist bereits vollständig gespeichert; sie bleibt unverändert.")
    else:
        wind_fg.insert(live_feature_df, wait=True)
        print("Aktuelle Feature-Zeile in Hopsworks gespeichert.")


if __name__ == "__main__":
    main()