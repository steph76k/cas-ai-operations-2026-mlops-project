import argparse
import json
import os
from pathlib import Path

import hopsworks
import pandas as pd
from weather_api import get_weather


def load_historical_data(config):
    url = config["open_meteo_archive_url"]

    params = {
        "latitude": config["latitude"],
        "longitude": config["longitude"],
        "start_date": config["historical_start_date"],
        "end_date": config["historical_end_date"],
        "hourly": "wind_speed_10m",
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }

    weather_data = get_weather(url, params)

    wind_df = pd.DataFrame(weather_data["hourly"])
    wind_df["time"] = pd.to_datetime(wind_df["time"], utc=True)

    return wind_df


def calculate_features(wind_df):
    wind_df = wind_df.sort_values("time").copy()
    if not wind_df["time"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Die Winddaten enthalten Zeitlücken oder doppelte Stunden.")

    wind_df["wind_current"] = wind_df["wind_speed_10m"]
    wind_df["wind_mean_6h"] = (
        wind_df["wind_current"].rolling(window=6, min_periods=6).mean()
    )
    wind_df["wind_next_hour"] = wind_df["wind_current"].shift(-1)
    return wind_df[["time", "wind_current", "wind_mean_6h", "wind_next_hour"]]


def historical_features(config):
    features = calculate_features(load_historical_data(config)).dropna(
        subset=["wind_current", "wind_mean_6h", "wind_next_hour"]
    ).copy()
    features["location"] = config["location"]
    return features


def current_features(config):
    current_data = get_weather(
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
    )

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

    live_feature_df = calculate_features(last_six_hours.reset_index(names="time"))
    live_feature_df["location"] = config["location"]
    live_feature_df = live_feature_df.tail(1).copy()

    return live_feature_df


def main():
    parser = argparse.ArgumentParser(description="Historische oder aktuelle Windfeatures speichern.")
    parser.add_argument("--mode", required=True, choices=["historical", "current"])
    args = parser.parse_args()

    # Projektordner anhand des Speicherorts dieser Python-Datei finden.
    project_root = Path(__file__).resolve().parents[1]

    # Einstellungen für Basel laden.
    config_path = project_root / "configs" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    print(f"Feature-Pipeline | {args.mode}")

    if args.mode == "historical":
        print(
            f"Zeitraum: {config['historical_start_date']} bis "
            f"{config['historical_end_date']} (UTC)"
        )
        features = historical_features(config)
    else:
        features = current_features(config)

    if args.mode == "historical":
        print(f"Vollständige Beispiele: {len(features):,}".replace(",", "’"))
    else:
        feature_time = features["time"].iloc[0]
        print(f"Durchschnitt für: {feature_time:%d.%m.%Y %H:%M} UTC")
        print(f"Stundenwert: {features['wind_current'].iloc[0]:.2f} km/h")
        print(f"Sechsstunden-Durchschnitt: {features['wind_mean_6h'].iloc[0]:.2f} km/h")

    print("Speichere Features in Hopsworks …")
    project = hopsworks.login(
        host=os.environ["HOPSWORKS_HOST"],
        project=os.environ["HOPSWORKS_PROJECT"],
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
    )

    fs = project.get_feature_store()

    wind_fg = fs.get_or_create_feature_group(
        name=config["feature_group_name"],
        version=config["feature_group_version"],
        description=f"Stündliche Winddaten für {config['location']}",
        primary_key=["location", "time"],
        event_time="time",
        online_enabled=False,
    )

    if args.mode == "historical":
        wind_fg.insert(features, wait=True)
        print("Historische Trainingsbeispiele in Hopsworks gespeichert.")
        return

    # Prüfen, ob diese Stunde bereits einen Zielwert hat.
    existing_df = wind_fg.read()
    existing_df["time"] = pd.to_datetime(existing_df["time"], utc=True)

    existing_row = existing_df.loc[
        (existing_df["location"] == config["location"])
        & (existing_df["time"] == features["time"].iloc[0])
    ]

    if existing_row["wind_next_hour"].notna().any():
        print("Die Stunde ist bereits vollständig gespeichert; sie bleibt unverändert.")
    else:
        wind_fg.insert(features, wait=True)
        print("Aktuelle Feature-Zeile in Hopsworks gespeichert.")


if __name__ == "__main__":
    main()
