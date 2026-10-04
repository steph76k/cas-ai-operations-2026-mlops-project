import json
import os
from pathlib import Path

import hopsworks
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


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

    # Zeit, zwei Modelleingaben und den Zielwert auswählen.
    query = wind_fg.select([
        "time",
        "wind_current",
        "wind_mean_6h",
        "wind_next_hour",
    ])

    feature_view = fs.get_or_create_feature_view(
        name=config["feature_view_name"],
        version=config["feature_view_version"],
        description="Windvorhersage mit Zeit als Hilfsspalte",
        query=query,
        labels=["wind_next_hour"],
        training_helper_columns=["time"],
    )

    print("Feature View ist bereit.")

    # Eingaben und Zielwerte über die Feature View laden.
    features_df, labels_df = feature_view.training_data(
        description="Trainingsdaten für die Windvorhersage in Basel",
        training_helper_columns=True,
    )

    print("Geladene Zeilen:", len(features_df))
    print(features_df.head())
    print(labels_df.head())

    # Eingaben und die zugehörigen Zielwerte zusammenführen.
    if len(features_df) != len(labels_df):
        raise ValueError("Anzahl der Eingaben und Zielwerte stimmt nicht überein.")

    training_df = features_df.reset_index(drop=True).copy()
    training_df["wind_next_hour"] = labels_df["wind_next_hour"].to_numpy()

    # Nur vollständige Beispiele fürs Training verwenden.
    training_df = training_df.dropna(
        subset=["time", "wind_current", "wind_mean_6h", "wind_next_hour"]
    )

    # Chronologisch ordnen, bevor wir Training und Test aufteilen.
    training_df = (
        training_df.sort_values("time")
        .reset_index(drop=True)
    )

    print("Vollständige Trainingsbeispiele:", len(training_df))

    # Frühere 80 % zum Trainieren, spätere 20 % zum Testen.
    split_index = int(len(training_df) * 0.8)

    if split_index == 0 or split_index >= len(training_df):
        raise ValueError("Zu wenige vollständige Beispiele für Training und Test.")

    feature_columns = ["wind_current", "wind_mean_6h"]

    X_train = training_df.iloc[:split_index][feature_columns]
    y_train = training_df.iloc[:split_index]["wind_next_hour"]

    X_test = training_df.iloc[split_index:][feature_columns]
    y_test = training_df.iloc[split_index:]["wind_next_hour"]

    print("Trainingsbeispiele:", len(X_train))
    print("Testbeispiele:", len(X_test))

    # Dasselbe Modell wie im Notebook trainieren.
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    model.fit(X_train, y_train)
    print("Modell trainiert.")

    # Modell auf den zurückgehaltenen Testdaten auswerten.
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    # Vergleich: Der Wind bleibt eine weitere Stunde gleich.
    baseline_predictions = X_test["wind_current"]
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    print(f"Random Forest – MAE: {mae:.2f} km/h")
    print(f"Basisannahme – MAE: {baseline_mae:.2f} km/h")

    # Trainiertes Modell lokal speichern.
    model_dir = project_root / "models" / "wind_basel"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    print("Modell lokal gespeichert:", model_path)

    # Die Model Registry öffnen.
    mr = project.get_model_registry()

    # Modellbeschreibung und Fehlerwert hinterlegen.
    registered_model = mr.python.create_model(
        name=config["model_name"],
        description="Wind eine Stunde voraus, mit zwei Wind-Features",
        metrics={"mae_kmh": float(mae)},
        feature_view=feature_view,
    )

    # Modelldatei nach Hopsworks hochladen.
    registered_model.save(str(model_dir))

    print("Modell in Hopsworks gespeichert.")


if __name__ == "__main__":
    main()