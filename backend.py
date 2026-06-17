from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src.prediction.data_preprocessor import DataPreprocessor
from src.prediction.lstm_model import EnergyDemandLSTM

app = Flask(__name__, static_url_path="", static_folder=".")
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
DEFAULT_CSV = ROOT_DIR / "data" / "energy_demand.csv"


def jsonify_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {k: jsonify_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonify_safe(v) for v in value]
    return value


def load_csv_dataframe() -> pd.DataFrame:
    if "file" in request.files and request.files["file"].filename:
        uploaded = request.files["file"]
        return pd.read_csv(BytesIO(uploaded.read()), parse_dates=["timestamp"], dayfirst=False)

    if DEFAULT_CSV.exists():
        return pd.read_csv(DEFAULT_CSV, parse_dates=["timestamp"], dayfirst=False)

    raise FileNotFoundError("Nenhum arquivo CSV enviado e o arquivo data/energy_demand.csv não foi encontrado.")


def build_training_data(df: pd.DataFrame):
    preprocessor = DataPreprocessor(target_column="power_demand")
    df = preprocessor.handle_missing_values(df, strategy="ffill")
    df = preprocessor.add_time_features(df, date_column="timestamp")
    df = preprocessor.normalize(
        df,
        columns=["power_demand", "hour_sin", "hour_cos", "day_sin", "day_cos"],
    )

    X, y = preprocessor.create_sequences(
        df,
        target_column="power_demand",
        window_size=24,
        horizon=6,
        feature_columns=["power_demand", "hour_sin", "hour_cos", "day_sin", "day_cos"],
    )
    if len(X) < 20:
        raise ValueError("Dados insuficientes para treinar. Forneça pelo menos 30 registros.")

    X_train, X_test, y_train, y_test = preprocessor.split_train_test(X, y, train_ratio=0.8)
    return preprocessor, X_train, X_test, y_train, y_test


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(str(ROOT_DIR), "index.html")


@app.route("/train", methods=["POST"])
def train_model():
    try:
        df = load_csv_dataframe()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        preprocessor, X_train, X_test, y_train, y_test = build_training_data(df)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    model = EnergyDemandLSTM(
        input_window=24,
        forecast_horizon=6,
        feature_dim=X_train.shape[2],
        lstm_units=64,
        epochs=20,
        batch_size=16,
        model_dir=str(MODEL_DIR),
    )
    model.scaler = preprocessor.scaler
    model.target_scaler = preprocessor.target_scaler

    history = model.train(
        X_train,
        y_train,
        X_val=X_test[: int(len(X_test) * 0.5)],
        y_val=y_test[: int(len(y_test) * 0.5)],
        save_model=True,
    )
    metrics = model.evaluate(X_test[int(len(X_test) * 0.5) :], y_test[int(len(y_test) * 0.5) :])

    return jsonify(
        {
            "message": "Treinamento concluído.",
            "metrics": jsonify_safe(metrics),
            "history": jsonify_safe(history),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    steps = int(request.form.get("steps", 6))
    if steps <= 0 or steps > 30:
        return jsonify({"error": "steps deve ser um número entre 1 e 30."}), 400

    try:
        df = load_csv_dataframe()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400

    preprocessor = DataPreprocessor(target_column="power_demand")
    df = preprocessor.handle_missing_values(df, strategy="ffill")
    df = preprocessor.add_time_features(df, date_column="timestamp")
    df = preprocessor.normalize(
        df,
        columns=["power_demand", "hour_sin", "hour_cos", "day_sin", "day_cos"],
    )

    X, y = preprocessor.create_sequences(
        df,
        target_column="power_demand",
        window_size=24,
        horizon=6,
        feature_columns=["power_demand", "hour_sin", "hour_cos", "day_sin", "day_cos"],
    )
    if len(X) < 1:
        return jsonify({"error": "Dados insuficientes para gerar previsão."}), 400

    model = EnergyDemandLSTM(
        input_window=24,
        forecast_horizon=6,
        feature_dim=X.shape[2],
        model_dir=str(MODEL_DIR),
    )
    try:
        model.load_model()
    except FileNotFoundError:
        return jsonify({"error": "Modelo não encontrado. Treine o modelo antes de solicitar previsões."}), 400

    model.target_scaler = preprocessor.target_scaler
    predictions = model.predict(X[-1:], steps=steps).flatten()

    return jsonify(
        {
            "message": "Previsão gerada.",
            "steps": steps,
            "predictions": jsonify_safe(predictions),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
