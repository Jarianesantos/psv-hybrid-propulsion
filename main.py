import sys
from pathlib import Path

from src.prediction.data_preprocessor import DataPreprocessor
from src.prediction.lstm_model import EnergyDemandLSTM


def run_training(data_path: Path) -> None:
    preprocessor = DataPreprocessor(target_column="power_demand")
    df = preprocessor.load_data(str(data_path), date_column="timestamp")
    df = preprocessor.handle_missing_values(df, strategy="ffill")
    df = preprocessor.add_time_features(df, date_column="timestamp")
    df = preprocessor.normalize(df, columns=["power_demand", "hour_sin", "hour_cos", "day_sin", "day_cos"])

    X, y = preprocessor.create_sequences(
        df,
        target_column="power_demand",
        window_size=24,
        horizon=6,
        feature_columns=["power_demand", "hour_sin", "hour_cos", "day_sin", "day_cos"],
    )

    X_train, X_test, y_train, y_test = preprocessor.split_train_test(X, y, train_ratio=0.8)

    model = EnergyDemandLSTM(
        input_window=24,
        forecast_horizon=6,
        lstm_units=64,
        epochs=50,
        batch_size=16,
        model_dir="models",
    )
    model.scaler = preprocessor.scaler

    history = model.train(X_train, y_train, X_val=X_test[: int(len(X_test) * 0.5)], y_val=y_test[: int(len(y_test) * 0.5)])
    metrics = model.evaluate(X_test[int(len(X_test) * 0.5) :], y_test[int(len(y_test) * 0.5) :])

    print("Treinamento finalizado")
    print("Métricas:", metrics)


if __name__ == "__main__":
    data_file = Path("data/energy_demand.csv")
    if not data_file.exists():
        print("Arquivo data/energy_demand.csv não encontrado.")
        print("Coloque um CSV com colunas 'timestamp' e 'power_demand' em data/energy_demand.csv")
        sys.exit(1)

    run_training(data_file)
