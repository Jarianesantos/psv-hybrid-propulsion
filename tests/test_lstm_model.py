import numpy as np
from pathlib import Path

from src.prediction.lstm_model import EnergyDemandLSTM


def test_lstm_model_train_predict_evaluate(tmp_path):
    X = np.random.rand(20, 24, 1).astype(np.float32)
    y = np.random.rand(20, 6, 1).astype(np.float32)

    model_dir = tmp_path / "models"
    model = EnergyDemandLSTM(
        input_window=24,
        forecast_horizon=6,
        lstm_units=16,
        epochs=1,
        batch_size=4,
        model_dir=str(model_dir),
    )
    model.scaler = None

    history = model.train(X, y, save_model=True)
    assert "loss" in history
    assert len(history["loss"]) == 1

    predictions = model.predict(X[:2])
    assert predictions.shape == (2, 6)

    metrics = model.evaluate(X[:2], y[:2])
    assert "mse" in metrics and "mae" in metrics and "rmse" in metrics

    loaded_model = EnergyDemandLSTM(
        input_window=24,
        forecast_horizon=6,
        lstm_units=16,
        epochs=1,
        batch_size=4,
        model_dir=str(model_dir),
    )
    loaded_model.load_model(str(model_dir / "lstm_model.h5"))
    loaded_predictions = loaded_model.predict(X[:2])
    assert loaded_predictions.shape == (2, 6)
