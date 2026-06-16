import os
from pathlib import Path
from typing import Optional

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib


class EnergyDemandLSTM:
    """Modelo LSTM para previsão de demanda energética híbrido-elétrica."""

    def __init__(
        self,
        input_window: int = 24,
        forecast_horizon: int = 6,
        feature_dim: int = 1,
        lstm_units: int = 128,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        model_dir: str = "models",
    ):
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = None
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential(
            [
                Bidirectional(
                    LSTM(self.lstm_units, return_sequences=True),
                    input_shape=(self.input_window, self.feature_dim),
                ),
                Dropout(self.dropout_rate),
                Bidirectional(LSTM(self.lstm_units, return_sequences=False)),
                Dropout(self.dropout_rate),
                Dense(self.forecast_horizon),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse", metrics=["mae"])
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_model: bool = True,
    ) -> dict:
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=self.model_dir / "lstm_model.h5",
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        if save_model:
            self.model.save(self.model_dir / "lstm_model.h5")
            if self.scaler is not None:
                joblib.dump(self.scaler, self.model_dir / "scaler.pkl")

        return history.history

    def predict(self, X: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        if steps is None:
            steps = self.forecast_horizon

        predictions = self.model.predict(X)
        if steps > self.forecast_horizon:
            predictions = self._recursive_forecast(X, steps)

        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        return predictions

    def _recursive_forecast(self, X: np.ndarray, steps: int) -> np.ndarray:
        all_predictions = []
        current_X = X.copy()

        while current_X.shape[1] >= self.input_window and len(all_predictions) * self.forecast_horizon < steps:
            pred = self.model.predict(current_X)
            all_predictions.append(pred)
            if current_X.ndim == 3:
                last_window = current_X[:, -self.forecast_horizon :, :]
                pred_window = pred.reshape(-1, self.forecast_horizon, 1)
                current_X = np.concatenate([last_window, pred_window], axis=1)
            else:
                break

        all_predictions = np.concatenate(all_predictions, axis=1)
        return all_predictions[:, :steps]

    def load_model(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = self.model_dir / "lstm_model.h5"
        if Path(model_path).exists():
            self.model = load_model(model_path)
            scaler_path = self.model_dir / "scaler.pkl"
            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        predictions = self.model.predict(X_test)
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
            y_test = self.scaler.inverse_transform(y_test)

        mse = np.mean(np.square(y_test - predictions))
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(mse)
        return {"mse": mse, "mae": mae, "rmse": rmse}
