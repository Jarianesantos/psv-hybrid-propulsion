import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import joblib
import os
from pathlib import Path

class EnergyDemandLSTM:
    """
    Modelo LSTM para previsão de demanda energética.
    """

    def __init__(
        self,
        input_window: int = 24,
        forecast_horizon: int = 6,
        lstm_units: int = 128,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        model_dir: str = "models"
    ):
        """
        Inicializa o modelo LSTM.

        Args:
            input_window (int): Janela de entrada (número de timesteps).
            forecast_horizon (int): Horizonte de previsão (número de passos à frente).
            lstm_units (int): Número de unidades LSTM.
            dropout_rate (float): Taxa de dropout.
            learning_rate (float): Taxa de aprendizado.
            epochs (int): Número de épocas de treinamento.
            batch_size (int): Tamanho do batch.
            model_dir (str): Diretório para salvar o modelo.
        """
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.scaler: Optional[MinMaxScaler] = None
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        """Constrói o modelo LSTM."""
        model = Sequential([
            Bidirectional(
                LSTM(self.lstm_units, return_sequences=True),
                input_shape=(self.input_window, 1)
            ),
            Dropout(self.dropout_rate),
            Bidirectional(LSTM(self.lstm_units, return_sequences=False)),
            Dropout(self.dropout_rate),
            Dense(self.forecast_horizon)
        ])

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def _create_dataset(
        self,
        data: np.ndarray,
        window_size: int,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria datasets de entrada e saída para o LSTM.

        Args:
            data (np.ndarray): Série temporal de dados.
            window_size (int): Tamanho da janela de entrada.
            horizon (int): Horizonte de previsão.

        Returns:
            Tuple[np.ndarray, np.ndarray]: X (entrada), y (saída).
        """
        X, y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size:i + window_size + horizon])
        return np.array(X), np.array(y)

    def preprocess_data(
        self,
        data: pd.DataFrame,
        target_column: str = "power_demand"
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Pré-processa os dados para treinamento.

        Args:
            data (pd.DataFrame): Dados brutos.
            target_column (str): Coluna alvo para previsão.

        Returns:
            Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
                X (entrada), y (saída), scaler.
        """
        # Extrair a coluna alvo
        target = data[target_column].values.reshape(-1, 1)

        # Normalizar os dados
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(target)

        # Criar datasets
        X, y = self._create_dataset(
            scaled_data,
            self.input_window,
            self.forecast_horizon
        )

        return X, y, self.scaler

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_model: bool = True
    ) -> dict:
        """
        Treina o modelo LSTM.

        Args:
            X_train (np.ndarray): Dados de entrada para treinamento.
            y_train (np.ndarray): Dados de saída para treinamento.
            X_val (np.ndarray): Dados de entrada para validação.
            y_val (np.ndarray): Dados de saída para validação.
            save_model (bool): Se deve salvar o modelo treinado.

        Returns:
            dict: Histórico de treinamento.
        """
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=self.model_dir / "lstm_model.h5",
                monitor="val_loss",
                save_best_only=True
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )

        if save_model:
            self.model.save(self.model_dir / "lstm_model.h5")
            if self.scaler:
                joblib.dump(self.scaler, self.model_dir / "scaler.pkl")

        return history.history

    def predict(
        self,
        X: np.ndarray,
        steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Faz previsões com o modelo treinado.

        Args:
            X (np.ndarray): Dados de entrada para previsão.
            steps (int): Número de passos à frente para prever (usando previsão recursiva).

        Returns:
            np.ndarray: Previsões.
        """
        if steps is None:
            steps = self.forecast_horizon

        # Previsão direta
        predictions = self.model.predict(X)

        # Se steps > forecast_horizon, faz previsão recursiva
        if steps > self.forecast_horizon:
            predictions = self._recursive_forecast(X, steps)

        # Desnormaliza as previsões
        if self.scaler:
            predictions = self.scaler.inverse_transform(predictions)

        return predictions

    def _recursive_forecast(
        self,
        X: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Faz previsão recursiva para horizontes maiores que forecast_horizon.

        Args:
            X (np.ndarray): Dados de entrada.
            steps (int): Número total de passos à frente.

        Returns:
            np.ndarray: Previsões recursivas.
        """
        all_predictions = []
        current_X = X.copy()

        for _ in range(steps // self.forecast_horizon + 1):
            # Previsão para o horizonte
            pred = self.model.predict(current_X)
            all_predictions.append(pred)

            # Atualiza X para a próxima previsão
            if len(current_X.shape) == 3:
                # Se X for 3D (batch, timesteps, features)
                last_window = current_X[:, -self.forecast_horizon:, :]
                new_X = np.concatenate([last_window, pred.reshape(-1, self.forecast_horizon, 1)], axis=1)
            else:
                # Se X for 2D (timesteps, features)
                last_window = current_X[:, -self.forecast_horizon:]
                new_X = np.concatenate([last_window, pred.reshape(-1, self.forecast_horizon)], axis=1)

            current_X = new_X.reshape(1, *new_X.shape)

        # Combina todas as previsões
        all_predictions = np.concatenate(all_predictions, axis=1)
        return all_predictions[:, :steps]

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Carrega um modelo treinado.

        Args:
            model_path (str): Caminho para o modelo. Se None, usa o padrão.
        """
        if model_path is None:
            model_path = self.model_dir / "lstm_model.h5"

        if os.path.exists(model_path):
            self.model = Sequential.load_model(model_path)
            print(f"✅ Modelo carregado de: {model_path}")

            # Carrega o scaler
            scaler_path = self.model_dir / "scaler.pkl"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Scaler carregado de: {scaler_path}")
        else:
            print(f"❌ Modelo não encontrado em: {model_path}")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Avalia o modelo em dados de teste.

        Args:
            X_test (np.ndarray): Dados de entrada para teste.
            y_test (np.ndarray): Dados de saída para teste.

        Returns:
            dict: Métricas de avaliação (MSE, MAE, RMSE).
        """
        predictions = self.model.predict(X_test)

        # Desnormaliza
        if self.scaler:
            predictions = self.scaler.inverse_transform(predictions)
            y_test = self.scaler.inverse_transform(y_test)

        mse = np.mean(np.square(y_test - predictions))
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(mse)

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse
        }




📄 src/prediction/data_preprocessing.py
python
Copiar

"""
Pré-processamento de dados para o modelo LSTM.
Autor: Jariane Santos
Data: 2026
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class DataPreprocessor:
    """
    Classe para pré-processamento de dados de demanda energética.
    """

    def __init__(self, target_column: str = "power_demand"):
        """
        Inicializa o pré-processador.

        Args:
            target_column (str): Coluna alvo para previsão.
        """
        self.target_column = target_column
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(
        self,
        file_path: str,
        date_column: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Carrega dados de um arquivo CSV.

        Args:
            file_path (str): Caminho para o arquivo CSV.
            date_column (str): Nome da coluna de data/hora.

        Returns:
            pd.DataFrame: Dados carregados.
        """
        df = pd.read_csv(file_path)

        # Converte coluna de data/hora para datetime
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])

        # Ordena por data/hora
        if date_column in df.columns:
            df = df.sort_values(date_column)

        return df

    def add_time_features(
        self,
        df: pd.DataFrame,
        date_column: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Adiciona features temporais (hora, dia da semana, etc.).

        Args:
            df (pd.DataFrame): DataFrame com dados.
            date_column (str): Nome da coluna de data/hora.

        Returns:
            pd.DataFrame: DataFrame com features temporais.
        """
        if date_column not in df.columns:
            return df

        df = df.copy()

        # Extrair features temporais
        df["hour"] = df[date_column].dt.hour
        df["day_of_week"] = df[date_column].dt.dayofweek
        df["day_of_month"] = df[date_column].dt.day
        df["month"] = df[date_column].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Seno e cosseno para ciclos temporais
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 30)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 30)

        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "ffill"
    ) -> pd.DataFrame:
        """
        Trata valores ausentes no DataFrame.

        Args:
            df (pd.DataFrame): DataFrame com dados.
            strategy (str): Estratégia para preencher valores ausentes ("ffill", "bfill", "mean").

        Returns:
            pd.DataFrame: DataFrame com valores ausentes tratados.
        """
        df = df.copy()

        if strategy == "ffill":
            df = df.ffill()
        elif strategy == "bfill":
            df = df.bfill()
        elif strategy == "mean":
            df = df.fillna(df.mean())
        else:
            df = df.dropna()

        return df

    def normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None
    ) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        Normaliza colunas específicas do DataFrame.

        Args:
            df (pd.DataFrame): DataFrame com dados.
            columns (list): Colunas a serem normalizadas. Se None, normaliza todas.

        Returns:
            Tuple[pd.DataFrame, MinMaxScaler]: DataFrame normalizado e scaler.
        """
        if columns is None:
            columns = df.columns.tolist()

        # Remove colunas não numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        columns = [col for col in columns if col in numeric_columns]

        if not columns:
            return df, self.scaler

        df[columns] = self.scaler.fit_transform(df[columns])
        return df, self.scaler

    def create_sequences(
        self,
        df: pd.DataFrame,
        target_column: str,
        window_size: int,
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências para treinamento do LSTM.

        Args:
            df (pd.DataFrame): DataFrame com dados.
            target_column (str): Coluna alvo.
            window_size (int): Tamanho da janela de entrada.
            horizon (int): Horizonte de previsão.

        Returns:
            Tuple[np.ndarray, np.ndarray]: X (entrada), y (saída).
        """
        target = df[target_column].values.reshape(-1, 1)
        X, y = [], []

        for i in range(len(target) - window_size - horizon + 1):
            X.append(target[i:i + window_size])
            y.append(target[i + window_size:i + window_size + horizon])

        return np.array(X), np.array(y)



