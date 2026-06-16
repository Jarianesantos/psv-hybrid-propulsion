import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


class DataPreprocessor:
    """Classe para pré-processar dados de demanda energética."""

    def __init__(self, target_column: str = "power_demand"):
        self.target_column = target_column
        self.scaler: Optional[MinMaxScaler] = None
        self.target_scaler: Optional[MinMaxScaler] = None

    def load_data(self, file_path: str, date_column: str = "timestamp") -> pd.DataFrame:
        """Carrega dados energéticos de um CSV."""
        path = Path(file_path)
        df = pd.read_csv(path)

        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df = df.sort_values(date_column).reset_index(drop=True)

        return df

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "ffill") -> pd.DataFrame:
        """Trata valores ausentes no DataFrame."""
        df = df.copy()
        if strategy == "ffill":
            return df.ffill()
        if strategy == "bfill":
            return df.bfill()
        if strategy == "mean":
            return df.fillna(df.mean(numeric_only=True))
        return df.dropna()

    def add_time_features(self, df: pd.DataFrame, date_column: str = "timestamp") -> pd.DataFrame:
        """Adiciona features temporais para previsão de demanda."""
        df = df.copy()
        if date_column not in df.columns:
            return df

        df["hour"] = df[date_column].dt.hour
        df["day_of_week"] = df[date_column].dt.dayofweek
        df["day_of_month"] = df[date_column].dt.day
        df["month"] = df[date_column].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

        return df

    def normalize(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Normaliza os dados numéricos usando MinMaxScaler."""
        df = df.copy()
        numeric = df.select_dtypes(include=[np.number])
        if columns is None:
            columns = numeric.columns.tolist()
        else:
            columns = [col for col in columns if col in numeric.columns]

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        if columns:
            df[columns] = self.scaler.fit_transform(df[columns])

        if self.target_column in columns:
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler.fit(df[[self.target_column]])

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        target_column: str,
        window_size: int,
        horizon: int,
        feature_columns: Optional[list[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cria sequências para treinamento de LSTM."""
        if feature_columns is None:
            feature_columns = [target_column]

        data = df[feature_columns].values
        target = df[target_column].values

        X, y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i : i + window_size])
            y.append(target[i + window_size : i + window_size + horizon])

        return np.array(X), np.array(y)

    def split_train_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divide os dados em treino e teste."""
        split_idx = int(len(X) * train_ratio)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
