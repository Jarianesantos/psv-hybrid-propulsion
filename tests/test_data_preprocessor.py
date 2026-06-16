import pandas as pd
import numpy as np

from src.prediction.data_preprocessor import DataPreprocessor


def test_handle_missing_values_ffill():
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "power_demand": [100.0, np.nan, 110.0],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    preprocessor = DataPreprocessor(target_column="power_demand")

    result = preprocessor.handle_missing_values(df, strategy="ffill")

    assert not result["power_demand"].isna().any()
    assert result["power_demand"].iloc[1] == 100.0


def test_add_time_features():
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]) }
    )
    preprocessor = DataPreprocessor(target_column="power_demand")

    result = preprocessor.add_time_features(df)

    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "is_weekend" in result.columns
    assert result.loc[0, "hour"] == 0


def test_create_sequences_returns_expected_shape():
    values = np.arange(10.0)
    df = pd.DataFrame({"power_demand": values})
    preprocessor = DataPreprocessor(target_column="power_demand")

    X, y = preprocessor.create_sequences(
        df,
        target_column="power_demand",
        window_size=4,
        horizon=2,
    )

    assert X.shape == (5, 4, 1)
    assert y.shape == (5, 2)
    assert np.array_equal(X[0].flatten(), np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.array_equal(y[0], np.array([4.0, 5.0]))
