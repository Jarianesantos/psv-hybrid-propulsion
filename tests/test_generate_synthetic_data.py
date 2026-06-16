from datetime import datetime

import pandas as pd

from generate_synthetic_data import generate_energy_demand, save_dataset


def test_generate_energy_demand_creates_dataframe():
    df = generate_energy_demand(datetime(2026, 1, 1, 0, 0, 0), hours=24)

    assert len(df) == 24
    assert list(df.columns) == ["timestamp", "power_demand"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert (df["power_demand"] >= 0).all()


def test_save_dataset_writes_csv(tmp_path):
    df = generate_energy_demand(datetime(2026, 1, 1, 0, 0, 0), hours=12)
    output_path = tmp_path / "energy_demand.csv"

    save_dataset(df, output_path)

    assert output_path.exists()

    loaded = pd.read_csv(output_path)
    assert "timestamp" in loaded.columns
    assert "power_demand" in loaded.columns
