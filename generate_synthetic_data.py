import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path


def generate_energy_demand(
    start_time: datetime,
    hours: int,
    base_demand: float = 120.0,
    day_peak: float = 90.0,
    night_drop: float = 20.0,
    weekend_reduction: float = 0.9,
    noise_scale: float = 5.0,
) -> pd.DataFrame:
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    demands = []

    for ts in timestamps:
        seasonal = base_demand
        hour = ts.hour
        if 6 <= hour < 10:
            seasonal += day_peak * np.sin((hour - 6) / 4 * np.pi)
        elif 10 <= hour < 17:
            seasonal += day_peak * 0.7
        elif 17 <= hour < 21:
            seasonal += day_peak * np.sin((hour - 17) / 4 * np.pi)
        else:
            seasonal -= night_drop

        if ts.weekday() >= 5:
            seasonal *= weekend_reduction

        noise = np.random.normal(0, noise_scale)
        demands.append(max(0, seasonal + noise))

    return pd.DataFrame({"timestamp": timestamps, "power_demand": np.round(demands, 2)})


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✔️ Dataset salvo em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de dados sintéticos de demanda energética.")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Número de dias de dados horarios a gerar.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/energy_demand.csv",
        help="Caminho de saída para o CSV gerado.",
    )
    args = parser.parse_args()

    df = generate_energy_demand(
        start_time=datetime(2026, 1, 1, 0, 0, 0),
        hours=args.days * 24,
    )
    save_dataset(df, Path(args.output))
