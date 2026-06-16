import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.prediction.data_preprocessor import DataPreprocessor
from src.prediction.lstm_model import EnergyDemandLSTM


def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    default_path = Path("data/energy_demand.csv")
    if default_path.exists():
        return pd.read_csv(default_path)

    return None


def main():
    st.set_page_config(page_title="IA Energia Híbrido-Elétrica", layout="wide")
    st.title("IA para Energia Híbrido-Elétrica")

    st.markdown(
        """
        Este aplicativo treina um modelo LSTM para prever demanda de energia a partir de dados históricos.
        Faça upload de um CSV com colunas `timestamp` e `power_demand`, ou deixe o arquivo em `data/energy_demand.csv`.
        """
    )

    uploaded_file = st.file_uploader("Carregar CSV de demanda energética", type=["csv"])
    df = load_dataset(uploaded_file)

    if df is None:
        st.warning("Nenhum arquivo CSV encontrado. Por favor, carregue um arquivo ou crie data/energy_demand.csv.")
        return

    st.subheader("Amostra de dados")
    st.dataframe(df.head(10))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if st.button("Treinar modelo LSTM"):
        preprocessor = DataPreprocessor(target_column="power_demand")
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

        if len(X) < 10:
            st.error("Dados insuficientes para treinar. Forneça pelo menos 30 registros.")
            return

        X_train, X_test, y_train, y_test = preprocessor.split_train_test(X, y, train_ratio=0.8)

        model = EnergyDemandLSTM(
            input_window=24,
            forecast_horizon=6,
            lstm_units=64,
            epochs=20,
            batch_size=16,
            model_dir="models",
        )
        model.scaler = preprocessor.scaler

        progress_text = st.empty()
        progress_text.text("Treinando modelo...")

        history = model.train(X_train, y_train, X_val=X_test[: int(len(X_test) * 0.5)], y_val=y_test[: int(len(y_test) * 0.5)])
        metrics = model.evaluate(X_test[int(len(X_test) * 0.5) :], y_test[int(len(y_test) * 0.5) :])

        st.success("Treinamento concluído")
        st.write("### Métricas de avaliação")
        st.json(metrics)

        st.write("### Histórico de treinamento")
        history_df = pd.DataFrame(history)
        st.line_chart(history_df)

        st.write("### Previsão para os últimos passos")
        predictions = model.predict(X_test[:5])
        actual = y_test[:5]
        predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])
        actual = actual.reshape(actual.shape[0], actual.shape[1])

        df_pred = pd.DataFrame(
            {
                "actual": actual.flatten(),
                "predicted": predictions.flatten(),
            }
        )
        st.line_chart(df_pred)

        st.write("### Previsão adicional de próximos 6 passos")
        last_window = X_test[-1:]
        future_forecast = model.predict(last_window, steps=6).flatten()
        st.bar_chart(pd.DataFrame({"forecast": future_forecast}))

    st.write("---")
    st.markdown(
        """
        **Instruções:**
        1. Coloque seus dados em `data/energy_demand.csv` ou faça upload direto.
        2. Clique em "Treinar modelo LSTM".
        3. Veja as métricas e as previsões.
        """
    )


if __name__ == "__main__":
    main()
