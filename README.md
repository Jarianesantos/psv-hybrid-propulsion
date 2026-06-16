# psv-hybrid-propulsion

Projeto de previsão de demanda de energia híbrido-elétrica.

## Como rodar

1. Ative a virtualenv:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Instale dependências de runtime:
   ```powershell
   pip install -r requirements.txt
   ```
3. Instale dependências de desenvolvimento (opcional):
   ```powershell
   pip install -r requirements-dev.txt
   ```
4. Execute o app Streamlit:
   ```powershell
   streamlit run app.py
   ```

## Dados

O arquivo de exemplo `data/energy_demand.csv` já está disponível. Ele contém as colunas `timestamp` e `power_demand`.

Se quiser gerar dados sintéticos, rode:
```powershell
python generate_synthetic_data.py --days 30
```

## Atalhos de execução

No Windows PowerShell, use:
```powershell
.\run.ps1 -Task install
.\run.ps1 -Task generate-data
.\run.ps1 -Task run-app
.\run.ps1 -Task train
.\run.ps1 -Task test
```

No Linux/macOS ou em um terminal que suporte Make:
```bash
make install
make generate-data
make run-app
make train
make test
```

Ou use o script bash:
```bash
./run.sh install
./run.sh generate-data
./run.sh run-app
./run.sh train
./run.sh test
```

## Estrutura

- `app.py`: interface Streamlit para treinar e avaliar o modelo LSTM
- `main.py`: pipeline simples de treinamento via script
- `src/prediction/data_preprocessor.py`: pré-processamento de dados
- `src/prediction/lstm_model.py`: modelo LSTM de previsão de demanda
