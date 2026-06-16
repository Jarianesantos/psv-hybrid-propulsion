#!/usr/bin/env bash
set -e

TASK="$1"

if [ -z "$TASK" ]; then
  echo "Uso: ./run.sh <install|generate-data|run-app|train>"
  exit 1
fi

case "$TASK" in
  install)
    echo "Instalando dependências..."
    python -m pip install -r requirements.txt
    ;;
  generate-data)
    echo "Gerando dados sintéticos..."
    python generate_synthetic_data.py --days 30
    ;;
  run-app)
    echo "Iniciando Streamlit..."
    streamlit run app.py
    ;;
  train)
    echo "Treinando modelo..."
    python main.py
    ;;
  test)
    echo "Executando testes..."
    pytest
    ;;
  *)
    echo "Tarefa desconhecida: $TASK"
    echo "Uso: ./run.sh <install|generate-data|run-app|train|test>"
    exit 1
    ;;
esac
