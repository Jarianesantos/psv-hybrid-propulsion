param(
    [string]$Task = "help"
)

switch ($Task) {
    "install" {
        Write-Host "Installing dependencies..."
        python -m pip install -r requirements.txt
        break
    }
    "generate-data" {
        Write-Host "Generating synthetic data..."
        python generate_synthetic_data.py --days 30
        break
    }
    "run-app" {
        Write-Host "Starting Streamlit app..."
        streamlit run app.py
        break
    }
    "train" {
        Write-Host "Training model..."
        python main.py
        break
    }
    "test" {
        Write-Host "Running tests..."
        pytest
        break
    }
    default {
        Write-Host "Available tasks: install, generate-data, run-app, train, test"
        Write-Host "Example: .\run.ps1 -Task run-app"
        break
    }
}
