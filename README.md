# ARIMA Sales Forecast – Terminal App

A simple Python terminal app that will forecast total product sales using ARIMA with seasonality. Just provide a CSV file, product name, and forecast dates and it handles the rest.

## Features

- Accepts various CSV formats (wide or tidy)
- Handles messy columns and product names
- Forecasts total sales using seasonal ARIMA
- Ignores outlying spikes using smoothing
- Saves the forecast and monthly trend plots
- Optionally exports daily forecast to CSV

## Usage

```bash
pip install -r requirements.txt
python app.py
```

You will be asked to input:
- The path to your CSV file
- Your product name
- Start and end dates of the time period (DD-MM-YYYY)
- Whether to export daily forecast

The forecast results and plots are saved automatically.