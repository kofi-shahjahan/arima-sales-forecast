# ARIMA + Flask Sales Forecast App

This is a small web app made to test forecasting sales of a product using ARIMA with seasonality. Using Flask and Statsmodels, it accepts uploaded CSV sales data and uses historical trends to generate short-term forecasts.

## Features
- You can upload any format of sales data (tidy or wide formats)
- Seasonal forecast using ARIMA
- Spike smoothing and moving average logic
- Graphs using matplotlib for daily forecast and monthly sales patterns

## Running the App

```bash
pip install -r requirements.txt
python app.py