import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
from flask import Flask, render_template, request
import difflib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def find_best_product_match(user_input, product_list):
    # Normalize inputs
    user_input = user_input.lower().replace(" ", "").replace("_", "")
    normalized_products = {
        product: product.lower().replace(" ", "").replace("_", "")
        for product in product_list
    }

    # Use difflib to find closest match
    matches = difflib.get_close_matches(user_input, normalized_products.values(), n=1, cutoff=0.6)
    if not matches:
        return None

    # Find original product name that matched
    for original, normalized in normalized_products.items():
        if normalized == matches[0]:
            return original

    return None

# --- Helper Function: Normalize Sales Data ---
def normalize_sales_data(filepath, selected_product):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()

    # Step 1: Find date column
    date_col = next((col for col in df.columns if "date" in col), None)
    if not date_col:
        return None, "❌ Could not find a 'date' column."

    # Step 2: Check if file is tidy (has 'product' and 'sales' columns)
    has_product_col = any("product" in col for col in df.columns)
    has_sales_col = any("sale" in col for col in df.columns)

    if has_product_col and has_sales_col:
        # Tidy format
        product_col = next(col for col in df.columns if "product" in col)
        sales_col = next(col for col in df.columns if "sale" in col)

        df = df[[date_col, product_col, sales_col]].dropna()

        # Clean types
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
        df = df.dropna(subset=[date_col, sales_col])

        product_names = df[product_col].dropna().unique()
        best_match = find_best_product_match(selected_product, product_names)

        if not best_match:
            return None, f"⚠️ No match found for '{selected_product}'. Try checking the product name."

        df = df[df[product_col] == best_match]

        daily_sales = df.groupby(df[date_col])[sales_col].sum().asfreq("D").fillna(0)
        return daily_sales, None

    else:
        # Wide format — one column per product
        product_columns = [col for col in df.columns if col != date_col]
        if not product_columns:
            return None, "❌ No product columns found in wide format."

        df = pd.melt(df, id_vars=[date_col], value_vars=product_columns,
                     var_name="product", value_name="sales")

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
        df = df.dropna(subset=[date_col, "sales"])

        product_names = df["product"].dropna().unique()
        best_match = find_best_product_match(selected_product, product_names)

        if not best_match:
            return None, f"⚠️ No match found for '{selected_product}' in wide format. Try checking the product name."

        df = df[df["product"] == best_match]

        daily_sales = df.groupby(df[date_col])["sales"].sum().asfreq("D").fillna(0)
        return daily_sales, None

# --- Flask Route ---
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        product = request.form["product"]
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]
        file = request.files["file"]

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        daily_sales, error = normalize_sales_data(filepath, product)

        if error:
            result = error
        else:
            model_data = daily_sales[daily_sales.index >= "2021-01-01"]
            # --- Spike Detection & Filtering ---
            # Compute rolling median and slope of moving average
            rolling_median = model_data.rolling(window=30, min_periods=1).median()
            moving_avg_30 = model_data.rolling(window=30, min_periods=1).mean()
            slope = moving_avg_30.diff()

            # Identify potential spike days in last 14 days
            filtered_data = model_data.copy()
            for date in model_data[-14:].index:
                recent_val = model_data.loc[date]
                recent_median = rolling_median.loc[date]
                recent_slope = slope.loc[date]
                # Define spike: value more than 2× median and no rising slope
                if recent_val > 2 * recent_median and recent_slope <= 0:
                    filtered_data.loc[date] = rolling_median.loc[date - pd.Timedelta(days=1)]  # use prior day's median

            spike_dates = [date for date in model_data[-14:].index if model_data.loc[date] > 2 * rolling_median.loc[date] and slope.loc[date] <= 0]
            if spike_dates:
                filtered_data[spike_dates] = rolling_median.shift(1)[spike_dates]

            # Step 1: Smooth last 14 days of model_data with 7-day rolling median
            model_data_smooth = filtered_data.copy()
            model_data_smooth[-14:] = filtered_data[-14:].rolling(window=7, min_periods=1).median()
            # Step 2: Fit SARIMA model with monthly (30-day) seasonality
            model = SARIMAX(model_data_smooth, order=(1, 1, 1), seasonal_order=(1, 1, 1, 30))
            model_fit = model.fit(disp=False)

            # Step 3: Forecast range
            forecast_index = pd.date_range(start=start_date, end=end_date, freq='D')
            forecast = model_fit.predict(start=forecast_index[0], end=forecast_index[-1])
            total_forecast = forecast.sum()

            # Removed comparison to seasonal average and replaced with simplified result display
            forecast_mean = forecast.mean()

            debug_info = (
                f"<br><br><b>Debug Info:</b><br>"
                f"- Training period: {model_data.index.min().date()} to {model_data.index.max().date()}<br>"
                f"- Training days: {len(model_data)}<br>"
                f"- Forecast days: {len(forecast_index)}<br>"
            )

            result = f"ARIMA Forecast for '{product}' from {start_date} to {end_date}: {total_forecast:.2f}"
            result += debug_info

            # Plot actual and forecast
            plt.figure(figsize=(10, 5))
            # Improved y-axis scaling: round up to next multiple of 10, based only on actual and forecasted values
            y_max_val = max(model_data.max(), forecast.max())
            y_max = (int(y_max_val // 10) + 2) * 10  # add more headroom for clarity
            plt.ylim(0, y_max)
            plt.plot(model_data, label="Historical Sales")
            plt.plot(model_data.rolling(window=30).mean(), label="30-day Moving Average", linestyle="--")
            plt.plot(forecast_index, forecast, label="Forecast", color="orange")
            # Add 7-day rolling mean to forecast
            plt.plot(forecast_index, forecast.rolling(window=7, min_periods=1).mean(), label="Forecast 7-day Avg", linestyle="--", color="red")
            plt.title(f"Sales Forecast for {product}")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.legend()

            # Convert plot to base64
            img = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close()

            result += f'<br><img src="data:image/png;base64,{plot_url}"/>'

            # Monthly Sales Graph
            monthly_sales = model_data.resample("M").sum()
            plt.figure(figsize=(10, 5))
            plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linestyle='-', color='green')
            plt.title(f"Monthly Sales for {product}")
            plt.xlabel("Month")
            plt.ylabel("Total Sales")
            plt.grid(True)
            img2 = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img2, format='png')
            img2.seek(0)
            monthly_plot_url = base64.b64encode(img2.getvalue()).decode('utf8')
            plt.close()

            result += f'<br><br><b>Monthly Sales Overview:</b><br><img src="data:image/png;base64,{monthly_plot_url}"/>'

    return render_template("index.html", result=result)

# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True)
