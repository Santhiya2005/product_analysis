from flask import Flask, render_template, url_for
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima.model import ARIMA

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    # Load the CSV file
    file_path = 'order_report (1).csv'
    if not os.path.exists(file_path):
        return "Error: CSV file 'order_report (1).csv' not found."

    df = pd.read_csv(file_path)

    # Parse and clean date column
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)
    df['Year'] = df['Order Date'].dt.year
    df['Date'] = df['Order Date'].dt.date

    # 1. Year-wise Product Orders Summary
    products_by_year = df.groupby(['Year', 'Product']).size().reset_index(name='Count')
    products_by_year_sorted = products_by_year.sort_values(['Year', 'Count'], ascending=[False, False])
    yearly_orders_data = products_by_year_sorted.to_dict(orient='records')

    # 2. Pie Chart for Latest Year Product Distribution
    latest_year = products_by_year['Year'].max()
    latest_year_data = products_by_year[products_by_year['Year'] == latest_year]
    top_products = latest_year_data.sort_values('Count', ascending=False)

    # Ensure 'static' folder exists
    os.makedirs('static', exist_ok=True)

    # Save pie chart for latest year
    latest_year_pie_path = os.path.join('static', 'latest_year_pie.png')
    if os.path.exists(latest_year_pie_path):
        os.remove(latest_year_pie_path)

    plt.figure(figsize=(8, 8))
    plt.pie(top_products['Count'], labels=top_products['Product'], autopct='%1.1f%%', startangle=140)
    plt.title(f'Product Distribution in {latest_year}')
    plt.tight_layout()
    plt.savefig(latest_year_pie_path)
    plt.close()

    # 3. Weekly Product Sales Forecast
    df['Week'] = df['Order Date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_sales = df.groupby(['Week', 'Product']).size().reset_index(name='Count')
    weekly_pivot = weekly_sales.pivot(index='Week', columns='Product', values='Count').fillna(0).sort_index()

    # Forecast next week's demand using ARIMA
    forecast_results = {}
    for product in weekly_pivot.columns:
        series = weekly_pivot[product]
        try:
            model = ARIMA(series, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            forecast_value = max(0, int(round(forecast[0])))
            forecast_results[product] = forecast_value
        except Exception as e:
            forecast_results[product] = None
            print(f"[ERROR] Forecast failed for '{product}': {e}")

    forecast_df = pd.DataFrame(
        [(k, v) for k, v in forecast_results.items() if v is not None],
        columns=['Product', 'Predicted_Next_Week']
    ).sort_values(by='Predicted_Next_Week', ascending=False)

    # Save forecast pie chart
    forecast_pie_path = os.path.join('static', 'forecast_pie.png')
    if os.path.exists(forecast_pie_path):
        os.remove(forecast_pie_path)

    plt.figure(figsize=(8, 8))
    plt.pie(forecast_df['Predicted_Next_Week'], labels=forecast_df['Product'], autopct='%1.1f%%', startangle=140)
    plt.title('Forecasted Product Share for Next Week')
    plt.tight_layout()
    plt.savefig(forecast_pie_path)
    plt.close()

    # Data for rendering in HTML
    forecast_data = forecast_df.to_dict(orient='records')

    return render_template(
        'index.html',
        yearly_orders=yearly_orders_data,
        latest_year_pie=url_for('static', filename='latest_year_pie.png'),
        forecast=forecast_data,
        forecast_pie=url_for('static', filename='forecast_pie.png'),
        latest_year=latest_year
    )

if __name__ == '__main__':
    app.run(debug=True)
