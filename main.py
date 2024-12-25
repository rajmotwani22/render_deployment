from typing import Optional
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime
app = FastAPI()

# Directories
CHARTS_FOLDER = "static/saved_charts"
os.makedirs(CHARTS_FOLDER, exist_ok=True)

# Jinja2 Templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
DATABASE = "financial_data.db"


def initialize_database():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                rent REAL NOT NULL,
                utilities REAL NOT NULL,
                groceries REAL NOT NULL,
                other_expenses REAL NOT NULL,
                earnings REAL NOT NULL,
                total_expenses REAL NOT NULL,
                savings REAL NOT NULL,
                post_investment REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()

initialize_database()

@app.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard(request: Request):
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, savings, post_investment, timestamp
                FROM user_data
                ORDER BY post_investment DESC
            """)
            leaderboard_data = cursor.fetchall()

        return templates.TemplateResponse(
            "leaderboard.html",
            {
                "request": request,
                "leaderboard_data": leaderboard_data,
                "enumerate": enumerate
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": str(e)}
        )


# Visualization Functions
def visualize_expense_breakdown(expenses, filename):
    labels = ['Rent', 'Utilities', 'Groceries', 'Other Expenses']
    pie_path = os.path.join(CHARTS_FOLDER, f"{filename}_expenses_pie.png")
    plt.figure(figsize=(8, 6))
    plt.pie(expenses, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Monthly Expense Breakdown')
    plt.savefig(pie_path)
    plt.close()
    return pie_path

def visualize_savings_growth(savings, filename):
    savings_growth_path = os.path.join(CHARTS_FOLDER, f"{filename}_savings_growth.png")
    months = [f'Month {i+1}' for i in range(12)]
    savings_over_time = [savings * (1 + 0.02) ** i for i in range(12)]
    plt.figure(figsize=(10, 6))
    plt.plot(months, savings_over_time, marker='o', color='green')
    plt.title('Savings Growth Over Time')
    plt.xlabel('Months')
    plt.ylabel('Savings ($)')
    plt.grid(True)
    plt.savefig(savings_growth_path)
    plt.close()
    return savings_growth_path

# Investment Allocation Visualization
def visualize_investment_allocation(investment_amount, filename, investment_type):
    if investment_type == "large":
        data = {
            'Company': ['Apple', 'Amazon', 'Microsoft', 'Google', 'Tesla'],
            'Expected Return (%)': [12, 15, 10, 11, 20],
            'Risk (Volatility %)': [18, 22, 15, 17, 30]
        }
    elif investment_type == "mid":
        data = {
            'Company': ['NVIDIA', 'AMD', 'Qualcomm', 'Texas Instruments', 'Micron'],
            'Expected Return (%)': [8, 10, 9, 7, 12],
            'Risk (Volatility %)': [16, 19, 14, 12, 20]
        }
    elif investment_type == "small":
        data = {
            'Company': ['Plug Power', 'Roku', 'Peloton', 'Zoom', 'Shopify'],
            'Expected Return (%)': [14, 18, 20, 22, 25],
            'Risk (Volatility %)': [30, 35, 28, 25, 32]
        }
    else:
        raise ValueError("Invalid investment type")

    df = pd.DataFrame(data)
    df['Weight'] = df['Expected Return (%)'] / df['Expected Return (%)'].sum()
    df['Investment ($)'] = df['Weight'] * investment_amount
    df['Post-Investment Amount ($)'] = df['Investment ($)'] * (1 + df['Expected Return (%)'] / 100)

    pie_path = os.path.join(CHARTS_FOLDER, f"{filename}_allocation_pie.png")
    plt.figure(figsize=(8, 6))
    plt.pie(df['Investment ($)'], labels=df['Company'], autopct='%1.1f%%', startangle=140)
    plt.title(f'{investment_type.capitalize()} Cap Investment Allocation')
    plt.savefig(pie_path)
    plt.close()

    pie_path = os.path.join(CHARTS_FOLDER, f"{filename}_allocation_pie.png")
    plt.figure(figsize=(8, 6))
    plt.pie(df['Investment ($)'], labels=df['Company'], autopct='%1.1f%%', startangle=140)
    plt.title(f'{investment_type.capitalize()} Cap Investment Allocation')
    plt.savefig(pie_path)
    plt.close()

    line_path = os.path.join(CHARTS_FOLDER, f"{filename}_allocation_line.png")
    df['Cumulative Investment ($)'] = df['Investment ($)'].cumsum()

    # Generate line chart
    plt.figure(figsize=(10, 6))
    plt.plot(df['Company'], df['Cumulative Investment ($)'], marker='o', color='green')
    plt.title(f'{investment_type.capitalize()} Investment Cumulative Amount')
    plt.xlabel('Company')
    plt.ylabel('Cumulative Investment ($)')
    plt.grid()
    plt.savefig(line_path)
    plt.close()

    return pie_path, line_path, df

# Stock Prediction Functions
def train_model(data):
    data['Moving Average'] = data['Close'].rolling(window=10).mean()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data['Daily Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    if data.empty:
        raise ValueError("Not enough data to train the model.")

    X = data[['Moving Average', 'Volatility', 'Daily Return']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_stock_prices(model, live_data):
    live_data['Moving Average'] = live_data['Close'].rolling(window=10).mean()
    live_data['Volatility'] = live_data['Close'].rolling(window=10).std()
    live_data['Daily Return'] = live_data['Close'].pct_change()
    
    live_data.dropna(inplace=True)

    try:
        live_data['Predicted Price'] = model.predict(live_data[['Moving Average', 'Volatility', 'Daily Return']])
        future_price = live_data['Predicted Price'].iloc[-1]
        current_price = live_data['Close'].iloc[-1]
    except KeyError:
        raise ValueError("Missing required columns in the input data.")
    return live_data, future_price, current_price

def visualize_predictions(data, ticker, filename, future_price, current_price):
    prediction_path = os.path.join(CHARTS_FOLDER, f"{filename}_prediction.png")
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Actual Price', color='blue')
    plt.plot(data.index, data['Predicted Price'], label='Predicted Price', color='orange')
    plt.title(f'{ticker} - Actual vs Predicted Prices\nFuture Price: ${future_price:.2f}  | Current Price: ${current_price:.2f}')
    plt.legend()
    plt.grid()
    plt.savefig(prediction_path)
    plt.close()
    return prediction_path

def visualize_investment_bar_chart(before_investment, after_investment, filename):
    bar_chart_path = os.path.join(CHARTS_FOLDER, f"{filename}_investment_bar_chart.png")
    categories = ['Before Investment', 'After Investment']
    values = [before_investment, after_investment]

    plt.figure(figsize=(8, 6))
    plt.bar(categories, values, color=['red', 'green'])
    plt.title("Investment Comparison")
    plt.ylabel("Amount ($)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(bar_chart_path)
    plt.close()
    return bar_chart_path


@app.post("/predict_stock", response_class=HTMLResponse)
async def predict_stock(request: Request, ticker: str = Form(...)):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"stock_{timestamp}"

        stock_data = yf.Ticker(ticker).history(period="1y")
        if stock_data.empty:
            raise ValueError("No stock data available for the given ticker.")
        
        model = train_model(stock_data)
        predicted_data, future_price, current_price = predict_stock_prices(model, stock_data)
        prediction_chart = visualize_predictions(predicted_data, ticker, filename, future_price, current_price)

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "ticker": ticker,
                "prediction_chart": prediction_chart,
                "future_price": f"{future_price:.2f}",
                "current_price": f"{current_price:.2f}",
                "name": "N/A",
                "total_expenses": 0.0,
                "earnings": 0.0,
                "savings": 0.0,
                "investment_bar_chart": None,
            },
        )
    except Exception as e:
        return templates.TemplateResponse("results.html", {"request": request, "error": str(e)})




@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
#async def root():
#    return {"message": "Hello World"}


@app.post("/calculate", response_class=HTMLResponse)
async def calculate(
    request: Request,
    name: str = Form(...),
    rent: float = Form(0.0),
    utilities: float = Form(0.0),
    groceries: float = Form(0.0),
    other_expenses: float = Form(0.0),
    earnings: float = Form(...),
    investment_type: str = Form("large"),
    ticker: str = Form("AAPL"),
):
    try:
        total_expenses = (rent + utilities + groceries + other_expenses) * 12
        savings = earnings - total_expenses
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"user_{timestamp}"

        # Save to DB
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_data (name, rent, utilities, groceries, other_expenses, earnings, total_expenses, savings, post_investment, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, rent, utilities, groceries, other_expenses, earnings, total_expenses, savings, 0, timestamp))
            conn.commit()

        # Create charts
        pie_allocation_path, line_allocation_path, investment_df = visualize_investment_allocation(
            savings, filename, investment_type)

        stock_data = yf.Ticker(ticker).history(period="1y")
        model = train_model(stock_data)
        predicted_data, future_price, current_price = predict_stock_prices(model, stock_data)
        prediction_chart = visualize_predictions(predicted_data, ticker, filename, future_price, current_price)

        expense_pie_chart = visualize_expense_breakdown([rent, utilities, groceries, other_expenses], filename)
        savings_growth_chart = visualize_savings_growth(savings, filename)
        post_investment = savings * 1.15  # Simulate investment growth
        bar_chart_path = visualize_investment_bar_chart(savings, post_investment, filename)

        total_post_investment_amount = investment_df["Post-Investment Amount ($)"].sum()


        # Render results
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "name": name,
                "total_expenses": total_expenses,
                "earnings": earnings,
                "savings": savings,
                "post_investment": total_post_investment_amount,
                "investment_table": investment_df.to_dict('records'),
                "pie_allocation_path": pie_allocation_path,
                "line_allocation_path": line_allocation_path,
                "prediction_chart": prediction_chart,
                "ticker": ticker,
                "expense_pie_chart": expense_pie_chart,
                "savings_growth_chart": savings_growth_chart,
                "investment_type": investment_type.capitalize(),
                "future_price": f"{future_price:.2f}",
                "current_price": f"{current_price:.2f}",
                "investment_bar_chart": bar_chart_path,
            },
        )
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
