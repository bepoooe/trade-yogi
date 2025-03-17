import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import flash,Flask, redirect, url_for, render_template, request, session, jsonify
import numpy as np
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
import pyrebase
from firebase_admin import auth, credentials, initialize_app
from functools import wraps

# Load environment variables
load_dotenv()

# Configure API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
history = []

# Initialize model for chatbot
generation_config = {
    "temperature": 0.15,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=(
            "You are an expert at Trading insight of India. Your task is to engage in conversations about trade "
            "and answer questions. Explain specifically about the company so that users easily understand the background, "
            "history, and performance of that specific company in the trading market. Use humor and formality to make "
            "conversations educational and interesting."
        ),
    )
except Exception as e:
    print(f"Failed to configure model: {e}")
    exit()

# Load configuration from JSON file
with open('config.json', 'r') as c:
    params = json.load(c)["params"]

# Firebase Configuration
config = {
    'apiKey': "AIzaSyCLurZvJiKCp-EnViPYgcdZATt1TLtfmWo",
    'authDomain': "protrade-3f3b8.firebaseapp.com",
    'projectId': "protrade-3f3b8",
    'storageBucket': "protrade-3f3b8.firebasestorage.app",
    'messagingSenderId': "9677822271",
    'appId': "1:9677822271:web:b9929b2fb4dac33e435b8a",
    'measurementId': "G-N7Q5D58Y03",
    'databaseURL': ""
}

# Initialize Firebase
firebase = pyrebase.initialize_app(config)
auth_instance = firebase.auth()

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'super-secret-key'

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:  # Check if 'user' is in session
            flash('Please log in to access this page.', 'warning')  # Flash a message
            return redirect(url_for('index'))  # Redirect to the index (login page)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/profile')
@login_required
def profile():
    # The user is logged in, so return their profile data
    return 'Welcome to your profile page!'


@app.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('pass')

    try:
        # Your logic to authenticate the user (e.g., using Firebase or any method)
        user = auth_instance.sign_in_with_email_and_password(email, password)
        
        # Store the user in the session after successful login
        session['user'] = email  # This is the session variable
        session['token'] = user['idToken']

        return jsonify({'success': 'Login successful'}), 200
    except Exception as e:
        error_message = json.loads(e.args[1])['error']['message']
        return jsonify({'error': error_message}), 400


@app.route('/signup', methods=['POST'])
def signup():
    firstname = request.json.get('firstname')
    email = request.json.get('email')
    password = request.json.get('pass')

    try:
        # Your logic for signing up the user
        user = auth_instance.create_user_with_email_and_password(email, password)
        
        # Optionally send an email verification if needed

        return jsonify({'success': 'Signup successful. Please verify your email.'}), 200
    except Exception as e:
        error_message = json.loads(e.args[1])['error']['message']
        return jsonify({'error': error_message}), 400


@app.route('/logout')
def logout():
    session.clear()  # Clears the session data
    flash('You have been logged out.', 'success')  # Flash success message
    return redirect(url_for('index'))  # Redirect to the homepage after logout



@app.route('/')
@app.route('/index')

def index():
    return render_template('index.html', params=params)

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if request.method == 'POST':
        # Get user input from the JSON body
        user_input = request.json.get('user_input')
        
        if user_input:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(user_input)
            model_response = response.text

            # Update history with user input and model response
            history.append({"role": "user", "parts": [user_input]})
            history.append({"role": "model", "parts": [model_response]})

            # Return model's response as JSON
            return jsonify({"response": model_response})

    # If not a POST request, render the chat page
    return render_template('chatbot.html', params=params)
@app.route('/contact')
@login_required
def contact():
    return render_template('contact.html', params=params)

@app.route('/future', methods=['GET', 'POST'])
@login_required
def future():
    if request.method == 'POST':
        selected_companies = request.form.getlist('stocks')
        investment_amount = float(request.form['investment_amount'])
        holding_period = request.form['holding_period']
        if holding_period.replace('.', '', 1).isdigit() and holding_period.count('.') <= 1:
            holding_period = round(float(holding_period))
        else:
            holding_period = 0

        start_date = "2023-01-01"
        end_date = "2023-11-29"
        results = run_analysis(selected_companies, investment_amount, holding_period, start_date, end_date)
        return render_template('results.html', results=results, abs=abs, params=params)

    stock_symbols = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ITC.NS', 'SBIN.NS', 'LT.NS', 'AXISBANK.NS',
        'ICICIBANK.NS', 'HINDUNILVR.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'WIPRO.NS', 'NTPC.NS', 'ADANIPORTS.NS',
        'HCLTECH.NS', 'KOTAKBANK.NS', 'M&M.NS', 'INDUSINDBK.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'BHEL.NS',
        'BASFINDIA.NS', 'BOSCHLTD.NS', 'HINDALCO.NS', 'DLF.NS', 'EICHERMOT.NS', 'MARICO.NS', 'LUPIN.NS'
    ]
    return render_template('future.html', stock_symbols=stock_symbols, params=params)

def fetch_historical_data_india(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if stock_data.empty:
            print(f"Warning: No data fetched for {symbol}. Skipping this stock.")
            return None
        
        # Moving Averages
        stock_data['SMA_50'] = stock_data['Close'].rolling(window=50, min_periods=1).mean()
        stock_data['SMA_200'] = stock_data['Close'].rolling(window=200, min_periods=1).mean()
        
        # Relative Strength Index (RSI)
        stock_data['RSI'] = calculate_rsi(stock_data['Close'], 14)
        
        # Volatility (Standard Deviation over 14-day window)
        stock_data['Volatility'] = stock_data['Close'].rolling(window=14, min_periods=1).std()
        
        # Target (Next Day Close Price)
        stock_data['Target'] = stock_data['Close'].shift(-1)
        
        stock_data.dropna(inplace=True)  # Drop NaN values after calculations
        return stock_data
    
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_model(data):
    if data is None or data.empty:
        print("No data available for training the model.")
        return None
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_future_price(symbol, model, holding_period):
    stock_data = yf.download(symbol, period="1d", interval="1d")
    real_time_features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
    last_close = real_time_features[0][3]
    predicted_prices = [last_close]
    
    for _ in range(holding_period):
        predicted_price = model.predict(real_time_features)[0]
        real_time_features[0][3] = predicted_price
        next_day = pd.DataFrame({'Close': [predicted_price], 'Volume': [real_time_features[0][4]]}, 
                              index=[stock_data.index[-1] + pd.Timedelta(days=1)])
        stock_data = pd.concat([stock_data, next_day])
        predicted_prices.append(predicted_price)
    
    return predicted_prices

def calculate_investment(buy_price, predicted_sell_price, investment_amount):
    quantity = investment_amount / buy_price
    final_value = quantity * predicted_sell_price
    profit_loss = final_value - investment_amount
    return float(profit_loss), float(final_value)

def run_analysis(selected_companies, investment_amount, holding_period, start_date, end_date):
    results = []
    for symbol in selected_companies:
        print(f"\nProcessing {symbol}...")
        historical_data = fetch_historical_data_india(symbol, start_date, end_date)
        if historical_data is None:
            continue
        model = train_model(historical_data)
        if model is None:
            continue
        predicted_prices = predict_future_price(symbol, model, holding_period)
        latest_close_price = historical_data['Close'].iloc[-1]
        predicted_sell_price = predicted_prices[-1]
        profit_loss, final_value = calculate_investment(latest_close_price, predicted_sell_price, investment_amount)
        results.append({
            "symbol": symbol,
            "investment_amount": investment_amount,
            "predicted_sell_price": predicted_sell_price,
            "final_value": final_value,
            "profit_loss": profit_loss,
        })
    return results

if __name__ == "__main__":
    app.run(debug=True)  