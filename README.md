# Trade Yogi

Trade Yogi is a web application for stock market analysis and predictions focusing on the Indian market. It provides AI-powered trading insights, stock prediction models, and an interactive chatbot assistant.

## Features

- **AI-Powered Trading Assistant**: Gemini-powered chatbot that provides expert insights on Indian trading markets
- **Stock Price Prediction**: Machine learning models to predict future stock prices
- **Investment Analysis**: Calculate potential profits/losses based on prediction models
- **User Authentication**: Secure login and registration using Firebase
- **Responsive Interface**: Modern web interface for desktop and mobile users

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **AI Model**: Google Gemini 1.5 Flash
- **ML Models**: RandomForestRegressor from scikit-learn
- **Data Source**: Stock data cache system with fallback to synthetic data
- **Authentication**: Firebase Authentication

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A Gemini API key from Google AI Studio
- Git (optional, for cloning the repository)

## Installation

1. Clone the repository or download the source code:
   ```
   git clone https://github.com/yourusername/trade-yogi.git
   cd trade-yogi
   ```

2. Create and activate a virtual environment:
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your Gemini API key: `GEMINI_API_KEY=your_key_here`

5. Make sure your `config.json` file is set up correctly with your parameters

## Running the Application

1. Once the dependencies are installed and environment variables are set, run the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. The application should now be running on your local development server

## Usage Guide

### Registration and Login
1. Click on "Register" to create a new account with your email and password
2. After registration, log in with your credentials
3. You'll now have access to all features of the application

### Using the Chatbot
1. Navigate to the Chat section
2. Type your question about Indian trading markets
3. The AI assistant will provide detailed insights and information

### Stock Predictions
1. Go to the Future section
2. Select the stocks you want to analyze (up to 5 at a time)
3. Enter your investment amount
4. Specify the holding period in days
5. Submit to see the prediction results

### Viewing Results
- The results page will show:
  - Current stock price
  - Predicted future price
  - Potential profit/loss
  - Final portfolio value

## Troubleshooting

- **Issue**: Firebase authentication errors
  - **Solution**: Verify Firebase configuration in `app.py`

- **Issue**: "API key not configured" error
  - **Solution**: Ensure your `.env` file contains the correct Gemini API key

- **Issue**: Stock data not loading
  - **Solution**: The application uses a fallback to synthetic data if real-time data is unavailable

## Project Structure

- `app.py`: Main application file with Flask routes and ML logic
- `config.json`: Configuration parameters for the application
- `requirements.txt`: Python dependencies
- `templates/`: HTML templates for the web interface
- `static/`: CSS, JavaScript, and other static assets
- `stock_data_cache/`: Local cache for stock data

## Deployment Options

### PythonAnywhere (Recommended for Beginners)
1. Create an account on [PythonAnywhere](https://www.pythonanywhere.com/)
2. Go to the Web tab and add a new web app
3. Select Flask and Python 3.10
4. Set up a git clone of your repository
5. Create a `.env` file with your Gemini API key
6. Set up a virtual environment and install requirements
7. Configure the WSGI file to point to your Flask app

### Heroku
1. Create a `Procfile` with: `web: gunicorn app:app`
2. Add `gunicorn` to requirements.txt
3. Set environment variables in Heroku dashboard
4. Deploy using Heroku Git or GitHub integration

### Railway
1. Create a account on [Railway](https://railway.app/)
2. Link your GitHub repository
3. Add environment variables in the Railway dashboard
4. Deploy automatically from your GitHub repository

### Google Cloud Run
1. Install Google Cloud SDK
2. Create a `Dockerfile`:
   ```
   FROM python:3.10-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   CMD gunicorn --bind :$PORT app:app
   ```
3. Build and deploy:
   ```
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/trade-yogi
   gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/trade-yogi --platform managed
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 