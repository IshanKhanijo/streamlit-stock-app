import streamlit as st
import yfinance as yf
import requests
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from openai import OpenAI
import plotly.graph_objects as go
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

# Load API keys from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")
client = OpenAI(api_key=openai_key)

st.title("Stock Information, News & Price Prediction")
ticker = st.text_input("Enter Stock Ticker", value="AAPL")

if ticker:
    today = datetime.date.today()
    from_date = today - datetime.timedelta(days=30)

    # --- Company Info ---
    stock = yf.Ticker(ticker)
    stock_info = stock.info
    st.subheader("Company Info")
    metrics = {
        "Name": stock_info.get("longName", "N/A"),
        "Sector": stock_info.get("sector", "N/A"),
        "Market Cap": stock_info.get("marketCap", "N/A"),
        "PE Ratio": stock_info.get("trailingPE", "N/A"),
        "EPS": stock_info.get("trailingEps", "N/A"),
        "Dividend Yield": stock_info.get("dividendYield", "N/A"),
        "ROE": stock_info.get("returnOnEquity", "N/A"),
        "Beta": stock_info.get("beta", "N/A")
    }
    for key, value in metrics.items():
        st.write(f"**{key}:** {value}")

    # --- News Fetch ---
    url = (
        f"https://newsapi.org/v2/everything?q={ticker}&from={from_date}&to={today}"
        f"&sortBy=publishedAt&language=en&apiKey={news_api_key}"
    )
    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json().get("articles", [])
        st.subheader("Latest News Articles")
        for article in articles[:5]:
            st.markdown(f"**{article['title']}**")
            st.write(article['description'])
            st.write(article['url'])
            st.write("---")

        # --- OpenAI News Summary ---
        st.subheader("AI-generated News Summary")
        combined_text = "\n\n".join(
            f"{article.get('title', '')}\n{article.get('description', '')}" for article in articles[:5]
        )
        if combined_text.strip():
            prompt = f"Summarize the following news about {ticker}:\n\n{combined_text}"
            try:
                summary_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = summary_response.choices[0].message.content
                st.write(summary)
            except Exception as e:
                st.error(f"OpenAI API Error: {e}")
        else:
            st.info("No content available for summarization.")
    else:
        st.error("Failed to fetch news.")
    
        # --- Historical Stock Price Chart ---
    st.subheader("Historical Stock Price Chart")

    # Timeframe selection
    chart_period = st.radio(
        "Select Timeframe for Graph",
        ["1y", "2y", "5y", "10y"],
        index=3,
        horizontal=True
    )

    # Get closing price data only
    chart_data = yf.download(ticker, period=chart_period, interval="1d")[['Close']].dropna()

    # Display line chart
    if not chart_data.empty:
        st.line_chart(chart_data['Close'])
    else:
        st.warning("No data available for the selected period.")
    

        # --- LSTM-Based Stock Price Prediction (Notebook-Aligned) ---
    st.subheader("LSTM-Based Stock Price Prediction")

    df = yf.download(ticker, period="10y", interval="1d")[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    if df.empty:
        st.warning("No valid OHLCV data found for this ticker.")
    else:
        data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        dataset = data.values

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(dataset)

        x_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            x_train.append(scaled_data[i-60:i])
            y_train.append(scaled_data[i, 3])  # 'Close' column

        x_train, y_train = np.array(x_train), np.array(y_train)

        @st.cache_resource
        def train_lstm_model(x_train, y_train, input_shape):
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
            model.add(LSTM(50))
            model.add(Dense(25))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=0)
            return model

        # Cache and retrieve model
        model = train_lstm_model(x_train, y_train, (x_train.shape[1], x_train.shape[2]))

        last_60_days = scaled_data[-60:]
        X_test = np.array([last_60_days])
        predicted_close_scaled = model.predict(X_test)

        dummy_row = last_60_days[-1].copy()
        dummy_row[3] = predicted_close_scaled[0][0]  # Replace scaled 'Close'
        predicted_full = scaler.inverse_transform([dummy_row])
        predicted_price = predicted_full[0][3]

    st.success(f"**Predicted Closing Price for Next Day:** ${predicted_price:.2f}")