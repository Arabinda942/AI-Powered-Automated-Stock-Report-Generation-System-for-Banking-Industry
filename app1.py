# ============================================
# AI STOCK REPORT SYSTEM - FINAL APP (FIXED)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("📊 AI Stock Analysis Dashboard")

# ============================================
# NIFTY 50 STOCKS (TATAMOTORS REMOVED)
# ============================================
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","KOTAKBANK.NS","LT.NS","SBIN.NS",
    "BHARTIARTL.NS","ASIANPAINT.NS","AXISBANK.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS",
    "WIPRO.NS","HCLTECH.NS","ADANIENT.NS","ADANIPORTS.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","POWERGRID.NS","NTPC.NS",
    "ONGC.NS","COALINDIA.NS","JSWSTEEL.NS","TATASTEEL.NS",
    "INDUSINDBK.NS","TECHM.NS","HDFCLIFE.NS","SBILIFE.NS",
    "CIPLA.NS","DRREDDY.NS","DIVISLAB.NS","GRASIM.NS",
    "EICHERMOT.NS","HEROMOTOCO.NS","BPCL.NS","IOC.NS",
    "HINDALCO.NS","APOLLOHOSP.NS","BRITANNIA.NS"
]

# ============================================
# FILE UPLOAD OR LIVE DATA
# ============================================
st.sidebar.header("📁 Data Source")

mode = st.sidebar.radio("Select Mode", ["Upload File", "Live Data (NIFTY 50)"])
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=None)

# ============================================
# CLEANING FUNCTION (IMPORTANT FIX)
# ============================================
def clean_data(df):

    df = df.copy()

    # Remove unwanted index column
    df = df.drop(columns=["Sl No."], errors="ignore")

    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    # Clean Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Clean numeric columns
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["Date","Ticker","Close"])

    return df

# ============================================
# LOAD UPLOADED DATA
# ============================================
def load_uploaded(file):
    try:
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        return clean_data(df)

    except Exception as e:
        st.error(f"Error: {e}")
        return None


# ============================================
# LIVE DATA
# ============================================
@st.cache_data
def load_live():
    data = yf.download(NIFTY50, period="2y", group_by="ticker", threads=False)

    data.index = pd.to_datetime(data.index)

    df = data.stack(level=0).reset_index()
    df.columns = ["Date","Ticker","Open","High","Low","Close","Volume"]

    return clean_data(df)

# ============================================
# SELECT DATA
# ============================================
if mode == "Upload File":
    if uploaded_file:
        df = load_uploaded(uploaded_file)
    else:
        st.warning("Upload file to continue")
        st.stop()
else:
    df = load_live()

# ============================================
# CLEANING (FINAL SAFETY PASS)
# ============================================
df = df.dropna()
df = df.sort_values(["Ticker","Date"])
df.reset_index(drop=True, inplace=True)

# ============================================
# FEATURE ENGINEERING
# ============================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

df["Return"] = df.groupby("Ticker")["Close"].pct_change()
df["MA20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
df["MA50"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50).mean())
df["RSI"] = df.groupby("Ticker")["Close"].transform(compute_rsi)

df["EMA12"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=12).mean())
df["EMA26"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=26).mean())
df["MACD"] = df["EMA12"] - df["EMA26"]
df["MACD_Signal"] = df.groupby("Ticker")["MACD"].transform(lambda x: x.ewm(span=9).mean())

df["Volatility"] = df.groupby("Ticker")["Return"].transform(lambda x: x.rolling(20).std())

df["Volume_MA"] = df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(20).mean())
df["Volume_Spike"] = df["Volume"] > (df["Volume_MA"] * 1.5)

df["Target"] = np.where(df.groupby("Ticker")["Close"].shift(-1) > df["Close"], 1, 0)

df = df.dropna()

# ============================================
# MODELS
# ============================================
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "DecisionTree": DecisionTreeClassifier(),
    "GradientBoost": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

features = ["Return","MA20","MA50","RSI","Volatility"]

X = df[features]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle=False,test_size=0.2)

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# ============================================
# PREDICTIONS
# ============================================
for name, model in trained_models.items():
    df[name] = model.predict(X)

df["Vote"] = df[list(models.keys())].sum(axis=1)

df["Signal"] = np.where(df["Vote"]>0,1,np.where(df["Vote"]<0,-1,0))
df["Signal_Label"] = df["Signal"].map({1:"BUY",-1:"SELL",0:"HOLD"})

df["Confidence"] = (abs(df["Vote"])/len(models))*100

df["Risk"] = np.where(df["Volatility"]<0.01,"Low",
              np.where(df["Volatility"]<0.02,"Medium","High"))

df["Stop_Loss"] = np.where(df["Signal"]==1, df["Close"]*0.97,
                  np.where(df["Signal"]==-1, df["Close"]*1.03, np.nan))

df["Target_Price"] = np.where(df["Signal"]==1, df["Close"]*1.05,
                     np.where(df["Signal"]==-1, df["Close"]*0.95, np.nan))

# ============================================
# SIDEBAR FILTERS
# ============================================
stocks = df["Ticker"].unique()
stock = st.sidebar.selectbox("Stock", stocks)

date_range = st.sidebar.date_input("Date Range", [])

signal_filter = st.sidebar.selectbox("Signal", ["All","BUY","SELL","HOLD"])
ma_filter = st.sidebar.selectbox("MA Filter", ["All","Bullish","Bearish"])
rsi_filter = st.sidebar.selectbox("RSI Filter", ["All","Overbought","Oversold"])

auto_refresh = st.sidebar.checkbox("Auto Refresh")

# ============================================
# FILTER DATA
# ============================================
stock_df = df[df["Ticker"]==stock]

if len(date_range)==2:
    stock_df = stock_df[
        (stock_df["Date"]>=pd.to_datetime(date_range[0])) &
        (stock_df["Date"]<=pd.to_datetime(date_range[1]))
    ]

if signal_filter!="All":
    stock_df = stock_df[stock_df["Signal_Label"]==signal_filter]

if ma_filter=="Bullish":
    stock_df = stock_df[stock_df["MA20"]>stock_df["MA50"]]
elif ma_filter=="Bearish":
    stock_df = stock_df[stock_df["MA20"]<stock_df["MA50"]]

if rsi_filter=="Overbought":
    stock_df = stock_df[stock_df["RSI"]>70]
elif rsi_filter=="Oversold":
    stock_df = stock_df[stock_df["RSI"]<30]

# ============================================
# DISPLAY
# ============================================
st.subheader("📄 Data")
st.dataframe(stock_df.tail(20))

csv = stock_df.to_csv(index=False)
st.download_button("Download CSV", csv, "filtered_data.csv")

# Candlestick
st.subheader("🟥🟩 Chart Candlestick")
fig = go.Figure(data=[go.Candlestick(
    x=stock_df["Date"],
    open=stock_df["Open"],
    high=stock_df["High"],
    low=stock_df["Low"],
    close=stock_df["Close"]
)])
st.plotly_chart(fig, use_container_width=True)

# Indicators
st.subheader("📈 Indicators")
st.line_chart(stock_df.set_index("Date")[["Close","MA20","MA50"]])
st.line_chart(stock_df.set_index("Date")[["RSI"]])
st.line_chart(stock_df.set_index("Date")[["MACD","MACD_Signal"]])

# ============================================
# METRICS
# ============================================
latest = stock_df.iloc[-1]
c1,c2,c3 = st.columns(3)
c1.metric("Signal", latest["Signal_Label"])
c2.metric("Confidence %", round(latest["Confidence"],2))
c3.metric("Risk", latest["Risk"])

# ============================================
# TOP 5
# ============================================
latest_all = df.groupby("Ticker").tail(1)

st.subheader("🔥 Top 5 Bullish")
st.dataframe(latest_all.sort_values("Vote",ascending=False).head(5))

st.subheader("🔻 Top 5 Bearish")
st.dataframe(latest_all.sort_values("Vote").head(5))

# ============================================
# MODEL BREAKDOWN
# ============================================
st.subheader("🤖 Model Predictions")
st.dataframe(stock_df[list(models.keys()) + ["Signal_Label"]].tail(10))

# ============================================
# AUTO REFRESH
# ============================================
if auto_refresh:
    time.sleep(10)
    st.rerun()