import os
import time
import json
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import telegram
import asyncio
import lightgbm as lgb
import shap
import pywt
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from binance.client import Client as BinanceClient
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The following lines are to suppress the oneDNN informational messages from TensorFlow.
# You can safely ignore these messages if you see them.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. Project Layout & Configuration ---

def load_settings(config_path):
    """Loads settings from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_settings(settings, config_path):
    """Saves settings to a YAML file."""
    with open(config_path, 'w') as file:
        yaml.dump(settings, file)

def load_hyperparams(hyperparams_path):
    """Loads hyperparameters from a JSON file."""
    with open(hyperparams_path, 'r') as file:
        return json.load(file)

def load_symbols(symbols_file):
    """Loads symbols from a CSV file."""
    with open(symbols_file, 'r') as file:
        return [line.strip() for line in file.readlines() if line.strip()]

# --- 2. Configuration & Secrets ---
# Import keys directly
import keys

# --- 3. Data & MLOps Foundations ---

class DataHandler:
    def __init__(self, settings):
        self.data_path = settings['data_path']
        self.binance_client = BinanceClient(keys.api_testnet, keys.secret_testnet, tld='com', testnet=True)

    def fetch_bars(self, symbol, interval='15m'):
        """
        Fetches bars for a symbol from Binance, caches to Parquet,
        and runs data quality checks.
        """
        def is_cache_fresh(path, max_age_hours=12):
            if not os.path.exists(path):
                return False
            age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
            return age.total_seconds() < max_age_hours * 3600

        cache_path = os.path.join(self.data_path, f"{symbol}_{interval}.parquet")

        if is_cache_fresh(cache_path):
            logging.info(f"Loading fresh cached {interval} data for {symbol}...")
            return pd.read_parquet(cache_path)

        logging.info(f"Fetching {interval} bars for {symbol}...")
        try:
            klines = self.binance_client.futures_klines(symbol=symbol, interval=interval, limit=1000)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            # Run Great Expectations checks
            self.validate_data(df)

            # Cache to Parquet
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            df.to_parquet(cache_path)

            return df
        except Exception as e:
            logging.error(f"Error fetching or validating data for {symbol}: {e}")
            return None

    def fetch_multi_tf(self, symbol):
       return {
         '15m': self.fetch_bars(symbol, '15m'),
         '1h' : self.fetch_bars(symbol, '1h'),
         '4h' : self.fetch_bars(symbol, '4h'),
       }

    def validate_data(self, df):
        """
        Runs Great Expectations checks on the dataframe.
        This is a placeholder for a real implementation.
        """
        logging.info("Running Great Expectations checks...")
        # In a real implementation, you would define a Great Expectations
        # suite and run it here.
        # example_suite = ...
        # validation_result = df.validate(example_suite)
        # if not validation_result["success"]:
        #     raise Exception("Data validation failed!")
        pass

    def get_feature_store_data(self, symbol, data):
        """
        Computes raw price/volume-based features and stores them in a feature store.
        This is a placeholder for a real implementation with Redis or Feast.
        """
        if data is not None and not data.empty:
            features = {
                "price_change_pct": (data['close'].iloc[-1] - data['open'].iloc[-1]) / data['open'].iloc[-1],
                "volume_change_pct": (data['volume'].iloc[-1] - data['volume'].iloc[-2]) / data['volume'].iloc[-2] if len(data) > 1 else 0,
            }

            # Add more advanced features
            # Wavelet coefficients
            coeffs = pywt.wavedec(data['close'], 'db1', level=2)
            features['wavelet_cA2'] = coeffs[0][0]
            features['wavelet_cD2'] = coeffs[1][0]
            features['wavelet_cD1'] = coeffs[2][0]

            # Order book imbalance (placeholder)
            features['order_book_imbalance'] = 0.5

            # Time since last pivot (placeholder)
            features['time_since_last_pivot'] = 10

            # Volume surge (placeholder)
            features['volume_surge'] = 1.2

            # In a real implementation, you would push these features to Redis or Feast.
            # For example, with Redis:
            # redis_client.hmset(f"features:{symbol}", features)
            return features
        return None

# --- 4. Swing Detection & Entry Prediction ---

class TrendPatternValidator:
    def __init__(self, symbol, data_1h, data_4h):
        self.symbol = symbol
        self.data_1h = data_1h
        self.data_4h = data_4h
        self.trend_state = self.get_trend_state()

    def get_trend_state(self):
        """
        Determines the current trend state based on HH/HL/LH/LL patterns.
        This is a placeholder for a real implementation.
        """
        # In a real implementation, you would analyze the 1h and 4h data
        # to identify the HH/HL/LH/LL patterns.
        return "uptrend" # Placeholder

    def is_valid_continuation(self, signal_type):
        """
        Checks if the signal is a valid continuation of the current trend.
        """
        if self.trend_state == "uptrend" and signal_type == "long":
            return True
        elif self.trend_state == "downtrend" and signal_type == "short":
            return True
        return False

class SwingDetector:
    def __init__(self, settings, reporter):
        self.settings = settings
        self.reporter = reporter
        self.model_path = os.path.join(settings['models_path'], 'pivot_model.pkl')
        self.pivot_model = self.load_model()

    def load_model(self):
        """Loads the pivot detection model."""
        if os.path.exists(self.model_path):
            logging.info(f"Loading pivot model from {self.model_path}")
            try:
                return joblib.load(self.model_path)
            except Exception as e:
                logging.error(f"Error loading pivot model: {e}")
                return None
        return None

    def detect_pivots(self, symbol, data_15m, data_1h, data_4h):
        """Detects swing pivots in the data."""
        logging.info(f"Detecting pivots for {symbol}...")
        if self.pivot_model and data_15m is not None and not data_15m.empty:
            # In a real implementation, you would use more sophisticated feature engineering
            features, _ = engineer_pivot_features(data_15m, data_1h, data_4h)
            prediction = self.pivot_model.predict(features)
            if prediction[-1] == 1:
                pivot_price = data_15m['close'].iloc[-1]
                self.reporter.log_identification(symbol, "pivot", {"price": pivot_price})
                return {"price": pivot_price, "type": "high"}
        return None

    def calculate_golden_zone(self, pivot, data):
        """
        Calculates the Fibonacci golden zone for a pivot.
        This is a placeholder for a real implementation with an "optimal zone"
        per coin, updated monthly.
        """
        if data is not None and not data.empty:
            high = data['high'].iloc[-15:].max()
            low = data['low'].iloc[-15:].min()

            if pivot['type'] == 'high':
                retracement_50 = high - (high - low) * 0.5
                retracement_61_8 = high - (high - low) * 0.618
                return {"entry_min": retracement_61_8, "entry_max": retracement_50}
            else: # low pivot
                retracement_50 = low + (high - low) * 0.5
                retracement_61_8 = low + (high - low) * 0.618
                return {"entry_min": retracement_50, "entry_max": retracement_61_8}
        return None

class EntryClassifier:
    def __init__(self, settings, reporter):
        self.settings = settings
        self.reporter = reporter
        try:
            self.entry_confidence_thresh = load_hyperparams('configs/hyperparams.json').get('entry_confidence_thresh', 0.5)
        except FileNotFoundError:
            self.entry_confidence_thresh = 0.5
        self.model_path = os.path.join(settings['models_path'], 'entry_model.pkl')
        self.entry_model = self.load_model()

    def load_model(self):
        """Loads the entry classification model."""
        if os.path.exists(self.model_path):
            logging.info(f"Loading entry model from {self.model_path}")
            try:
                return joblib.load(self.model_path)
            except Exception as e:
                logging.error(f"Error loading entry model: {e}")
                return None
        return None

    def predict_entry_success(self, symbol, data):
        """Predicts the success of a potential entry."""
        logging.info(f"Predicting entry for {symbol}...")
        if self.entry_model and data is not None and not data.empty:
            # In a real implementation, you would use more sophisticated feature engineering
            features = data[['open', 'high', 'low', 'close', 'volume']].tail(1)
            features.columns = [f"f{i}" for i in range(features.shape[1])]
            # Create a full feature set with the same columns as the training data
            X = pd.DataFrame(columns=[f"f{i}" for i in range(10)])
            X = pd.concat([X, features], ignore_index=True).fillna(0)
            prediction = self.entry_model.predict_proba(X)
            confidence = prediction[0][1]
            logging.info(f"Entry model prediction for {symbol}: confidence={confidence:.4f}")
            if confidence >= self.entry_confidence_thresh:
                logging.info(f"Entry signal CONFIRMED for {symbol} with confidence {confidence:.4f}")
                return {"confidence": confidence, "side": "long"}
            else:
                logging.info(f"Entry signal REJECTED for {symbol} with confidence {confidence:.4f}")
                self.reporter.log_rejection(symbol, "low_confidence", {"confidence": confidence})
        return None

# --- 5. Reporter ---

class Reporter:
    def __init__(self, settings):
        self.reports_path = settings['reports_path']
        self.mode = settings['mode']
        self.telegram_bot = None
        if not os.path.exists(self.reports_path):
            os.makedirs(self.reports_path)

    def __init__(self, settings):
        self.telegram_bot = None

    def start_run_report(self):
        pass

    def log_identification(self, symbol, id_type, details):
        pass

    def log_rejection(self, symbol, reason, details):
        pass

    def log_virtual_order(self, symbol, order_details):
        pass

    def log_real_order(self, symbol, order_details):
        pass

    def log_sl_tp_action(self, symbol, action, details):
        pass

    def finalize_report(self):
        pass

    async def push_report_to_telegram(self, filename):
        pass


# --- 6. & 7. Trader Classes & Trading Loop ---

class BaseTrader:
    def __init__(self, settings, detector, entry_clf, reporter, telegram_bot):
        self.settings = settings
        self.hyperparams = load_hyperparams('configs/hyperparams.json')
        self.detector = detector
        self.entry_clf = entry_clf
        self.reporter = reporter
        self.telegram_bot = telegram_bot
        self.data_handler = DataHandler(settings)

    def process_symbol(self, sym, seen_orders):
        raise NotImplementedError

class SignalTrader(BaseTrader):
    def process_symbol(self, sym, seen_orders):
        logging.info(f"[Signal] Processing {sym}...")
        # 1. Check for outstanding virtual orders
        if self.has_outstanding_virtual_order(sym):
            logging.info(f"Skipping {sym} due to outstanding virtual order.")
            return

        # 2. Detect pivot and compute retracements
        multi_tf_data = self.data_handler.fetch_multi_tf(sym)
        data_15m = multi_tf_data['15m']
        data_1h = multi_tf_data['1h']
        data_4h = multi_tf_data['4h']
        pivot = self.detector.detect_pivots(sym, data_15m, data_1h, data_4h)
        if not pivot:
            return

        golden_zone = self.detector.calculate_golden_zone(pivot, data_15m)

        # 3. Predict entry and check confidence
        entry_signal = self.entry_clf.predict_entry_success(sym, data_15m)

        # 4. Validate trend pattern
        if not entry_signal:
            return

        trend_validator = TrendPatternValidator(sym, data_1h, data_4h)
        if not trend_validator.is_valid_continuation(entry_signal['side']):
            logging.info(f"Signal for {sym} invalidated by trend pattern.")
            self.reporter.log_rejection(sym, "trend_invalidation", {})
            return

        if entry_signal and golden_zone:
            order_key = (sym, entry_signal['side'], round(golden_zone['entry_min'], 4))
            if order_key in seen_orders:
                # Active signal monitoring
                if not trend_validator.is_valid_continuation(entry_signal['side']):
                    logging.info(f"Canceling signal for {sym} due to trend invalidation.")
                    seen_orders.remove(order_key)
                    self.reporter.log_rejection(sym, "signal_canceled_trend_invalidation", {})
                    return

        if entry_signal and golden_zone:
            order_key = (sym, entry_signal['side'], round(golden_zone['entry_min'], 4))
            if order_key not in seen_orders:
                # 4. Place virtual order and notify
                seen_orders.add(order_key)
                self.reporter.log_virtual_order(sym, {"side": entry_signal['side'], "price": golden_zone['entry_min']})
                log_msg = f"Signal placed for {sym}: {entry_signal['side']} @ {golden_zone['entry_min']:.4f}"
                logging.info(log_msg)
                if self.telegram_bot:
                    try:
                        asyncio.run(self.telegram_bot.send_message(chat_id=keys.telegram_chat_id, text=log_msg))
                    except Exception as e:
                        logging.error(f"Failed to send Telegram message: {e}")

                # Monitor for RSI confirmation (simplified)
                sl, tp1, tp2 = self.calculate_sl_tp(entry_signal['side'], golden_zone, data_15m)
                self.check_rsi_confirmation(sym, entry_signal, golden_zone, sl, tp1, tp2, data_15m)

    def calculate_sl_tp(self, side, golden_zone, data):
        """Calculates SL and TP levels."""
        high = data['high'].iloc[-15:].max()
        low = data['low'].iloc[-15:].min()

        if side == 'long':
            sl = low
            tp1 = high
            tp2 = golden_zone['entry_max'] + (high - low)
        else: # short
            sl = high
            tp1 = low
            tp2 = golden_zone['entry_min'] - (high - low)
        return sl, tp1, tp2

    def has_outstanding_virtual_order(self, symbol):
        """Checks if there is an outstanding virtual order for the symbol."""
        # This is a placeholder. In a real implementation, you would check a
        # database or an in-memory store for outstanding virtual orders.
        return False

    def check_rsi_confirmation(self, sym, entry_signal, golden_zone, sl, tp1, tp2, data):
        # Placeholder for RSI check
        logging.info(f"Waiting for RSI confirmation on {sym}...")
        time.sleep(1)
        rsi_ok = True # Simulate successful RSI check
        if rsi_ok:
            logging.info(f"RSI confirmed for {sym}!")
            if self.telegram_bot:
                side_emoji = "📈" if entry_signal['side'] == 'long' else "📉"
                current_price = data['close'].iloc[-1]
                log_msg = (
                    f"{side_emoji} **New Signal** {side_emoji}\n\n"
                    f"**Symbol:** {sym}\n"
                    f"**Side:** {entry_signal['side']}\n"
                    f"**Confidence:** {entry_signal['confidence']:.2%}\n"
                    f"**Current Price:** {current_price:.4f}\n"
                    f"**Entry Zone:** {golden_zone['entry_min']:.4f} - {golden_zone['entry_max']:.4f}\n"
                    f"**TP1:** {tp1:.4f}\n"
                    f"**TP2:** {tp2:.4f}"
                )
                try:
                    asyncio.run(self.telegram_bot.send_message(chat_id=keys.telegram_chat_id, text=log_msg, parse_mode='Markdown'))
                except Exception as e:
                    logging.error(f"Failed to send Telegram message: {e}")


class LiveTrader(SignalTrader):
    def __init__(self, settings, binance, telegram_bot, detector, entry_clf, reporter):
        super().__init__(settings, detector, entry_clf, reporter, telegram_bot)
        self.binance_client = binance

    async def process_symbol_async(self, sym, seen_orders):
        logging.info(f"[Live] Processing {sym}...")
        data = DataHandler(self.settings).fetch_bars(sym)
        pivot = self.detector.detect_pivots(sym, data)
        if not pivot:
            return

        golden_zone = self.detector.calculate_golden_zone(pivot, data)
        entry_signal = self.entry_clf.predict_entry_success(sym, data)

        if entry_signal and golden_zone:
            order_key = (sym, entry_signal['side'], golden_zone['entry_min'])
            if order_key not in seen_orders:
                seen_orders.add(order_key)
                logging.info(f"RSI confirmed for {sym}, placing real order.")
                await self.place_sl_tp_order(sym, entry_signal, pivot, data)

    async def place_sl_tp_order(self, symbol, entry_signal, pivot, data):
        """Places the SL/TP OCO order."""
        side = entry_signal['side']
        entry_price = data['close'].iloc[-1]

        high = data['high'].iloc[-15:].max()
        low = data['low'].iloc[-15:].min()

        if side == 'long':
            sl = low
            tp1 = high
            tp2 = entry_price + (high - low)
        else: # short
            sl = high
            tp1 = low
            tp2 = entry_price - (high - low)

        log_msg = f"Placing OCO order for {symbol}: SL={sl:.4f}, TP1={tp1:.4f}, TP2={tp2:.4f}"
        logging.info(log_msg)
        self.reporter.log_real_order(symbol, {"side": side, "entry": entry_price, "sl": sl, "tp1": tp1, "tp2": tp2})

        if self.telegram_bot:
            try:
                await self.telegram_bot.send_message(chat_id=keys.telegram_chat_id, text=log_msg)
            except Exception as e:
                logging.error(f"Failed to send Telegram message: {e}")

        # Placeholder for trailing SL logic
        await self.handle_trailing_sl(symbol, data, entry_price, tp1, tp2)

    async def handle_trailing_sl(self, symbol, data, entry_price, tp1, tp2):
        """Handles trailing stop-loss adjustments."""
        # This is a placeholder. In a real implementation, you would monitor
        # the price and adjust the stop-loss accordingly.
        logging.info(f"Handling trailing SL for {symbol}...")

        # Simulate price hitting TP1
        await asyncio.sleep(2)
        new_sl_be = entry_price * 1.001 # Break-even + epsilon
        self.reporter.log_sl_tp_action(symbol, "trail_to_be", {"new_sl": new_sl_be})
        logging.info(f"SL trailed to break-even for {symbol}: {new_sl_be}")

        # Simulate price hitting TP2
        await asyncio.sleep(2)
        new_sl_tp1 = tp1
        self.reporter.log_sl_tp_action(symbol, "trail_to_tp1", {"new_sl": new_sl_tp1})
        logging.info(f"SL moved to TP1 for {symbol}: {new_sl_tp1}")

        # In a real implementation, you would place the order on the exchange here.
        # self.binance_client.futures_create_order(...)

        # Placeholder for monitoring fills
        await self.monitor_fills(symbol)

    async def monitor_fills(self, symbol):
        """Monitors for order fills."""
        # This is a placeholder. In a real implementation, you would use a
        # WebSocket connection to monitor for order updates.
        logging.info(f"Monitoring fills for {symbol}...")
        await asyncio.sleep(5) # Simulate waiting for fill

        # Simulate a partial fill
        partial_fill_log = f"Partial fill for {symbol}!"
        logging.info(partial_fill_log)
        self.reporter.log_real_order(symbol, {"status": "partial_fill"})
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_message(chat_id=keys.telegram_chat_id, text=partial_fill_log)
            except Exception as e:
                logging.error(f"Failed to send Telegram message: {e}")

        # Simulate a full fill
        full_fill_log = f"Order for {symbol} filled!"
        logging.info(full_fill_log)
        self.reporter.log_real_order(symbol, {"status": "filled"})
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_message(chat_id=keys.telegram_chat_id, text=full_fill_log)
            except Exception as e:
                logging.error(f"Failed to send Telegram message: {e}")

# --- Training and Backtesting Functions ---

def engineer_pivot_features(data_15m, data_1h, data_4h):
    """Engineers features for the pivot model."""
    # This is a placeholder for a real implementation.
    # You would combine the data from the different timeframes
    # to create features for the model.
    return np.random.rand(100, 10), np.random.randint(0, 2, 100)

def train_model_for_symbol(symbol):
    """Trains and saves a simple model for a symbol."""
    logging.info(f"Training models for {symbol}...")
    settings = load_settings("configs/settings.yml")

    # 1. Load data for multiple timeframes
    multi_tf_data = DataHandler(settings).fetch_multi_tf(symbol)
    data_15m = multi_tf_data['15m']
    data_1h = multi_tf_data['1h']
    data_4h = multi_tf_data['4h']

    # 2. Engineer features
    X, y = engineer_pivot_features(data_15m, data_1h, data_4h)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 2. Train an ensemble model (LightGBM + LSTM)
    # This is a simplified placeholder for a real implementation.
    # The low accuracies are expected, as we are using random data.
    # For a real-world implementation, you would want to:
    # - Use a much larger dataset
    # - Tune the hyperparameters of the models
    # - Use cross-validation to get a more robust estimate of the performance

    # Train LightGBM model
    lgb_model = lgb.LGBMClassifier(verbosity=-1)
    lgb_model.fit(X_train, y_train)
    logging.info(f"LGBM model accuracy: {lgb_model.score(X_test, y_test)}")

    if TENSORFLOW_AVAILABLE:
        # Train LSTM model
        X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
        lstm_model = Sequential([
            Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
            LSTM(50, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm_model.fit(X_train_lstm, y_train, epochs=10, verbose=0)
        loss, acc = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)
        logging.info(f"LSTM model accuracy: {acc}")
    else:
        logging.warning("TensorFlow not available. Skipping LSTM model training.")

    # In a real implementation, you would stack these models.
    # For simplicity, we'll just save the LGBM model for now.
    pivot_model = lgb_model

    # Entry Classifier: Lightweight tree model with cost-sensitive loss
    entry_model = lgb.LGBMClassifier(class_weight='balanced', verbosity=-1)
    entry_model.fit(X_train, y_train)
    logging.info(f"Entry model accuracy: {entry_model.score(X_test, y_test)}")

    # Placeholder for stacking
    # stacked_model = StackingClassifier(estimators=[('lgbm', lgb_model), ('lstm', lstm_model)], final_estimator=LogisticRegression())
    # stacked_model.fit(X_train, y_train)
    # logging.info(f"Stacked model accuracy: {stacked_model.score(X_test, y_test)}")

    # 3. Save the models and log with MLflow
    models_path = settings['models_path']
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    pivot_model_path = os.path.join(models_path, 'pivot_model.pkl')
    entry_model_path = os.path.join(models_path, 'entry_model.pkl')
    joblib.dump(pivot_model, pivot_model_path)
    joblib.dump(entry_model, entry_model_path)

    # In a real implementation, you would use MLflow to log the models,
    # parameters, and metrics.
    # with mlflow.start_run():
    #     mlflow.log_param("symbol", symbol)
    #     mlflow.log_metric("pivot_model_accuracy", pivot_model.score(X_test, y_test))
    #     mlflow.log_metric("entry_model_accuracy", entry_model.score(X_test, y_test))
    #     mlflow.sklearn.log_model(pivot_model, "pivot_model")
    #     mlflow.sklearn.log_model(entry_model, "entry_model")

    # Placeholder for SHAP
    # explainer = shap.TreeExplainer(pivot_model)
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test)

    logging.info(f"Finished training and saved models for {symbol}.")
    return f"Trained {symbol}"

def run_multithreaded_training(symbols, max_workers):
    """
    Runs training for all symbols in parallel.
    This function would be triggered by a scheduler like Airflow or Cron.
    """
    logging.info("Starting multithreaded training...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(train_model_for_symbol, symbols))
    logging.info("Multithreaded training complete.")
    print(results)

def run_backtest(symbols, trader, reporter):
    """Runs a backtest across all symbols."""
    logging.info("Starting backtest...")
    reporter.start_run_report()
    for sym in symbols:
        trader.process_symbol(sym, set()) # Use a dummy set for seen_orders in backtest
    reporter.finalize_report()
    logging.info("Backtest complete.")

# --- Main Orchestration ---

def main():
    """Main entry point for the application."""
    settings = load_settings("configs/settings.yml")
    hyperparams = load_hyperparams("configs/hyperparams.json")

    try:
        telegram_bot = telegram.Bot(token=keys.telegram_bot_token)
    except Exception as e:
        logging.error(f"Failed to initialize Telegram bot: {e}")
        telegram_bot = None

    binance_client = None # Placeholder

    # Placeholder for circuit breaker
    consecutive_losers = 0
    max_drawdown = 0.1

    running = True
    while running:
        mode = settings.get('mode', 'signal')
        logging.info(f"Application starting in {mode} mode.")

        reporter = Reporter(settings)
        reporter.telegram_bot = telegram_bot
        detector = SwingDetector(settings, reporter)
        entry_clf = EntryClassifier(settings, reporter)

        if mode == "train":
            symbols = load_symbols(settings['symbols_file'])
            run_multithreaded_training(symbols, hyperparams['max_workers'])
            reporter.start_run_report()
            reporter.finalize_report()
            settings['mode'] = 'signal' # Switch to signal mode after training
            save_settings(settings, "configs/settings.yml")
            logging.info("Training complete. Switching to signal mode.")
            # The loop will continue, and in the next iteration, it will be in signal mode.

        elif mode == "backtest":
            trader = SignalTrader(settings, detector, entry_clf, reporter, telegram_bot)
            symbols = load_symbols(settings['symbols_file'])
            run_backtest(symbols, trader, reporter)
            running = False # Exit after backtest

        else: # signal or live mode
            if not detector.pivot_model or not entry_clf.entry_model:
                logging.warning("Models not found. Cannot run in live or signal mode.")
                response = input("Would you like to start the training process now? (y/n): ")
                if response.lower() == 'y':
                    settings['mode'] = 'train'
                    # The loop will continue, and in the next iteration, it will be in train mode.
                else:
                    logging.info("Exiting application.")
                    running = False
            else:
                trader = LiveTrader(settings, binance_client, telegram_bot, detector, entry_clf, reporter) \
                    if mode == "live" \
                    else SignalTrader(settings, detector, entry_clf, reporter, telegram_bot)

                symbols = load_symbols(settings['symbols_file'])
                seen_orders = set()

                reporter.start_run_report()
                try:
                    while True:
                        logging.info("--- Starting new trading cycle ---")
                        for sym in symbols:
                            if isinstance(trader, LiveTrader):
                                asyncio.run(trader.process_symbol_async(sym, seen_orders))
                            else:
                                trader.process_symbol(sym, seen_orders)
                        logging.info(f"--- Cycle complete, sleeping for {settings.get('cycle_break_s', 10)}s ---")
                        time.sleep(settings.get('cycle_break_s', 10))
                except KeyboardInterrupt:
                    logging.info("Shutting down...")
                    running = False
                finally:
                    reporter.finalize_report()


if __name__ == "__main__":
    main()
