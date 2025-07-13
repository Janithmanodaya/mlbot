import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from app import SignalTrader, SwingDetector, EntryClassifier, Reporter

class TestSignalTrader(unittest.TestCase):
    def setUp(self):
        self.settings = {
            'reports_path': 'reports',
            'mode': 'test',
            'pivot_lookback': 50,
            'feature_lookback': 100,
            'rsi_period': 14,
            'sltp_lookback': 15,
            'max_virtual_orders_per_symbol': 1,
            'order_expiry_cycles': 10,
            'pivot_confidence_thresh': 0.6
        }
        self.reporter = Reporter(self.settings)
        self.detector = SwingDetector(self.settings, self.reporter)
        self.entry_clf = EntryClassifier(self.settings, self.reporter)
        self.trader = SignalTrader(self.settings, self.detector, self.entry_clf, self.reporter, None)

    def test_rsi_ok(self):
        data = pd.DataFrame({'close': np.linspace(100, 150, 100)})
        self.assertFalse(self.trader.rsi_ok('long', data, 14))
        data = pd.DataFrame({'close': np.linspace(150, 100, 100)})
        self.assertFalse(self.trader.rsi_ok('short', data, 14))

        data = pd.DataFrame({'close': np.random.rand(100) * 10 + 100})
        data['close'].iloc[-1] = 90
        self.assertFalse(self.trader.rsi_ok('long', data, 14))
        data['close'].iloc[-1] = 110
        self.assertFalse(self.trader.rsi_ok('short', data, 14))

    def test_calculate_sl_tp(self):
        data = pd.DataFrame({'high': np.linspace(100, 150, 20), 'low': np.linspace(90, 140, 20)})
        sl, tp1, tp2 = self.trader.calculate_sl_tp('long', {'entry_max': 145}, data, 15)
        self.assertAlmostEqual(sl, 129.4736842105263)
        self.assertAlmostEqual(tp1, 150.0)
        self.assertAlmostEqual(tp2, 165.5263157894737)

    def test_detect_pivots(self):
        with patch.object(self.detector.pivot_model, 'predict_proba', return_value=np.array([[0.1, 0.9]])):
            data = pd.DataFrame({'close': np.linspace(100, 150, 100)})
            pivot = self.detector.detect_pivots('BTCUSDT', data, data, data, 50)
            self.assertIsNotNone(pivot)
            self.assertEqual(pivot['type'], 'high')

        with patch.object(self.detector.pivot_model, 'predict_proba', return_value=np.array([[0.9, 0.1]])):
            data = pd.DataFrame({'close': np.linspace(100, 150, 100)})
            pivot = self.detector.detect_pivots('BTCUSDT', data, data, data, 50)
            self.assertIsNone(pivot)

if __name__ == '__main__':
    unittest.main()
