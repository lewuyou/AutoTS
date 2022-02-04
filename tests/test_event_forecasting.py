# -*- coding: utf-8 -*-
"""
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
from autots import (
    load_weekly,
    load_daily,
    EventRiskForecast,
)
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# SeasonalNaive 365
# write a unit test (uses a mix of manual and forecast limits from motif result windows)
    # one with motif model
    # one with non-motif model (pass pred intervals)
    # all four limit inputs tested
    # univariate case

forecast_length = 6
df_full = load_weekly(long=False)
df = df_full[0: (df_full.shape[0] - forecast_length)]
df_test = df[(df.shape[0] - forecast_length):]

upper_limit = 0.95
# if using manual array limits, historic limit must be defined separately (if used)
lower_limit = np.ones((forecast_length, df.shape[1]))
historic_lower_limit = np.ones(df.shape)

model = EventRiskForecast(
    df,
    forecast_length=forecast_length,
    upper_limit=upper_limit,
    lower_limit=lower_limit,
)
# .fit() is optional if model_name, model_param_dict, model_transform_dict are already defined (overwrites)
model.fit()
risk_df_upper, risk_df_lower = model.predict()
historic_upper_risk_df, historic_lower_risk_df = model.predict_historic(lower_limit=historic_lower_limit)
model.plot(0)
save_fig = False
if save_fig:
    plt.savefig("event_forecasting_3.png", dpi=300)

# also eval summed version
threshold = 0.1
eval_lower = EventRiskForecast.generate_historic_risk_array(df_test, model.lower_limit_2d, direction="lower")
eval_upper = EventRiskForecast.generate_historic_risk_array(df_test, model.upper_limit_2d, direction="upper")
pred_lower = np.where(model.lower_risk_array > threshold, 1, 0)
pred_upper = np.where(model.upper_risk_array > threshold, 1, 0)

"""
Precision = minimize false positives (so 1 = no false positives)
Recall = minimize false negatives
"""
multilabel_confusion_matrix(eval_upper, pred_upper).sum(axis=0)
print(classification_report(eval_upper, pred_upper, zero_division=1))  # target_names=df.columns,


class TestEventRisk(unittest.TestCase):

    def test_event_risk(self):
        """This at least assures no changes in behavior go unnoticed, hopefully."""

        self.assertTrue((output_res.avg_metrics.round(3) == known_avg_metrics).all())
        self.assertTrue((output_res.avg_metrics_weighted.round(3) == known_avg_metrics_weighted).all())
        self.assertTrue((output_res.per_series_metrics['b'].round(3) == b_avg_metrics).all())

    def test_event_risk_univariate(self):
        """This at least assures no changes in behavior go unnoticed, hopefully."""
        df = load_daily(long=False)
        df = df.iloc[:, 0:1]
        upper_limit = None
        lower_limit = {
            "model_name": "ARIMA",
            "model_param_dict": {'p': 1, "d": 0, "q": 1},
            "model_transform_dict": {},
            "prediction_interval": 0.9,
        }

        model = EventRiskForecast(
            df,
            forecast_length=forecast_length,
            upper_limit=upper_limit,
            lower_limit=lower_limit,
        )
        # .fit() is optional if model_name, model_param_dict, model_transform_dict are already defined (overwrites)
        model.fit()
        risk_df_upper, risk_df_lower = model.predict()
        historic_upper_risk_df, historic_lower_risk_df = model.predict_historic()
        model.plot(0)

        self.assertTrue((output_res.avg_metrics.round(3) == known_avg_metrics).all())
        self.assertTrue((output_res.avg_metrics_weighted.round(3) == known_avg_metrics_weighted).all())
        self.assertTrue((output_res.per_series_metrics['b'].round(3) == b_avg_metrics).all())
