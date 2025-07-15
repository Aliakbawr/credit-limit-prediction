# preprocess_credit_data.py
import pandas as pd
from datetime import datetime

df = pd.read_csv("my_feature_repo/feature_repo/data/CreditPrediction.csv")
df["event_timestamp"] = pd.to_datetime(datetime.now())  # Use fixed timestamp
df.to_csv("my_feature_repo/feature_repo/data/creditprediction_with_ts.csv", index=False)
