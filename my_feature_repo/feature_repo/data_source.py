# data_source.py
from feast import FileSource
from datetime import timedelta

credit_data_source = FileSource(
    path="data/creditprediction_with_ts.csv",
    timestamp_field="event_timestamp",
)
