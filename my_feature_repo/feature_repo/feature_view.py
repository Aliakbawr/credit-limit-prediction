from feast import FeatureView, Field
from feast.types import Int64, Float32
from datetime import timedelta
from data_source import credit_data_source
from entity import customer  # ✅ Make sure this is imported

credit_features = FeatureView(
    name="credit_features",
    entities=[customer],  # ✅ Use Entity object, not string
    ttl=timedelta(days=365),
    schema=[
        Field(name="Customer_Age", dtype=Int64),
        Field(name="Credit_Limit", dtype=Float32),
        Field(name="Total_Trans_Amt", dtype=Float32),
        Field(name="Total_Trans_Ct", dtype=Int64),
        Field(name="Avg_Utilization_Ratio", dtype=Float32),
        # Add more fields as needed
    ],
    online=True,
    source=credit_data_source,
)
