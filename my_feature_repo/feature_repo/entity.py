from feast import Entity
from feast import Entity, ValueType  # ✅ Use ValueType from feast

customer = Entity(
    name="client_id",
    join_keys=["CLIENTNUM"],
    value_type=ValueType.INT64 
)
