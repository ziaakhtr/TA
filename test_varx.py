import pandas as pd
from dashboard_x import detect_and_predict, PRODUCT_MODELS, FEATURE_EXOG, load_csv, DATA_PATH

df_raw = load_csv(DATA_PATH)
df = df_raw[df_raw["product"] == "MU-302-50Kg"].copy().sort_values("date").reset_index(drop=True)
df_test = df[df["split"] == "test"].copy()

available_features = [c for c in FEATURE_EXOG if c in df.columns]
X_test = df_test[available_features].fillna(0)

try:
    print("Testing VARX test set prediction...")
    res = detect_and_predict(X_test, PRODUCT_MODELS["MU-302-50Kg"]["model"], None)
    print("VARX test pred success:", len(res[0]))
except Exception as e:
    import traceback
    traceback.print_exc()

# Also test ARIMAX
df2 = df_raw[df_raw["product"] == "MU-380-40KG"].copy().sort_values("date").reset_index(drop=True)
df_test2 = df2[df2["split"] == "test"].copy()
X_test2 = df_test2[available_features].fillna(0)
try:
    print("\nTesting ARIMAX test set prediction...")
    res2 = detect_and_predict(X_test2, PRODUCT_MODELS["MU-380-40KG"]["model"], None)
    print("ARIMAX test pred success:", len(res2[0]))
except Exception as e:
    import traceback
    traceback.print_exc()
