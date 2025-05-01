


# In[ ]:


import joblib
import numpy as np
import pandas as pd

# Load preprocessor and models
preprocessor = joblib.load("models/preprocessor.pkl")
model_25 = joblib.load("models/quantile_lgb_model_25.pkl")
model_50 = joblib.load("models/quantile_lgb_model_50.pkl")
model_75 = joblib.load("models/quantile_lgb_model_75.pkl")

def predict(df):
    X_transformed = preprocessor.transform(df)

    # Apply rounding to each prediction in the list
    pred_25 = np.square([round(x, 0) for x in model_25.predict(X_transformed)])
    pred_50 = np.square([round(x, 0) for x in model_50.predict(X_transformed)])
    pred_75 = np.square([round(x, 0) for x in model_75.predict(X_transformed)])

    return {
        "Q25_prediction": pred_25.tolist(),
        "Q50_prediction": pred_50.tolist(),
        "Q75_prediction": pred_75.tolist()
    }


