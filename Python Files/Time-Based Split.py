#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# In[5]:


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DB_NAME = "../nba_data.db"
DB_URI = f"sqlite:///{DB_NAME}"
engine = create_engine(DB_URI, echo=False)


# In[6]:


# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
df = pd.read_sql("SELECT * FROM player_game_features", engine)

# Sort by player_id and game_date to maintain chronological order per player
df = df.sort_values(by=["player_id", "game_date"])

# We'll use the same features as before
features = [
    "rolling_pts_5",
    "rolling_min_5",
    "rolling_fg_pct_5",
    "rolling_ppm_5",
    "rolling_fgm_5",
    "rolling_fga_5",
    "reb",
    "ast"
]

# Drop rows with NaNs in features or target
df = df.dropna(subset=features + ["pts"])

X = df[features]
y = df["pts"]


# In[11]:


# ------------------------------------------------------------
# Time-Based Split
# ------------------------------------------------------------
# For a simple approach: use the first 80% of the data as "past" and the last 20% as "future".
# In reality, you might want to split by actual date boundaries, not just index-based.
split_index = int(len(df) * 0.8)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

print("Time-based validation results:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

