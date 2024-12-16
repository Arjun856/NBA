#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[5]:


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DB_NAME = "../nba_data.db"
DB_URI = f"sqlite:///{DB_NAME}"
engine = create_engine(DB_URI, echo=False)


# In[6]:


# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------
query = "SELECT * FROM player_game_features;"
df = pd.read_sql(query, engine)

# For simplicity, let's assume we want to predict 'pts' using some of the rolling averages and efficiency metrics we created.
# Features (X) could be:
# 'rolling_pts_5', 'rolling_min_5', 'rolling_fg_pct_5', 'rolling_ppm_5', 'rolling_fgm_5', 'rolling_fga_5', 'reb', 'ast'
# Target (y) = 'pts'
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

# Drop rows where these features might be NaN (first few games of each player might not have full rolling windows)
df = df.dropna(subset=features + ["pts"])

X = df[features]
y = df["pts"]


# In[7]:


# ------------------------------------------------------------
# 2. Split Data into Train and Test
# ------------------------------------------------------------
# We'll do a simple random split. More sophisticated approaches might respect time (train on past, test on future),
# but for a first pass, this is fine.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# ------------------------------------------------------------
# 3. Train a Simple Model
# ------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)


# In[9]:


# ------------------------------------------------------------
# 4. Evaluate the Model
# ------------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("Model Evaluation:")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")


# In[10]:


# ------------------------------------------------------------
# 5. Interpretation and Next Steps
# ------------------------------------------------------------
# At this point, you've got a baseline model. The results (MAE, MSE, RMSE) tell you how far off the predictions are.
# You can try:
# - Adding more features (opponent strength, rest days)
# - Trying a more advanced model (RandomForest, Gradient Boosting, Neural Network)
# - Using time-series validation instead of a simple random split for more realistic evaluation.

