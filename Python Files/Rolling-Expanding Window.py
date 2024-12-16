#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# In[2]:


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DB_NAME = "nba_data.db"
DB_URI = f"sqlite:///{DB_NAME}"
engine = create_engine(DB_URI, echo=False)


# In[3]:


# ------------------------------------------------------------
# Load Data & Sort
# ------------------------------------------------------------
df = pd.read_sql("SELECT * FROM player_game_features", engine)

# Ensure data is sorted by player and date
df = df.sort_values(by=["player_id", "game_date"])

# Extract the season or year from 'game_date'. 
# Assuming 'game_date' is in a format like "YYYY-MM-DD".
df['game_year'] = pd.to_datetime(df['game_date']).dt.year

# Features and target as before
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

df = df.dropna(subset=features + ["pts"])

X = df[features]
y = df["pts"]


# In[4]:


# ------------------------------------------------------------
# Rolling/Expanding Window Validation
# ------------------------------------------------------------
# Let's define which years to use for training and validation:
# For example:
#  - Train on 2016-2019, validate on 2020
#  - Train on 2017-2020, validate on 2021

# First, find the range of years available
available_years = sorted(df['game_year'].unique())
print("Available Years in Data:", available_years)

# We will define a 4-year rolling window for training.
training_window = 4

mae_scores = []
rmse_scores = []
years_tested = []

# Identify the range of years we can validate on. 
# If we start at 2020, that means we must have at least 2016-2019 available for training.
# So for each validation year, we look back training_window years.
for validate_year in available_years:
    start_train_year = validate_year - training_window
    # Check if we have enough years in the past to train on
    if start_train_year < available_years[0]:
        # Not enough past years to train on
        continue
    # Ensure that all those past years exist
    if not all(y in available_years for y in range(start_train_year, validate_year)):
        # If we don't have a continuous range of data
        continue

    # Define training and validation sets by year
    train_mask = (df['game_year'] >= start_train_year) & (df['game_year'] < validate_year)
    val_mask = (df['game_year'] == validate_year)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # If val set is empty or train set is empty, skip
    if len(X_train) == 0 or len(X_val) == 0:
        continue

    # Train a simple model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the validation year
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    mae_scores.append(mae)
    rmse_scores.append(rmse)
    years_tested.append(validate_year)

    print(f"Validation Year: {validate_year}")
    print(f"Train Years: {start_train_year} to {validate_year-1}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}\n")


# In[5]:


# ------------------------------------------------------------
# Aggregate Results
# ------------------------------------------------------------
if len(years_tested) > 0:
    print("Summary of Rolling Window Validation:")
    for yr, mae_score, rmse_score in zip(years_tested, mae_scores, rmse_scores):
        print(f"Year: {yr}, MAE: {mae_score:.2f}, RMSE: {rmse_score:.2f}")

    print("\nAverage MAE:", np.mean(mae_scores))
    print("Average RMSE:", np.mean(rmse_scores))
else:
    print("No rolling window validation was performed (check available data and logic).")


# In[ ]:




