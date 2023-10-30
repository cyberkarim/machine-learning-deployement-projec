from factorized import *

import pandas as pd
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to your data folder
data_folder_path = os.path.join(script_dir, "../data")

# Construct the path to your CSV file within the data folder
csv_file_path = os.path.join(data_folder_path, "ref_data.csv")

# Import the database
credit_card_df = load("../data/ref_data.csv")
# Drop the time axis and NaN cases
credit_card_df = credit_card_df.drop('Time', axis = 1)

credit_card_df = credit_card_df.dropna()

# Normalize the data
credit_card_df = normalize(credit_card_df, 0.2)

# Train data with XGBOOST
X_test, y_test, model = model_train(credit_card_df, xgb.XGBClassifier())
