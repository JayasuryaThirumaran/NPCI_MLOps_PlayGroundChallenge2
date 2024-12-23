# train.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
def load_data(file_path):
    # Read the dataset
    df = pd.read_csv(file_path, parse_dates=['dteday'])
    return df

# Preprocess the data
def preprocess_data(df):
    # Convert categorical columns to numeric using LabelEncoder
    label_encoders = {}
    categorical_columns = ['season', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Feature engineering: Drop 'dteday' and 'casual' since we want to predict 'cnt'
    df.drop(['dteday', 'casual'], axis=1, inplace=True)
    
    # Handle missing values (if any) by filling with the median
    df.fillna(df.median(), inplace=True)

    return df, label_encoders

# Train the Random Forest model
def train_model(df):
    # Define features (X) and target (y)
    X = df.drop(['cnt', 'registered'], axis=1)
    y = df['cnt']  # You can also use 'registered' if needed instead of 'cnt'

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model

# Save the model to a file
def save_model(model, label_encoders, model_path="random_forest_model.pkl", le_path="label_encoders.pkl"):
    # Save the trained model
    joblib.dump(model, model_path)
    # Save the label encoders
    joblib.dump(label_encoders, le_path)
    print(f"Model and Label Encoders saved to {model_path} and {le_path}")

# Main function to load, preprocess, train, and save the model
def main():
    # File path to the CSV dataset
    file_path = 'your_dataset.csv'  # Update this with the correct path to your dataset

    # Step 1: Load the data
    df = load_data(file_path)

    # Step 2: Preprocess the data
    df, label_encoders = preprocess_data(df)

    # Step 3: Train the model
    model = train_model(df)

    # Step 4: Save the trained model and label encoders
    save_model(model, label_encoders)

if __name__ == "__main__":
    main()
