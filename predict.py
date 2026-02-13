import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# STUDENT CODE: Predict next 3 days usage using the saved model

def predict_future():
    print("Loading model and data...")
    # Load the model we trained
    model = load_model('my_model.keras')
    
    # Load the scaling values (min and max)
    with open('scaling_values.txt', 'r') as f:
        lines = f.readlines()
        data_min = float(lines[0].strip())
        data_max = float(lines[1].strip())
        
    # Load the data again to get the last 30 days
    df = pd.read_csv('daily_electricity_usage.csv', index_col='datetime', parse_dates=True)
    data = df.values

    # We need the last 30 days to predict the next day
    # We make a copy so we can add our predictions to it without messing up the original data
    current_input_sequence = data[-30:].copy()
    
    print(f"\n--- Result ---")
    print(f"Based on the last 30 days of data:")

    # Loop to predict 3 days
    days_to_predict = 3
    for i in range(days_to_predict):
        # Scale the data using the formula: (value - min) / (max - min)
        # We need to reshape it to (1, 30, 1) to match (Batch, Time Steps, Features)
        current_input_scaled = (current_input_sequence - data_min) / (data_max - data_min)
        X_input = current_input_scaled.reshape(1, 30, 1)

        # Predict the next single day
        # verbose=0 hides the progress bar from Keras
        predicted_scaled = model.predict(X_input, verbose=0)

        # Un-scale the prediction to get the real kWh value
        predicted_value = predicted_scaled[0][0] * (data_max - data_min) + data_min

        print(f"Day {i+1}: {predicted_value:.2f} kWh")

        # Now we need to update our input sequence for the NEXT loop.
        # We perform a "rolling" update:
        # 1. Get the new predicted value
        # 2. Add it to the end of our sequence
        # 3. Remove the oldest value so we still have 30 days
        
        # Create a new numpy array with the predicted value
        new_row = np.array([[predicted_value]])
        
        # current_input_sequence[1:] takes everything EXCEPT the first one
        # then we append new_row
        current_input_sequence = np.append(current_input_sequence[1:], new_row, axis=0)

    print(f"--------------")
    
    # Check if we have an accuracy file
    try:
        with open('model_accuracy.txt', 'r') as f:
            accuracy = f.read().strip()
            print(f"Model Accuracy: {accuracy}%")
    except:
        print("Model accuracy info not found.")

if __name__ == "__main__":
    predict_future()
