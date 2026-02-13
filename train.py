import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# STUDENT CODE: Train the model
# We will use the last 30 days to predict the next 1 day.

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_model():
    print("Loading daily data...")
    df = pd.read_csv('daily_electricity_usage.csv', index_col='datetime', parse_dates=True)
    
    # We only have one column of interest
    data = df.values

    print("Scaling data to 0-1 range...")
    # Neural networks work better with small numbers
    # We save the min and max values to use them later for prediction
    data_min = np.min(data)
    data_max = np.max(data)
    
    # Formula for scaling: (value - min) / (max - min)
    scaled_data = (data - data_min) / (data_max - data_min)

    # Save the min and max to a simple text file
    with open('scaling_values.txt', 'w') as f:
        f.write(f"{data_min}\n{data_max}")
    print("Saved scaling values to 'scaling_values.txt'")

    seq_length = 30
    print(f"Creating sequences of {seq_length} days...")
    X, y = create_sequences(scaled_data, seq_length)

    # Split into train and test
    # Let's use 80% for training
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print("Building local LSTM model...")
    model = Sequential()
    # 50 neurons in the first layer
    # input_shape is (30 days, 1 feature)
    model.add(LSTM(50, input_shape=(seq_length, 1)))
    model.add(Dense(1)) # Output is 1 number (tomorrow's usage)

    model.compile(optimizer='adam', loss='mse')

    print("Training model... (this takes a minute)")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    print("Saving model...")
    model.save('my_model.keras')

    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Progress')
    plt.legend()
    plt.savefig('training_loss.png')
    print("Saved plot to 'training_loss.png'")

    # Let's see how well it predicts on the test set
    predictions = model.predict(X_test)
    
    # We need to un-scale the predictions to get real kWh values
    # Formula for un-scaling: value * (max - min) + min
    predictions = predictions * (data_max - data_min) + data_min
    y_test_real = y_test * (data_max - data_min) + data_min

    # Calculate Accuracy (Percentage)
    # First, we calculate the error percentage (Mean Absolute Percentage Error)
    # Formula: |(Actual - Predicted) / Actual| * 100
    mape = np.mean(np.abs((y_test_real - predictions) / y_test_real)) * 100
    
    # Accuracy is 100% - Error Percentage
    accuracy = 100 - mape
    print(f"Model Accuracy: {accuracy:.2f}%")

    # Save the accuracy to a text file so predict.py can read it
    with open('model_accuracy.txt', 'w') as f:
        f.write(f"{accuracy:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_real, label='Actual Usage')
    plt.plot(predictions, label='Predicted Usage')
    plt.title(f'Actual vs Predicted (Accuracy: {accuracy:.2f}%)')
    plt.legend()
    plt.savefig('prediction_results.png')
    print("Saved plot to 'prediction_results.png'")

if __name__ == "__main__":
    train_model()
