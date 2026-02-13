# Daily Electricity Usage Prediction

This is a deep learning project to predict tomorrow's electricity usage based on the last 30 days of data.
It uses an LSTM (Long Short-Term Memory) neural network.

## Project Structure
- `clean_data.py`: Reads the raw "household_power_consumption.txt" and creates `daily_electricity_usage.csv`.
- `train.py`: Trains the LSTM model on the daily data and saves it.
- `predict.py`: Loads the model and predicts the usage for the next day.
- `requirements.txt`: List of python libraries needed.

## How to Run

### 1. Setup
Make sure you have Python installed.
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Run the cleaning script to prepare the data:
```bash
python clean_data.py
```
This will create `daily_electricity_usage.csv`.

### 3. Train Model
Train the neural network:
```bash
python train.py
```
This will:
- Train the model for 20 epochs.
- Save the model to `my_model.keras`.
- Save the min/max values to `scaling_values.txt`.
- Generate `training_loss.png` and `prediction_results.png` so you can see how well it works.

### 4. Predict
To predict the next day's electricity usage:
```bash
python predict.py
```
It will output the predicted value in kWh.

## How to Run on a New Computer

If you copy this folder to another computer, follow these steps:

1.  **Install Python**: Make sure Python is installed on the new computer.
2.  **Open Terminal**: Go to the folder where you copied these files.
3.  **Install Libraries**: Run this command to get the needed tools:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run Prediction**: Since the model (`my_model.keras`) and scaling values (`scaling_values.txt`) are already there, you don't need to train again. Just run:
    ```bash
    python predict.py
    ```

## Results
Check `prediction_results.png` to see a graph of the Actual vs Predicted values on the test set.
