# aqi_model_training.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
data = pd.read_csv('/content/air_quality_data.csv')  # Make sure this path is correct
print("Initial data preview:\n", data.head())

# -----------------------------
# Step 2: Data Preprocessing
# -----------------------------
# Drop missing values
data.dropna(inplace=True)

# Standardize column names
data.columns = [col.strip().lower() for col in data.columns]

# Rename columns if needed
# Ensure these column names match your dataset
expected_columns = ['co aqi value', 'ozone aqi value', 'no2 aqi value', 'pm2.5 aqi value', 'aqi value']
if not all(col in data.columns for col in expected_columns):
    raise ValueError(f"Dataset must contain the following columns: {expected_columns}")

# -----------------------------
# Step 3: Exploratory Data Analysis (EDA)
# -----------------------------
# Pairplot
sns.pairplot(data[expected_columns])
plt.suptitle("Pairplot of AQI and Pollutants", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data[expected_columns].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Step 4: Feature Selection
# -----------------------------
X = data[['co aqi value', 'ozone aqi value', 'no2 aqi value', 'pm2.5 aqi value']]
y = data['aqi value']

# -----------------------------
# Step 5: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 6: Train the Model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 7: Predict and Evaluate
# -----------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("📊 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# -----------------------------
# Step 8: Visualization
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label='Actual AQI', color='blue', marker='o')
plt.plot(y_pred[:100], label='Predicted AQI', color='orange', linestyle='--', marker='x')
plt.title("Actual vs Predicted AQI (Sample of 100)")
plt.xlabel("Sample Index")
plt.ylabel("AQI Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
