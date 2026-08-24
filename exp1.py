import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Linear_Regression")

# Sample Dataset
data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5],
    "Scores": [10, 20, 30, 40, 50]
})

# Features and Target
X = data[["Hours"]]
y = data["Scores"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow Run
with mlflow.start_run():

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, predictions)

    # Log Parameters
    mlflow.log_param("model", "LinearRegression")

    # Log Metrics
    mlflow.log_metric("MSE", mse)

    # Log Model
    mlflow.sklearn.log_model(
        model,
        name="linear_regression_model"
    )

    print("Model Trained Successfully!")
    print("Mean Squared Error:", mse)