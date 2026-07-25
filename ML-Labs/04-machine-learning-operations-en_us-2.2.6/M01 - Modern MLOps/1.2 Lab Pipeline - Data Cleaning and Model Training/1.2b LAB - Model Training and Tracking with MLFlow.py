# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB - Model Training and Tracking with MLflow
# MAGIC In this lab, you will train a machine learning model on cleaned data and track it using MLflow.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _This lab contains the following tasks:_
# MAGIC 1. Load and split the cleaned dataset.
# MAGIC 2. Train a RandomForestClassifier model.
# MAGIC 3. Evaluate the model's performance.
# MAGIC 4. Save the best model for deployment.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): `17.3.x-cpu-ml-scala2.13`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:
# MAGIC

# COMMAND ----------

# MAGIC %pip install --upgrade urllib3

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup-1.1a

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions**
# MAGIC
# MAGIC Throughout this lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Data Preparation
# MAGIC This task involves loading the cleaned dataset, handling missing values, and splitting the data for training and testing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.1 Load the Cleaned Data
# MAGIC Load the cleaned data from the Delta table.

# COMMAND ----------

import pandas as pd
# Read dataset from the feature store table
table_name = f"{DA.catalog_name}.{DA.schema_name}.telco_cleaned_table"
feature_data_pd = spark.table(table_name).toPandas()

# Drop the 'customerID' column
feature_data_pd = feature_data_pd.drop(columns=['customerID'])

# Add unique_id column
feature_data_pd['unique_id'] = range(len(feature_data_pd))

# Convert all columns in the DataFrame to the 'double' data type
feature_data_pd = feature_data_pd.astype("double")

# Display the DataFrame and print the columns
display(feature_data_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.2 Handle Missing Values
# MAGIC Use an imputer to fill in the missing values.

# COMMAND ----------

from sklearn.impute import SimpleImputer

# Handle missing values
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(feature_data_pd)
feature_data_pd = pd.DataFrame(imputed_data, columns=feature_data_pd.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.3 Split the Data
# MAGIC Split the data into training and testing sets.

# COMMAND ----------

from sklearn.model_selection import train_test_split

print(f"We have {feature_data_pd.shape[0]} records in our source dataset")

# split target variable into its own dataset
target_col = "Churn"
X_all = feature_data_pd.drop(labels=target_col, axis=1)
y_all = feature_data_pd[target_col]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.8, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Fit and Log the Model
# MAGIC Train a RandomForestClassifier model, evaluate its performance, and log the model along with its parameters and metrics using MLflow.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure MLflow client to access models in Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# Fetch Model information
client = mlflow.tracking.MlflowClient()
model_name = f"{DA.catalog_name}.{DA.schema_name}.churn-prediction" 

# Helper function to get the latest model version
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions(f"name = '{model_name}'")
    return max([model_version_info.version for model_version_info in model_version_infos])

# Set the path for MLflow experiment
mlflow.set_experiment(f"/Users/{DA.username}/Lab-1.2b-Model-traning-with-MLflow")

# Turn off autologging
mlflow.sklearn.autolog(disable=True)

# Define model parameters
rf_params = {
    "n_estimators": 100,
    "random_state": 42
}

# Start an MLflow run
with mlflow.start_run(run_name="Model Training Lab") as run:
    # Log the dataset as artifacts
    feature_data_pd.to_csv("/tmp/feature_data.csv", index=False)
    X_train.to_csv("/tmp/X_train.csv", index=False)
    X_test.to_csv("/tmp/X_test.csv", index=False)
    
    mlflow.log_artifact("/tmp/feature_data.csv", artifact_path="feature_data")
    mlflow.log_artifact("/tmp/X_train.csv", artifact_path="training_data")
    mlflow.log_artifact("/tmp/X_test.csv", artifact_path="test_data")

    # Log our parameters
    mlflow.log_params(rf_params)

    # Fit the model
    rf = RandomForestClassifier(**rf_params)
    rf_mdl = rf.fit(X_train, y_train)

    # Define model signature
    signature = infer_signature(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=rf_mdl,
        artifact_path="model-artifacts",
        signature=signature,
        registered_model_name=model_name  # Provide a valid name for the registered model
    )

    # Evaluate on the training set
    y_train_pred = rf_mdl.predict(X_train)
    mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
    mlflow.log_metric("train_precision", precision_score(y_train, y_train_pred))
    mlflow.log_metric("train_recall", recall_score(y_train, y_train_pred))
    mlflow.log_metric("train_f1", f1_score(y_train, y_train_pred))

    # Evaluate on the test set
    y_test_pred = rf_mdl.predict(X_test)
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
    mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred))
    mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred))
    mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))

    # Set model alias
    latest_model_version = get_latest_model_version(model_name)
    client.set_registered_model_alias(model_name, "Baseline", latest_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Visualize Model Performance
# MAGIC Generate and log confusion matrix and feature importance plots.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.1 Confusion Matrix
# MAGIC Generate and log a confusion matrix to evaluate the model's performance.

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Computing the confusion matrix
cm = confusion_matrix(y_test, y_test_pred, labels=[1, 0])

# Creating a figure object and axes for the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))

# Plotting the confusion matrix using the created axes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Setting the title of the plot
ax.set_title('Confusion Matrix')

# Now 'fig' can be used with MLFlow's log_figure function
client.log_figure(run.info.run_id, figure=fig, artifact_file="confusion_matrix.png")

# Showing the plot here for demonstration
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.2 Feature Importance Plots
# MAGIC Generate and log a feature importance plot to understand the significance of each feature.

# COMMAND ----------

import numpy as np

# Retrieving feature importances
feature_importances = rf_mdl.feature_importances_
feature_names = X_train.columns.to_list()

# Plotting the feature importances
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(feature_names))
ax.bar(y_pos, feature_importances, align='center', alpha=0.7)
ax.set_xticks(y_pos)
ax.set_xticklabels(feature_names, rotation=45)
ax.set_ylabel('Importance')
ax.set_title('Feature Importances in  RandomForest Classifier')

# log to mlflow
client.log_figure(run.info.run_id, figure=fig, artifact_file="feature_importances.png")

# display here
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Search for the Best Run

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4.1 Train & Log Multiple Trials with MLflow
# MAGIC This cell trains several RandomForest configs and logs metrics (including `test_f1`) to MLflow.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure we keep using the same experiment as Task 2
experiment_path = f"/Users/{DA.username}/Lab-1.2b-Model-traning-with-MLflow"
mlflow.set_experiment(experiment_path)

# We'll handle logging ourselves
mlflow.sklearn.autolog(disable=True)

# A tiny hyperparameter sweep
param_grid = [
    {"n_estimators": 100, "max_depth": None, "random_state": 42},
    {"n_estimators": 200, "max_depth": None, "random_state": 42},
    {"n_estimators": 200, "max_depth": 10,   "random_state": 42},
    {"n_estimators": 400, "max_depth": None, "random_state": 42},
]

for i, params in enumerate(param_grid, start=1):
    with mlflow.start_run(run_name=f"rf_trial_{i}") as run:
        # Fit
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)

        # Log params & metrics
        mlflow.log_params(params)
        mlflow.log_metric("train_accuracy",  accuracy_score(y_train, y_train_pred))
        mlflow.log_metric("train_precision", precision_score(y_train, y_train_pred))
        mlflow.log_metric("train_recall",    recall_score(y_train, y_train_pred))
        mlflow.log_metric("train_f1",        f1_score(y_train, y_train_pred))
        mlflow.log_metric("test_accuracy",   accuracy_score(y_test,  y_test_pred))
        mlflow.log_metric("test_precision",  precision_score(y_test,  y_test_pred))
        mlflow.log_metric("test_recall",     recall_score(y_test,     y_test_pred))
        mlflow.log_metric("test_f1",         f1_score(y_test,         y_test_pred))

        # Log model (keep the same artifact path name used earlier so Task 5 works)
        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model-artifacts",
            signature=signature
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4.2 Find the Best Run Directly in MLflow
# MAGIC Query the experiment for completed runs and pick the highest `test_f1`.

# COMMAND ----------

import mlflow
from mlflow.entities import ViewType

# Resolve experiment ID
exp_id = mlflow.get_experiment_by_name(experiment_path).experiment_id

# Find best by test_f1
runs_df = mlflow.search_runs(
    experiment_ids=[exp_id],
    filter_string="attributes.status = 'FINISHED'",
    run_view_type=ViewType.ACTIVE_ONLY,
    order_by=["metrics.test_f1 DESC"]
)

display(runs_df)  # Databricks display for convenience

# Extract best
if runs_df.empty:
    raise RuntimeError("No completed runs found in the experiment.")

best_run = runs_df.iloc[0]
best_run_id = best_run.run_id
best_test_f1 = best_run["metrics.test_f1"]

print("Best run_id:", best_run_id)
print("Best test_f1:", best_test_f1)

# IMPORTANT: matches the artifact_path used above ("model-artifacts")
model_uri = f"runs:/{best_run_id}/model-artifacts"
print("Best model URI:", model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Save the Best Model
# MAGIC Save the model if it meets the accuracy threshold.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 5.1 Register the Best Model
# MAGIC Register the best-performing model from the MLflow experiment (identified in Task 4).

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# Use the best run selected in Task 4
print(f"Best run selected from MLflow experiment: {best_run_id}")

# Build the model URI (matches the artifact path used in Task 4)
model_uri = f"runs:/{best_run_id}/model-artifacts"

# Define a name for the registered model
model_name = "best_mlflow_model"

print("Model URI:", model_uri)
print("Model Name:", model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 5.2 Save and Log the Best Model
# MAGIC Save and log the best model to the MLflow model registry.

# COMMAND ----------

import time
import mlflow
from mlflow.tracking import MlflowClient

# Use Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# UC-qualified model name (catalog.schema.model)
best_model_name = f"{DA.catalog_name}.{DA.schema_name}.best_mlflow_model"

# Register the model found in Task 5.1
# (model_uri was set in Task 5.1 as: runs:/<best_run_id>/model-artifacts)
model_details = mlflow.register_model(model_uri=model_uri, name=best_model_name)
model_version = model_details.version
print(f"Registering model: {best_model_name} (version: {model_version})")

# Wait until model registration finishes (READY) before setting alias
status = None
while status not in {"READY", "FAILED"}:
    time.sleep(2)
    mv = client.get_model_version(name=best_model_name, version=model_version)
    status = mv.status
    print(f"Model version status: {status}")

if status == "FAILED":
    raise RuntimeError(f"Model registration failed for {best_model_name} v{model_version}")

# Set / update alias to point to this version
client.set_registered_model_alias(best_model_name, "Baseline", model_version)
print(f"Alias 'Baseline' -> {best_model_name} v{model_version}")
print("Model registered and aliased successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this lab, you learned how to train a machine learning model on cleaned data and track it using MLflow. You also evaluated the model's performance and saved the best model for deployment.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>