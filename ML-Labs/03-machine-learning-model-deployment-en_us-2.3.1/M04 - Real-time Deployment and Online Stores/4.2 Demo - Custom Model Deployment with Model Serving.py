# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Custom Model Deployment with Model Serving
# MAGIC
# MAGIC Model Serving provides an easy way of deploying ML models for real-time inference. In some cases, you may need to deploy custom pipelines for your models. An example would be implementing a pre or post processing of the inference result. 
# MAGIC
# MAGIC In this demo, we will demonstrate how you could use **MLflow's `PythonModel`** to implement a post-processing step for your model and serve it with Model Serving.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Deploy a model with custom logic using Model Serving.
# MAGIC
# MAGIC * Create and manage serving endpoints using the API.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## REQUIRED - SELECT A COMPUTE ENVIRONMENT
# MAGIC
# MAGIC <div style="border-left: 4px solid #F44336; background: #FFEBEE; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #C62828; font-size: 1.1em;">Select Compute</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Before starting this notebook, select the required compute environment listed below.</p>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Serverless Compute, Version 5</strong> — How to select an environment version: <a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976D2; text-decoration: underline;">AWS</a> | <a href="https://learn.microsoft.com/azure/databricks/compute/serverless/dependencies#select-an-environment-version" style="color: #1976D2; text-decoration: underline;">Azure</a> | <a href="https://docs.databricks.com/gcp/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976D2; text-decoration: underline;">GCP</a></li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V5</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **Serverless V5**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;"><div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">  <img src="../Includes/Images/genie-code.png" style="height: 64px; margin-bottom: 10px;">  <div style="font-size: 15pt; color: #0B2026; line-height: 1.7; margin-bottom: 16px;">    Want to know more about serving custom models with pre/post-processing logic in Databricks? Ask Genie Code. Click on the genie icon <img src="../Includes/Images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.  </div>  <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0B2026;">    <span id="genie-query-4-2" style="flex: 1;">How do I use MLflow's PythonModel (pyfunc) to wrap a model with custom pre-processing or post-processing logic, log it to Unity Catalog, and deploy it as a real-time endpoint with Model Serving?</span>    <button onclick="      var text = document.getElementById('genie-query-4-2').innerText;      var ta = document.createElement('textarea');      ta.value = text;      ta.style.position = 'fixed';      ta.style.opacity = '0';      document.body.appendChild(ta);      ta.select();      document.execCommand('copy');      document.body.removeChild(ta);      this.innerText = 'Copied!';      var btn = this;      setTimeout(function(){ btn.innerText = 'Copy'; }, 2000);    " style="background: #FF5F46; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-size: 13pt; cursor: pointer; white-space: nowrap;">Copy</button>  </div></div></div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-4.2

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation
# MAGIC
# MAGIC For this demonstration, we will use a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including internet subscription details such as subscription plans, monthly charges and payment methods.
# MAGIC
# MAGIC After loading the dataset, we will perform simple **data cleaning and feature selection**. 
# MAGIC
# MAGIC In the final step, we will split the dataset into **features** and **response** sets.

# COMMAND ----------

from pyspark.sql.functions import col
import mlflow


# Point to UC model registry
client = mlflow.MlflowClient()
mlflow.set_registry_uri("databricks-uc")

# Load dataset with spark
shared_volume_name = 'telco' # From Marketplace
csv_name = 'telco-customer-churn-missing' # CSV file name
dataset_p_telco = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

# Dataset specs
primary_key = "customerID"
response = "Churn"
features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] # Keeping numerical only for simplicity and demo purposes


# Read dataset (and drop nan)
# Convert all fields to double for spark compatibility
telco_df = spark.read.csv(dataset_p_telco, inferSchema=True, header=True, multiLine=True, escape='"')\
            .withColumn("TotalCharges", F.expr("try_cast(trim(TotalCharges) as double)"))\
            .withColumn("SeniorCitizen", col("SeniorCitizen").cast('double'))\
            .withColumn("Tenure", col("tenure").cast('double'))\
            .na.drop(how='any')

# Separate features and ground-truth
features_df = telco_df.select(primary_key, *features)
response_df = telco_df.select(primary_key, response)

# Convert data to pandas dataframes
X_train_pdf = features_df.drop(primary_key).toPandas()
Y_train_pdf = response_df.drop(primary_key).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Fit and Register a Custom Model
# MAGIC
# MAGIC Before we start model deployment process, we will **fit and register a custom model**. 
# MAGIC
# MAGIC Deploying custom pipeline for models typically involves following steps;
# MAGIC
# MAGIC 1. Declare **wrapper classes** for custom models
# MAGIC
# MAGIC 2. Train base model
# MAGIC
# MAGIC 3. Instantiate custom model using trained base model & log to registry

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Wrapper Class
# MAGIC
# MAGIC We will use MLflow's `PythonModel` class to create a custom pipeline. `predict` function of the class implements the custom logic.

# COMMAND ----------

import pandas as pd


# Model wrapper class to output labels and associated probabilities
class CustomProbaModel(mlflow.pyfunc.PythonModel):
    # Initialize model in the constructor
    def __init__(self, model):
        self.model = model
 
    # Prediction function
    def predict(self, context, model_input):
        # Predict the probabilities and class
        prediction_probabilities = self.model.predict_proba(model_input)
        predictions = self.model.predict(model_input)
 
        # Organize multiple outputs
        class_labels = ["No", "Yes"]
        result = pd.DataFrame(prediction_probabilities, columns=[f'prob_{label}' for label in class_labels])
        result['prediction'] = predictions
        
        return result
    
# Dummy model outputting array
class CustomCodeModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        pass
    def predict(self, context, data):
        return [ j for j in range(0, data.shape[0]) ]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Base Model
# MAGIC
# MAGIC In this step, **we will fit the model as normal**.
# MAGIC
# MAGIC Next, and the most important step is **wrapping the model with custom class that we created**.Then, **wrapped model is logged with MLflow**.

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


X_train, X_test, y_train, y_test = train_test_split(X_train_pdf, Y_train_pdf, test_size=0.2, random_state=42)
 
# Initialize and train DecisionTreeClassifier
rf = DecisionTreeClassifier(max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap and Log the Custom Model
# MAGIC Wrap the model and define the input and output schemas. From there, run and log the model using `pyfunc` flavor.

# COMMAND ----------

from mlflow.models import infer_signature

# Wrap the model in the ModelWrapper
wrapped_model = CustomProbaModel(rf)

# Define the input and output schemas
input_example = X_train[:1]
output_example = wrapped_model.predict([],X_train[:1])
signature = infer_signature(X_train[:1], output_example)
 
# Start an MLflow run and log the model
custom_model_name = f"{DA.catalog_name}.{DA.schema_name}.custom_ml_model"
with mlflow.start_run(run_name="Custom Model Example"):
    mlflow.pyfunc.log_model(name="model",
                            python_model=wrapped_model,
                            input_example=input_example,
                            signature=signature,
                            registered_model_name=custom_model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Wrapped Model
# MAGIC
# MAGIC Before serving the model, let's test it and review the result to make sure the post-processing is implemented successfully.

# COMMAND ----------

# Load the model from the run
run_id = mlflow.last_active_run().info.run_id
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
 
# Use the loaded model to predict on the test data
y_test_ = loaded_model.predict(X_test)
display(y_test_)

# COMMAND ----------

# Test custom code model
custom_code_model = CustomCodeModel()
y_cc_test = custom_code_model.predict([], X_train[:1])
print(y_cc_test)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Serve the Custom Model
# MAGIC
# MAGIC Let's serve the model with Model Serving. Here, we will use the API to create the endpoint and serving the model.
# MAGIC
# MAGIC Please note that you could simply use the UI for this task too.

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointTag


# Create/Update endpoint and deploy model+version
w = WorkspaceClient()
endpoint_name = f"ML_AS_03_Demo4_Custom_{DA.unique_name('_')}"
model_version = "1"
endpoint_config_dict = {
    "served_models": [
        {
            "model_name": custom_model_name,
            "model_version": model_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        }
    ]
}
endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

try:
  w.serving_endpoints.create_and_wait(
    name=endpoint_name,
    config=endpoint_config,
    tags=[EndpointTag.from_dict({"key": "db_academy", "value": "serve_custom_model_example"})]
  )
  print(f"Creating endpoint {endpoint_name} with models {custom_model_name} versions {model_version}")

except Exception as e:
  if "already exists" in e.args[0]:
    print(f"Endpoint with name {endpoint_name} already exists")

  else:
    raise(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the Endpoint
# MAGIC
# MAGIC Now that the endpoint is ready, we can query it using the test-sample as shown below. Note that the `predictions` is returned as string (Yes/No) as we implemented in wrapper class.

# COMMAND ----------

# Hard-code test-sample
dataframe_records = [
    {"SeniorCitizen": 0, "tenure":12, "MonthlyCharges":65, "TotalCharges":800},
    {"SeniorCitizen": 1, "tenure":24, "MonthlyCharges":40, "TotalCharges":500}
]

print("Inference results:")
query_response = w.serving_endpoints.query(name=endpoint_name, dataframe_records=dataframe_records)
print(query_response.predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we demonstrated how to build a custom model pipeline using MLflow's `PythonModel` class and serve it with Model Serving. Firstly, we defined the wrapper class with custom post-processing logic. Next, we fitted the model as usual and wrapped it with the custom model. Finally, we deployed the model with Model Serving and queried the serving endpoint using the API.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>