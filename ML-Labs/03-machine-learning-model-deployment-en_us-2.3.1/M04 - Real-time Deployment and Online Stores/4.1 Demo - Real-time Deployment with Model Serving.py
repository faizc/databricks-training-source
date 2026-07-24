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
# MAGIC # Real-time Deployment with Model Serving (Offline Features)
# MAGIC
# MAGIC This demo focuses on **real-time model deployment** using Model Serving with **offline feature tables** stored in Unity Catalog (Delta tables with a primary key). You’ll train on offline features, deploy two model versions to one endpoint, and validate **A/B testing** traffic splits.
# MAGIC
# MAGIC **Learning Objectives**
# MAGIC
# MAGIC By the end you will be able to:
# MAGIC - Use a **UC Delta feature table (offline)** for model training and evaluation.
# MAGIC - Register models to Unity Catalog and manage **Champion/Challenger** aliases.
# MAGIC - Create a **Model Serving** endpoint with **two versions** and configure a **traffic split** for A/B testing.
# MAGIC - **Query** the endpoint (batching requests) and **visualize** prediction distribution by served model.
# MAGIC - *(Optional)* Enable **inference table auto-capture** for request/response logging and observability.
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
# MAGIC * This demo uses **offline feature tables** (Delta tables in Unity Catalog) only. No online serving infrastructure is required. For real-time feature lookups, Databricks now recommends **Online Feature Stores** (backed by Lakebase Autoscaling), which replace the legacy standalone "online tables".

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;"><div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">  <img src="../Includes/Images/genie-code.png" style="height: 64px; margin-bottom: 10px;">  <div style="font-size: 15pt; color: #0B2026; line-height: 1.7; margin-bottom: 16px;">    Want to know more about real-time model deployment and A/B testing with Model Serving? Ask Genie Code. Click on the genie icon <img src="../Includes/Images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.  </div>  <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0B2026;">    <span id="genie-query-4-1" style="flex: 1;">How do I deploy a model to a Model Serving endpoint for real-time inference? How do I serve two model versions behind one endpoint and configure a traffic split for A/B testing using Champion and Challenger aliases in Unity Catalog?</span>    <button onclick="      var text = document.getElementById('genie-query-4-1').innerText;      var ta = document.createElement('textarea');      ta.value = text;      ta.style.position = 'fixed';      ta.style.opacity = '0';      document.body.appendChild(ta);      ta.select();      document.execCommand('copy');      document.body.removeChild(ta);      this.innerText = 'Copied!';      var btn = this;      setTimeout(function(){ btn.innerText = 'Copy'; }, 2000);    " style="background: #FF5F46; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-size: 13pt; cursor: pointer; white-space: nowrap;">Copy</button>  </div></div></div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-4.1

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
# MAGIC ## Offline Feature Tables for Real-Time Inferencing
# MAGIC
# MAGIC Before deploying a model for real-time inference, it’s important to understand the role of **feature tables** in Model Serving.
# MAGIC
# MAGIC A **feature table** is simply a **materialized Delta table in Unity Catalog** with a defined **primary key**. These tables store curated feature values used during both model training and inference.
# MAGIC
# MAGIC In this demo, we’ll focus entirely on **offline feature tables**, which:
# MAGIC - Reside in Unity Catalog as Delta tables.  
# MAGIC - Are typically refreshed on a schedule (for example, daily or hourly batch updates).  
# MAGIC - Can be used directly by Model Serving endpoints without any additional configuration.  
# MAGIC
# MAGIC > In contrast to real-time serving via **Online Feature Stores** (backed by Lakebase Autoscaling, which replace the legacy standalone “online tables”), offline tables are suitable for most production workloads where features are precomputed and served from Delta Lake.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Real-time Deployment With Offline Feature Tables
# MAGIC Here we consider a scenario where you have already gone through the development process (data preparation, and model development) and you're ready to deploy a model with offline features. We will first look at deploying two models that were created as a part of the classroom setup - a champion model and a challenger model with aliases `champion` and `challenger`, respectively. 
# MAGIC
# MAGIC We will serve our two models using a 50/50 traffic split for A/B Testing. First, Let's read in our data and explore its lineage. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Inspect Offline The Feature Table and Model Versions
# MAGIC
# MAGIC For this demonstration, we will use a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including internet subscription details such as subscription plans, monthly charges and payment methods. 
# MAGIC
# MAGIC As a part of the classroom setup for this course, a feature table was created called **features** that **did _not_ include feature lookups.** This is the table we are reading in during the next step.
# MAGIC
# MAGIC #### Lineage Inspection
# MAGIC - Navigate to the catalog and schema used with this Vocareum environment (see the output from the previous cell).
# MAGIC - Find the table called `features` and model called `ml_model`. 
# MAGIC   - Click on Lineage. 
# MAGIC   - Click on **See lineage graph** and inspect it. This will show the footprint of how the catalog assets were made.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Read in Features and Response Variable from Feature Store
# MAGIC
# MAGIC Here we will read in our dataset and split between features and response variables. We will show how this can be performed with the Databricks SDK using the Feature Engineering Client.
# MAGIC
# MAGIC > #### What's the difference between `fe.read_table()` and `read.spark.table()`?
# MAGIC Essentially, we use `fe.read_table()` whenever we are specifically working with feature tables stored within Feature Store and `spark.read.table()` for general-purpose reading. Note that `fe.read_table()` is part of the Databricks Feature Engineering API and integrates well with other Feature Store APIs like logging models (see ****Part 2: Real-Time Deployment with Online Feature Tables****). On the other hand, `spark.read.table()` is a broader Spark SQL method for reading data from any table within the Spark session.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# Define primary key 
primary_key = "customerID"

# Read in feature table
feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"
X_train_df = fe.read_table(name=feature_table_name)
X_train_pdf = X_train_df.drop(primary_key).toPandas()

# Read in response table 
response_table_name = f"{DA.catalog_name}.{DA.schema_name}.response"
Y_train_df = spark.read.table(response_table_name)
Y_train_pdf = Y_train_df.drop(primary_key).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Real-time A/B Testing with Model Serving
# MAGIC
# MAGIC Let's serve the two models we logged in the previous step using Model Serving. Model Serving supports endpoint management via the UI and the API. 
# MAGIC
# MAGIC Below you will find instructions for using the UI and it is simpler method compared to the API. **In this demo, we will use the API to configure and create the endpoint**.
# MAGIC
# MAGIC **Both the UI and the API support querying created endpoints in real-time**. We will use the API to query the endpoint using a test-set.
# MAGIC
# MAGIC
# MAGIC > #### What is A/B Testing? 
# MAGIC > A/B testing is a method to compare two versions of a model or system by splitting user traffic and measuring performance metrics to determine which version delivers better results.

# COMMAND ----------

endpoint_name = f"ML_AS_03_Demo4_{DA.unique_name('_')}"
print(f"Endpoint name: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1: Serve model(s) using UI
# MAGIC
# MAGIC After registering the (new version(s) of the) model to the model registry. To provision a serving endpoint via UI, follow the steps below.
# MAGIC
# MAGIC 1. In the left sidebar, click **Serving**.
# MAGIC
# MAGIC 2. To create a new serving endpoint, click **Create serving endpoint**.   
# MAGIC
# MAGIC     a. In the **Name** field, enter the name printed above.  
# MAGIC
# MAGIC     b. Click in the Entity field. A dialog appears. Go to **My models**, and then select the **'model_4_1_demo'** from the drop-down menus. 
# MAGIC
# MAGIC     c. Click **Confirm**.
# MAGIC
# MAGIC     d. In the **Version** drop-down menu, select the **version 1**.    
# MAGIC
# MAGIC     e. *[OPTIONAL]* to deploy another model (e.g. for A/B testing):
# MAGIC     - Click on **+Add served entity**.
# MAGIC     - Enter the above mentioned details as above, but use **version 2**.
# MAGIC     - Set the traffic split to 50% for each model.
# MAGIC
# MAGIC     f. Click **Create**. The endpoint page opens and the endpoint creation process starts.   
# MAGIC
# MAGIC See the Databricks documentation for details ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#ui-workflow)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints#--ui-workflow)).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2: Serve Model(s) Using the Databricks Python SDK
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get Models to Serve
# MAGIC
# MAGIC In order to serve the model, we will initialize the MLflow client with `MLflowClient` and the workspace client with `WorkspaceClient`. We will configure the MLflow client to point to Unity Catalog instead of the Workspace with `set_registry_uri("databricks-uc")`. The workspace client will be used to create the model serving endpoint.

# COMMAND ----------

from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointTag

# Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
# Initialize MLflow client
client = mlflow.MlflowClient()
# Initialize workspace client
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC Define variables that will be used for configuring the endpoint like `model_name`. The output from running the next cell will show version 1 of our model registered as the champion model and version 2 as being the challenger.

# COMMAND ----------

# Define model name
model_name = f"dbacademy.{DA.schema_name}.model_4_1_demo"
# Parse model name from UC namespace
served_model_name =  model_name.split('.')[-1]
# Define the endpoint name
endpoint_name = f"ML_AS_03_Demo4_{DA.unique_name('_')}"

# Get version of our model registered to UC as a part of the classroom setup
model_version_champion = client.get_model_version_by_alias(name=model_name, alias="Champion").version # Get champion version
model_version_challenger = client.get_model_version_by_alias(name=model_name, alias="Challenger").version # Get challenger version


print(f"Model version Champion: {model_version_champion}")
print(f"Model version Challenger: {model_version_challenger}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configure
# MAGIC
# MAGIC Define our model serving endpoint with `endpoint_config`. The configuration below shows two versions of the same being deployed (`model_version_champion` and `model_version_challenger`) along with how to configure traffic during inferencing.

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput

endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_name,
            "model_version": model_version_champion,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        },
        {
            "model_name": model_name,
            "model_version": model_version_challenger,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        },
    ],
    "traffic_config": {
        "routes": [
            {"served_model_name": f"{served_model_name}-{model_version_champion}", "traffic_percentage": 50},
            {"served_model_name": f"{served_model_name}-{model_version_challenger}", "traffic_percentage": 50},
        ]
    }
}


endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Serve the endpoint
# MAGIC Use the configuration just created to serve the model.
# MAGIC > The time to create a model serving endpoint < 1 minute

# COMMAND ----------

try:
  w.serving_endpoints.create(
    name=endpoint_name,
    config=endpoint_config,
    tags=[EndpointTag.from_dict({"key": "db_academy", "value": "serve_fs_model_example"})]
  )
  print(f"Creating endpoint {endpoint_name} with models {model_name} versions {model_version_champion} & {model_version_challenger}")

except Exception as e:
  if "already exists" in e.args[0]:
    print(f"Endpoint with name {endpoint_name} already exists")

  else:
    raise(e)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify Endpoint Creation
# MAGIC
# MAGIC Let's verify that the endpoint is created and ready to be used for inference using the `assert` command, which is used to check whether a given condition is true.

# COMMAND ----------

endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name)

assert endpoint.state.config_update.value == "NOT_UPDATING" and endpoint.state.ready.value == "READY" , "Endpoint not ready or failed"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Query the Endpoint and Visualize
# MAGIC
# MAGIC Here we will use the training dataset to query our endpoint.
# MAGIC
# MAGIC 1. Define the dataset to sample from.
# MAGIC 1. Query by batch to highlight model-split traffic.

# COMMAND ----------

dataframe_records = X_train_pdf.iloc[:1000].to_dict(orient='records') #1k sample records

# COMMAND ----------

# MAGIC %md
# MAGIC Here we will query in batches so we can see the traffic split per 100 rows (there are around 2000 rows in this dataset)
# MAGIC
# MAGIC To help visualize the A/B testing output, create a visual using the UI (you only need to do this once;  rerunning the cell will update the visualization). 
# MAGIC 1. After running the next cell, select the + sign on the second table and select **Visualization**. 
# MAGIC 1. The default visual should represent the Yes/No split per model.
# MAGIC
# MAGIC > Since the dataset we're working with is not very large, you might have to run the cell a few times to get a fairly close 50/50 split.

# COMMAND ----------

import pandas as pd

print("Inference results:")

batch_size = 100  # Number of records per batch
num_batches = (len(dataframe_records) + batch_size - 1) // batch_size  # Total number of batches

all_predictions = []
all_models = []

# Process data in batches
for i in range(num_batches):
    batch_records = dataframe_records[i * batch_size:(i + 1) * batch_size]  # Slice batch

    # Query the model serving endpoint
    query_response = w.serving_endpoints.query(name=endpoint_name, dataframe_records=batch_records)

    # Collect predictions and model served details
    all_predictions.extend(query_response.predictions)
    all_models.extend([query_response.served_model_name] * len(query_response.predictions))  # Duplicate model name per prediction

# Convert to DataFrame
results_df = pd.DataFrame({
    "prediction": all_predictions,
    "model_served": all_models
})

# Count occurrences of predictions
count_results = results_df['prediction'].value_counts().reset_index()
count_results.columns = ['prediction', 'count']

# Display aggregated count of predictions
display(count_results)

# Aggregate count of predictions per model
model_count_results = results_df.groupby(["model_served", "prediction"]).size().reset_index(name="count")

# Display results grouped by model and prediction type
display(model_count_results)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC This demonstration showed how to deploy and serve machine learning models in real time using **Model Serving** with **offline feature tables** stored in Unity Catalog.  
# MAGIC You learned how to:
# MAGIC
# MAGIC - Register and manage models with **Champion/Challenger** aliases.  
# MAGIC - Configure a **Model Serving endpoint** with multiple model versions for **A/B testing**.  
# MAGIC - Send inference requests and visualize traffic distribution between model versions.  
# MAGIC
# MAGIC This workflow highlights how Databricks simplifies real-time deployment and model experimentation using reliable, batch-refreshed **offline Delta feature tables**.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>