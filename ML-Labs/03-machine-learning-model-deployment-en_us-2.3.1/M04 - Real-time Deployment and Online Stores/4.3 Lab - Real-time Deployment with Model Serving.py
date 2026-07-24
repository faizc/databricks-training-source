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
# MAGIC # LAB - Real-time Deployment with Model Serving
# MAGIC
# MAGIC In this lab, you will deploy ML models with Model Serving using **offline feature tables** (Delta in Unity Catalog). This lab includes **two** sections.
# MAGIC
# MAGIC In the first section, you will deploy a model for real-time inference with Model Serving's **UI**. This section will demonstrate the most basic and simple way of deploying models with Model Serving.
# MAGIC
# MAGIC In the second section, you will deploy a model **programmatically using the Databricks SDK (API)**.
# MAGIC
# MAGIC For both sections, data preparation, model fitting and model registration are already done for you! You just need to focus on the deployment part.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC * Simple real-time deployment
# MAGIC   - **Task 1:** Serve the model using the UI
# MAGIC   - **Task 2:** Query the endpoint
# MAGIC
# MAGIC * API-based real-time deployment 
# MAGIC   - **Task 3:** Create an offline feature table
# MAGIC   - **Task 4:** Create a derived feature using a SQL function
# MAGIC   - **Task 5:** Prepare the feature table for inference
# MAGIC   - **Task 6:** (Optional) Define features with FeatureLookup/FeatureFunction for illustration
# MAGIC   - **Task 7:** Create training set and fit the model (offline join)
# MAGIC   - **Task 8:** Deploy the model
# MAGIC   - **Task 9:** Query the endpoint

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
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the lab. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-4.3

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data and Model Preparation
# MAGIC
# MAGIC Before you start the deployment process, you will need to fit and register a model. In this section, you will load dataset, fit a model and register it with UC.
# MAGIC
# MAGIC **Note:** All necessary code is provided, which means you don't need to complete anything in this section.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Dataset

# COMMAND ----------

from pyspark.sql.functions import col, monotonically_increasing_id

## Set the path of the dataset
shared_volume_name = 'cdc-diabetes' # From Marketplace
csv_name = 'diabetes_binary_5050split_BRFSS2015' # CSV file name
dataset_path = f"{DA.paths.datasets.cdc_diabetes}/{shared_volume_name}/{csv_name}.csv" # Full path


df = spark.read.csv(dataset_path, inferSchema=True, header=True, multiLine=True, escape='"')\
    .na.drop(how='any')

df = df.withColumn("uniqueID", monotonically_increasing_id())   # Add unique_id column

## Dataset specs
primary_key = "uniqueID"
response = "Diabetes_binary"

## Separate features and ground-truth
features_df = df.drop(response)
response_df = df.select(primary_key, response)

## Convert data to pandas dataframes
X_train_pdf = features_df.drop(primary_key).toPandas()
Y_train_pdf = response_df.drop(primary_key).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Model Registry with UC
# MAGIC
# MAGIC Before we start model deployment, we need to fit and register a model. In this lab, **we will log models to Unity Catalog**, which means first we need to setup the **MLflow Model Registry URI**.

# COMMAND ----------

import mlflow

## Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper Class for Model Creation

# COMMAND ----------

import time
import warnings
from mlflow.types.utils import _infer_schema
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeClassifier
from databricks.feature_engineering import FeatureEngineeringClient

model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_diabetes_model" ## Use 3-level namespace

def get_latest_model_version(model_name):
    """Helper function to get the latest model version as a string"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    model_version_list = [model_version_info.version for model_version_info in model_version_infos]
    ## Convert to integers for correct numeric comparison
    model_version_int_list = list(map(int, model_version_list))
    ## Find the maximum and convert back to a string
    return str(max(model_version_int_list))

def fit_and_register_model(X, Y, model_name_=model_name, random_state_=42, model_alias=None, log_with_fs=False, training_set_spec_=None):
    """Helper function to train and register a decision tree model"""

    clf = DecisionTreeClassifier(random_state=random_state_)
    with mlflow.start_run(run_name="LAB4-Real-Time-Deployment") as mlflow_run:

        ## Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_models=False,
            log_post_training_metrics=True,
            silent=True)
        
        clf.fit(X, Y)

        ## Log model and push to registry
        if log_with_fs:
            # Infer output schema
            try:
                output_schema = _infer_schema(Y)
            except Exception as e:
                warnings.warn(f"Could not infer model output schema: {e}")
                output_schema = None
            
            ## Log using feature engineering client and push to registry
            fe = FeatureEngineeringClient()
            fe.log_model(
                model = clf,
                artifact_path = "decision_tree",
                flavor = mlflow.sklearn,
                training_set = training_set_spec_,
                output_schema = output_schema,
                registered_model_name = model_name_
            )
        
        else:
            signature = infer_signature(X, Y)
            example = X[:3]
            mlflow.sklearn.log_model(
                clf,
                name = "decision_tree",
                signature = signature,
                input_example = example,
                registered_model_name = model_name_
            )

        ## Set model alias
        if model_alias:
            time.sleep(10) ## Wait 10secs for model version to be created
            client.set_registered_model_alias(model_name_, model_alias, get_latest_model_version(model_name_))

    return clf

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and Register the Model
# MAGIC
# MAGIC Before we start model deployment process, we will **fit and register a model**. The model's alias will be set to `Production` and it will be served with Model Serving in the next step.

# COMMAND ----------

model = fit_and_register_model(X_train_pdf, Y_train_pdf, model_name, 42, "Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Real-time Model Deployment
# MAGIC
# MAGIC Now that the model is registered and ready for deployment, the next step is to create a serving endpoint with Model Serving and serve the model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1: Serve the Model Using the UI
# MAGIC
# MAGIC Serve the **"Production"** model that we registered in the previous section using the following endpoint configuration.
# MAGIC
# MAGIC **Configuration:**
# MAGIC
# MAGIC * Name: `la4-1-diabetes-model`
# MAGIC
# MAGIC * Compute Size: `small` (CPU)
# MAGIC
# MAGIC * Autoscaling: `Scale to zero`
# MAGIC
# MAGIC * Tags: Define tags that might be meaningful for this deployment
# MAGIC
# MAGIC
# MAGIC **💡 Note:** Endpoint creation will take sometime. Therefore, you can work on the next section  while the endpoint is created for you.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2: Query the Endpoint 
# MAGIC
# MAGIC Test the model deployment using the **Query endpoint** feature in browsers. Use the provided **Example request** payload to use the model for inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-time Model Deployment with Model Serving
# MAGIC
# MAGIC In this section, you will deploy a model using **Model Serving** with an **offline feature table** stored in Unity Catalog.  
# MAGIC Unlike the previous section where you deployed through the **UI**, this time you will create and configure the serving endpoint programmatically using the **Databricks SDK (API)**.
# MAGIC
# MAGIC First, you will review the registered model that was trained and logged using features from a Delta table in Unity Catalog.  
# MAGIC Then, you will deploy this model as a real-time serving endpoint using the API. Finally, you will query the endpoint to perform live inference using sample data records.
# MAGIC
# MAGIC This workflow demonstrates how to automate model deployment with Model Serving using **offline feature tables**, providing a foundation for scalable and reproducible production ML workflows.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3: Create an Offline Feature Table
# MAGIC
# MAGIC Let's create an **offline feature table** to store the features that will be used for model training and batch or real-time inference.  
# MAGIC
# MAGIC For this task, you will set up the feature table as follows:
# MAGIC
# MAGIC - The feature table will include **all feature columns** from the dataset  
# MAGIC - Define a **primary key** for uniquely identifying each record  
# MAGIC - Add a **description** for the table in Unity Catalog  
# MAGIC
# MAGIC This table will be stored as a **Delta table** in Unity Catalog and can later be accessed directly by Model Serving for inference.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

## Define feature table name and initialize Feature Engineering client
feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.diabetes_features"
fe = FeatureEngineeringClient()

## Create the offline feature table
fe.<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 3: Create an Offline Feature Table
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC
# MAGIC ## Define the feature table name in Unity Catalog
# MAGIC feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.diabetes_features"
# MAGIC
# MAGIC ## Initialize Feature Engineering client
# MAGIC fe = FeatureEngineeringClient()
# MAGIC
# MAGIC ## Create the offline feature table
# MAGIC fe.create_table(
# MAGIC     name=feature_table_name,
# MAGIC     df=features_df,
# MAGIC     primary_keys=[primary_key],
# MAGIC     description="Offline feature table containing diabetes dataset features for model training and inference"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4: Create a Derived Feature Using SQL Function
# MAGIC
# MAGIC In this task, you will create a **derived feature** based on existing columns in the dataset.  
# MAGIC Instead of directly using **Education** and **Income**, you will compute a new field called  
# MAGIC **Education-Adjusted Income Index (EAI)** that represents a weighted interaction between the two.
# MAGIC
# MAGIC This field is calculated using the formula:  
# MAGIC **`Education-Adjusted Income = Income × Education`**
# MAGIC
# MAGIC *Note:* In real-world scenarios, correlated features such as income and education should be carefully examined for redundancy or multicollinearity. However, here the goal is to demonstrate how to define and register a simple SQL function in Unity Catalog that can be referenced during data processing or model training.
# MAGIC
# MAGIC The function should be structured as follows, using the variable names defined below:  
# MAGIC - **Function name:** `eai_function`  
# MAGIC - **Input:** `Income`, `Education`  
# MAGIC - **Output:** `eai`
# MAGIC

# COMMAND ----------

## Create or replace a SQL function to compute the derived feature
spark.sql(<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 4: Create a Derived Feature Using SQL Function
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ## Create or replace a SQL function to compute the derived feature
# MAGIC spark.sql(f"""
# MAGIC CREATE OR REPLACE FUNCTION eai_function (Income DOUBLE, Education DOUBLE)
# MAGIC RETURNS DOUBLE
# MAGIC LANGUAGE PYTHON AS
# MAGIC $$
# MAGIC eai = Income * Education
# MAGIC return eai
# MAGIC $$
# MAGIC """)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 5: Prepare the Feature Table for Inference
# MAGIC
# MAGIC In this task, you will make sure that the **offline feature table** you created earlier can be used directly for **model inference**.  
# MAGIC This step ensures that the feature table is properly configured in Unity Catalog and that change tracking is enabled for incremental updates.
# MAGIC
# MAGIC **Perform the following steps:**
# MAGIC
# MAGIC * Enable **Change Data Feed (CDF)** on the feature table to allow incremental updates and lineage tracking  
# MAGIC * Verify that the feature table is registered in Unity Catalog and available for use by Model Serving  
# MAGIC
# MAGIC The resulting table will remain an **offline Delta table**, suitable for both batch and real-time inference through Model Serving.

# COMMAND ----------

from pprint import pprint
from databricks.sdk import WorkspaceClient

## Initialize the Workspace client
workspace = WorkspaceClient()

## Enable Change Data Feed (CDF) on the offline feature table
spark.sql(<FILL_IN>)

## Retrieve and display table details from Unity Catalog
feature_table_details = <FILL_IN>

pprint(feature_table_details)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 5: Prepare the Feature Table for Inference
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from pprint import pprint
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC ## Initialize the Workspace client
# MAGIC workspace = WorkspaceClient()
# MAGIC
# MAGIC ## Enable Change Data Feed (CDF) on the offline feature table
# MAGIC spark.sql(f"""ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
# MAGIC
# MAGIC ## Retrieve and display table details from Unity Catalog
# MAGIC feature_table_details = workspace.tables.get(feature_table_name)
# MAGIC
# MAGIC pprint(feature_table_details)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 6: Define Features
# MAGIC
# MAGIC Now that you have an **offline feature table** and a registered SQL function, you will combine them so the model can use both the stored features and the derived feature during **training**.

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup, FeatureFunction
## Define combined feature lookup (offline table) and derived feature function
features=[<FILL_IN>]

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 6: Define Features
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureLookup, FeatureFunction
# MAGIC
# MAGIC ## Define combined feature lookup (offline table) and derived feature function
# MAGIC features = [
# MAGIC     FeatureLookup(
# MAGIC         table_name=feature_table_name,
# MAGIC         lookup_key=primary_key
# MAGIC     ),
# MAGIC     FeatureFunction(
# MAGIC         udf_name="eai_function",
# MAGIC         output_name="eai",
# MAGIC         input_bindings={
# MAGIC             "Education": "Education",
# MAGIC             "Income": "Income"
# MAGIC         },
# MAGIC     ),
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 7: Create Training Set and Fit the Model
# MAGIC
# MAGIC Now that all feature configuration is set and ready, create training set and fit the model.

# COMMAND ----------

from pyspark.sql import functions as F

## Build an offline training dataframe (join features + label; compute derived feature offline)
training_df_offline = (
    <FILL_IN>
)

## Convert to pandas
X_train_pdf2 = training_df_offline.drop(primary_key, response).toPandas()
Y_train_pdf2 = training_df_offline.select(response).toPandas()

## Fit and register the model (OFFLINE: no FS metadata)
model_name_2 = f"{DA.catalog_name}.{DA.schema_name}.ml_diabetes_model_fe"
model_fe = fit_and_register_model(
    X_train_pdf2,
    Y_train_pdf2,
    model_name_2,
    20,
    log_with_fs=False,          
    training_set_spec_=None 
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 7: Create Training Set and Fit the Model
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from pyspark.sql import functions as F
# MAGIC
# MAGIC ## Build an offline training dataframe (join features + label; compute derived feature offline)
# MAGIC training_df_offline = (
# MAGIC     features_df.join(response_df, on=primary_key, how="inner")
# MAGIC                .withColumn("eai", F.col("Income") * F.col("Education"))
# MAGIC )
# MAGIC
# MAGIC ## Convert to pandas
# MAGIC X_train_pdf2 = training_df_offline.drop(primary_key, response).toPandas()
# MAGIC Y_train_pdf2 = training_df_offline.select(response).toPandas()
# MAGIC
# MAGIC ## Fit and register the model (OFFLINE: no FS metadata)
# MAGIC model_name_2 = f"{DA.catalog_name}.{DA.schema_name}.ml_diabetes_model_fe"
# MAGIC model_fe = fit_and_register_model(
# MAGIC     X_train_pdf2,
# MAGIC     Y_train_pdf2,
# MAGIC     model_name_2,
# MAGIC     20,
# MAGIC     log_with_fs=False,          
# MAGIC     training_set_spec_=None 
# MAGIC )    
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 8: Deploy the Model (Offline Features)
# MAGIC
# MAGIC Create a serving endpoint with the following configuration:
# MAGIC
# MAGIC * Autoscaling: `Scale-to-zero`
# MAGIC * Compute size: `Small`
# MAGIC
# MAGIC **💡 Note:** Endpoint creation will take some time. Please wait while the endpoint is created.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag

## Initialize Workspace client
w = WorkspaceClient()

## Get the model version that will be served (from the offline-logged model)
model_version = <FILL_IN>

## Define the endpoint configuration
endpoint_config_dict = {
    "served_models": [
        {
            <FILL_IN>
        }
    ]
}
endpoint_config = <FILL_IN>

## Define endpoint name
endpoint_name = <FILL_IN>

## Create the serving endpoint
try:
    w.<FILL_IN>(
        name=<FILL_IN>,
        config=<FILL_IN>,
        tags=[EndpointTag.from_dict({"key": "db_academy", "value": "lab4_serve_offline_model"})]
    )
    print(f"Creating endpoint {endpoint_name} with model {model_name_2} version {model_version}")
except Exception as e:
    if "already exists" in e.args[0]:
        print(f"Endpoint with name {endpoint_name} already exists")
    else:
        raise e

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 8: Deploy the Model (Offline Features)
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC ## Get the model version that will be served (from the offline-logged model above)
# MAGIC model_version = get_latest_model_version(model_name_2)
# MAGIC
# MAGIC endpoint_config_dict = {
# MAGIC     "served_models": [
# MAGIC         {
# MAGIC             "model_name": model_name_2,
# MAGIC             "model_version": model_version,
# MAGIC             "scale_to_zero_enabled": True,
# MAGIC             "workload_size": "Small"
# MAGIC         }
# MAGIC     ]
# MAGIC }
# MAGIC endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)
# MAGIC
# MAGIC endpoint_name = f"ML_AS_03_Lab4_{DA.unique_name('_')}"
# MAGIC
# MAGIC try:
# MAGIC     w.serving_endpoints.create_and_wait(
# MAGIC         name=endpoint_name,
# MAGIC         config=endpoint_config,
# MAGIC         tags=[EndpointTag.from_dict({"key": "db_academy", "value": "lab4_serve_offline_model"})]
# MAGIC     )
# MAGIC     print(f"Creating endpoint {endpoint_name} with model {model_name_2} version {model_version}")
# MAGIC except Exception as e:
# MAGIC     if "already exists" in e.args[0]:
# MAGIC         print(f"Endpoint with name {endpoint_name} already exists")
# MAGIC     else:
# MAGIC         raise e
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 9: Query the Endpoint
# MAGIC
# MAGIC After the endpoint is created, it is time to test it. Use the following hard-coded test-sample to query the endpoint using the API.

# COMMAND ----------

# Sample a few records for testing
payload = X_train_pdf2.sample(3, random_state=42).to_dict(orient="records")

# COMMAND ----------

## Query the serving endpoint with test-sample
query_response = w.serving_endpoints.<FILL_IN>

print("Inference results:", query_response.predictions)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 9: Query the Endpoint
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ## Query the serving endpoint
# MAGIC query_response = w.serving_endpoints.query(
# MAGIC     name=endpoint_name,
# MAGIC     dataframe_records=payload
# MAGIC )
# MAGIC
# MAGIC print("Inference results:", query_response.predictions)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC Great job completing this lab!  
# MAGIC In this lab, you successfully explored two key ways of deploying machine learning models with **Model Serving** using **offline feature tables**.
# MAGIC
# MAGIC - In the **first section**, you deployed a model using the **UI**, demonstrating the simplest way to expose a registered model for real-time inference.  
# MAGIC - In the **second section**, you used the **Databricks SDK (API)** to automate model deployment. You created and prepared an offline feature table in Unity Catalog, defined a derived feature, trained and registered an offline model **without Feature Store metadata**, and deployed it to a real-time serving endpoint.  
# MAGIC - Finally, you tested your endpoint by sending complete feature vectors for live inference.
# MAGIC
# MAGIC This workflow provides a foundation for building scalable, reproducible, and fully managed **real-time inference pipelines** using Model Serving with **offline Delta-based feature tables**.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>