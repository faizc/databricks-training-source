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
# MAGIC # LAB - Batch Deployment
# MAGIC
# MAGIC Welcome to the "Batch Deployment" lab! This lab focuses on batch deployment of machine learning models using Databricks. You will engage in tasks related to model inference, model registry, and explore performance results for feature such as Liquid Clustering using `CLUSTER BY`.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC By the end of this lab, you will be able to;
# MAGIC
# MAGIC + **Task 1: Load Dataset**
# MAGIC     + Load Dataset
# MAGIC     + Split the dataset into features and response sets
# MAGIC
# MAGIC + **Task 2: Inference with feature table**
# MAGIC
# MAGIC     + Create Feature Table
# MAGIC     + Setup Feature Lookups
# MAGIC     + Fit and Register a Model with UC using Feature Table
# MAGIC     + Perform batch inference using Feature Engineering's  **`score_batch`** method.
# MAGIC
# MAGIC + **Task 3: Assess Liquid Clustering:**
# MAGIC
# MAGIC     + Evaluate the performance results for specific optimization techniques:
# MAGIC         + Liquid Clustering
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
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the Lab, run the provided classroom setup script. This script will define configuration variables necessary for the lab. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-2.2

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this Lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Load Dataset
# MAGIC
# MAGIC + Load a dataset:
# MAGIC   + Define the dataset path
# MAGIC   + Define the primary key (`customerID`), response variable (`Churn`), and feature variables (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) for further processing.
# MAGIC   + Read the dataset, casting relevant columns to the correct data types, and drop any rows with missing values
# MAGIC + Split the dataset into training and testing sets
# MAGIC   + Separate the features and the response for the training set
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col
## Load dataset with spark
shared_volume_name = 'telco' # From Marketplace
csv_name = 'telco-customer-churn-missing' # CSV file name
dataset_p_telco = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

## Features to use
primary_key = "customerID"
response = "Churn"
features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

## Read dataset (and drop nan)
telco_df = spark.read.csv(dataset_p_telco, inferSchema=True, header=True, multiLine=True, escape='"')\
            .withColumn("TotalCharges", F.expr("try_cast(trim(TotalCharges) as double)"))\
            .withColumn("SeniorCitizen", col("SeniorCitizen").cast('double'))\
            .withColumn("Tenure", col("tenure").cast('double'))\
            .na.drop(how='any')

## Split with 80 percent of the data in train_df and 20 percent of the data in test_df
train_df, test_df = telco_df.randomSplit([.8, .2], seed=42)

## Separate features and ground-truth
features_df = train_df.select<FILL_IN>
response_df = train_df.select<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 1: Load Dataset
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from pyspark.sql.functions import col
# MAGIC
# MAGIC ## Load dataset with spark
# MAGIC shared_volume_name = 'telco' # From Marketplace
# MAGIC csv_name = 'telco-customer-churn-missing' # CSV file name
# MAGIC dataset_p_telco = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path
# MAGIC
# MAGIC ## features to use
# MAGIC primary_key = "customerID"
# MAGIC response = "Churn"
# MAGIC features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
# MAGIC
# MAGIC ## Read dataset (and drop nan)
# MAGIC telco_df = spark.read.csv(dataset_p_telco, inferSchema=True, header=True, multiLine=True, escape='"')\
# MAGIC             .withColumn("TotalCharges", F.expr("try_cast(trim(TotalCharges) as double)"))\
# MAGIC             .withColumn("SeniorCitizen", col("SeniorCitizen").cast('double'))\
# MAGIC             .withColumn("Tenure", col("tenure").cast('double'))\
# MAGIC             .na.drop(how='any')
# MAGIC
# MAGIC ## Split with 80 percent of the data in train_df and 20 percent of the data in test_df
# MAGIC train_df, test_df = telco_df.randomSplit([.8, .2], seed=42)
# MAGIC
# MAGIC ## Separate features and ground-truth
# MAGIC features_df = train_df.select(primary_key, *features)
# MAGIC response_df = train_df.select(primary_key, response)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 2: Inference with feature table
# MAGIC In this task, you will perform batch inference using a feature table. Follow the steps below:
# MAGIC
# MAGIC + **Step 1: Create Feature Table**
# MAGIC
# MAGIC + **Step 2: Setup Feature Lookups**
# MAGIC
# MAGIC + **Step 3: Fit and Register a Model with UC using Feature Table**
# MAGIC
# MAGIC + **Step 4: Use the Model for Inference**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 1: Create Feature Table**
# MAGIC   + Begin by creating a feature table that incorporates the relevant features for inference. This involves selecting the appropriate columns, performing any necessary transformations, and storing the resulting data in a feature table.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

## Prepare feature set
features_df_all = telco_df.select(primary_key, *features)

## Feature table definition
fe = FeatureEngineeringClient()
feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"

## Drop table if exists
try:
    fe.drop_table(name=feature_table_name)
except:
    pass

## Create feature table
fe.create_table(
    <FILL_IN>
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Step 1: Create Feature Table
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC
# MAGIC ## Prepare feature set
# MAGIC features_df_all = telco_df.select(primary_key, *features)
# MAGIC
# MAGIC ## Feature table definition
# MAGIC fe = FeatureEngineeringClient()
# MAGIC feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"
# MAGIC
# MAGIC ## Drop table if exists
# MAGIC try:
# MAGIC     fe.drop_table(name=feature_table_name)
# MAGIC except:
# MAGIC     pass
# MAGIC
# MAGIC ## Create feature table
# MAGIC fe.create_table(
# MAGIC     name=feature_table_name,
# MAGIC     df=features_df_all,
# MAGIC     primary_keys=[primary_key],
# MAGIC     description="Lab feature table"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 2: Setup Feature Lookups**
# MAGIC   + Set up a feature lookup to create a training set from the feature table. 
# MAGIC   + Specify the `lookup_key` based on the columns that uniquely identify records in your feature table.

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup

fl_handle = <FILL_IN>

## Create a training set based on feature lookup
training_set_spec = fe.<FILL_IN>

## Load training dataframe based on defined feature-lookup specification
training_df = <FILL_IN>.load_df()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Step 2: Setup Feature Lookups
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureLookup
# MAGIC
# MAGIC fl_handle = FeatureLookup(
# MAGIC     table_name=feature_table_name,
# MAGIC     lookup_key=[primary_key]
# MAGIC )
# MAGIC
# MAGIC ##  Create a training set based on feature lookup
# MAGIC training_set_spec = fe.create_training_set(
# MAGIC     df=response_df,
# MAGIC     label=response,
# MAGIC     feature_lookups=[fl_handle],
# MAGIC     exclude_columns=[primary_key]
# MAGIC )
# MAGIC
# MAGIC ## Load training dataframe based on defined feature-lookup specification
# MAGIC training_df = training_set_spec.load_df()
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 3: Fit and Register a Model with UC using Feature Table**
# MAGIC   + Fit and register a Machine Learning Model using the created training set.
# MAGIC   + Train a model on the training set and register it in the model registry.

# COMMAND ----------

import mlflow
import warnings
from mlflow.types.utils import _infer_schema

## Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

## Helper function that we will use for getting latest version of a model
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

## Train a sklearn Decision Tree Classification model
from sklearn.tree import DecisionTreeClassifier

## Convert data to pandas dataframes
X_train_pdf = training_df.drop(<FILL_IN>, <FILL_IN>).toPandas()
Y_train_pdf = training_df.select(<FILL_IN>).toPandas()
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

## End the active MLflow run before starting a new one
mlflow.end_run()

with mlflow.start_run(run_name="Model-Batch-Deployment-lab-With-FS") as mlflow_run:

    ## Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        log_post_training_metrics=True,
        silent=True)
    
    clf.fit(<FILL_IN>, <FILL_IN>)

    ## Infer output schema
    try:
        output_schema = _infer_schema(<FILL_IN>)
    except Exception as e:
        warnings.warn(f"Could not infer model output schema: {e}")
        output_schema = None

    model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_model"
    
    ## Log using feature engineering client and push to registry
    fe.<FILL_IN>

    ## Set model alias (i.e. Champion)
    client.set_registered_model_alias(<FILL_IN>, <FILL_IN>, <FILL_IN>(<FILL_IN>))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Step 3: Fit and Register a Model with UC using Feature Table
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC import warnings
# MAGIC from mlflow.types.utils import _infer_schema
# MAGIC
# MAGIC ## Point to UC model registry
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC client = mlflow.MlflowClient()
# MAGIC
# MAGIC ## helper function that we will use for getting latest version of a model
# MAGIC def get_latest_model_version(model_name):
# MAGIC     """Helper function to get latest model version"""
# MAGIC     model_version_infos = client.search_model_versions("name = '%s'" % model_name)
# MAGIC     return max([model_version_info.version for model_version_info in model_version_infos])
# MAGIC
# MAGIC ## Train a sklearn Decision Tree Classification model
# MAGIC from sklearn.tree import DecisionTreeClassifier
# MAGIC
# MAGIC ## Convert data to pandas dataframes
# MAGIC X_train_pdf = training_df.drop(primary_key, response).toPandas()
# MAGIC Y_train_pdf = training_df.select(response).toPandas()
# MAGIC clf = DecisionTreeClassifier(max_depth=3, random_state=42)
# MAGIC
# MAGIC ## End the active MLflow run before starting a new one
# MAGIC mlflow.end_run()
# MAGIC
# MAGIC with mlflow.start_run(run_name="Model-Batch-Deployment-lab-With-FS") as mlflow_run:
# MAGIC
# MAGIC     ## Enable automatic logging of input samples, metrics, parameters, and models
# MAGIC     mlflow.sklearn.autolog(
# MAGIC         log_input_examples=True,
# MAGIC         log_models=False,
# MAGIC         log_post_training_metrics=True,
# MAGIC         silent=True)
# MAGIC
# MAGIC     clf.fit(X_train_pdf, Y_train_pdf)
# MAGIC
# MAGIC     ## Infer output schema
# MAGIC     try:
# MAGIC         output_schema = _infer_schema(Y_train_pdf)
# MAGIC     except Exception as e:
# MAGIC         warnings.warn(f"Could not infer model output schema: {e}")
# MAGIC         output_schema = None
# MAGIC
# MAGIC     model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_model"
# MAGIC
# MAGIC     ## Log using feature engineering client and push to registry
# MAGIC     fe.log_model(
# MAGIC         model=clf,
# MAGIC         artifact_path="decision_tree",
# MAGIC         flavor=mlflow.sklearn,
# MAGIC         training_set=training_set_spec,
# MAGIC         output_schema=output_schema,
# MAGIC         registered_model_name= model_name
# MAGIC     )
# MAGIC
# MAGIC     ## Set model alias (i.e. Champion)
# MAGIC     client.set_registered_model_alias(model_name, "Champion", get_latest_model_version(model_name))
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 4: Use the Model for Inference**
# MAGIC   + Join the lookup DataFrame with the feature table on the primary key to retrieve the full feature set.
# MAGIC   + Wrap the trained model in a `pandas_udf` and apply it to the feature-enriched DataFrame for distributed batch inference.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

## Load the model
model_uri = f"models:/<FILL_IN>"

## Define the test dataset (lookup key only)
test_features_df = test_df.select(<FILL_IN>)

## Retrieve features from feature table
feature_table_df = <FILL_IN>
lookup_with_features_df = <FILL_IN>

## Use the trained sklearn model as a UDF closure variable for distributed inference
@pandas_udf(StringType())
def predict_func(*cols):
    import pandas as pd
    input_df = pd.DataFrame({name: col for name, col in zip(features, cols)})
    return pd.Series(<FILL_IN>.predict(input_df).astype(str))

## Perform batch inference using feature-enriched lookup df
result_df = lookup_with_features_df.withColumn(<FILL_IN>, predict_func(<FILL_IN>))

## Display the inference results
display(result_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Step 4: Use the Model for Inference
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC from pyspark.sql.functions import pandas_udf
# MAGIC from pyspark.sql.types import StringType
# MAGIC
# MAGIC ## Load the model
# MAGIC model_uri = f"models:/{model_name}@champion"
# MAGIC
# MAGIC ## Define the test dataset (lookup key only)
# MAGIC test_features_df = test_df.select("customerID")
# MAGIC
# MAGIC ## Retrieve features from feature table (replicates score_batch feature lookup)
# MAGIC feature_table_df = spark.table(feature_table_name)
# MAGIC lookup_with_features_df = test_features_df.join(feature_table_df, on="customerID", how="left")
# MAGIC
# MAGIC ## clf is the trained sklearn model in memory — picklable and works as a UDF closure variable
# MAGIC @pandas_udf(StringType())
# MAGIC def predict_func(*cols):
# MAGIC     import pandas as pd
# MAGIC     input_df = pd.DataFrame({name: col for name, col in zip(features, cols)})
# MAGIC     return pd.Series(clf.predict(input_df).astype(str))
# MAGIC
# MAGIC ## Perform batch inference using feature-enriched lookup df
# MAGIC result_df = lookup_with_features_df.withColumn(
# MAGIC     "prediction",
# MAGIC     predict_func(*[lookup_with_features_df[f] for f in features])
# MAGIC )
# MAGIC
# MAGIC ## Display the inference results
# MAGIC display(result_df)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Assess Liquid Clustering:
# MAGIC
# MAGIC Evaluate the performance results for specific optimization techniques, such as: Liquid Clustering Follow the step-wise instructions below:  
# MAGIC + **Step 1:** Create `batch_inference_liquid_clustering` table and import the following columns: `customerID`, `Churn`, `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, and `prediction`.
# MAGIC + **Step 2:**  Begin by assessing Liquid Clustering, an optimization technique for improving performance by physically organizing data based on a specified clustering column.
# MAGIC + **Step 3:**  Optimize the target table for Liquid Clustering.
# MAGIC + **Step 4:** Specify the `CLUSTER BY` clause with the desired columns (e.g., (customerID, tenure)) to enable Liquid Clustering on the table.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE batch_inference_liquid_clustering(
# MAGIC   <FILL_IN> 
# MAGIC   )

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 3: Assess Liquid Clustering
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```sql
# MAGIC CREATE OR REPLACE TABLE batch_inference_liquid_clustering(
# MAGIC   customerID STRING,
# MAGIC   Churn STRING,
# MAGIC   SeniorCitizen DOUBLE,
# MAGIC   tenure DOUBLE,
# MAGIC   MonthlyCharges DOUBLE,
# MAGIC   TotalCharges DOUBLE,
# MAGIC   prediction STRING
# MAGIC   )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE batch_inference_liquid_clustering;
# MAGIC ALTER TABLE batch_inference_liquid_clustering
# MAGIC CLUSTER BY (<FILL_IN>, <FILL_IN>);

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 3: Assess Liquid Clustering
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```sql
# MAGIC OPTIMIZE batch_inference_liquid_clustering;
# MAGIC ALTER TABLE batch_inference_liquid_clustering
# MAGIC CLUSTER BY (customerID, tenure);
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC This lab provides you with hands-on experience in batch deployment, covering model inference, Model Registry usage, and the impact of features like Liquid Clustering on performance. you will gain practical insights into deploying models at scale in a batch-oriented environment.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>