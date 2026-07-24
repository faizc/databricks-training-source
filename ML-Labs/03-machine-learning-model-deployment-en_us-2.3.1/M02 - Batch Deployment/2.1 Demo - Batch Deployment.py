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
# MAGIC # Batch Deployment
# MAGIC
# MAGIC Batch inference is the most common way of deploying machine learning models.  This lesson introduces various strategies for deploying models using batch including Spark. In addition, we will show how to enable optimizations for Delta tables.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo you will be able to:*
# MAGIC
# MAGIC * Load a logged Model Registry model using `pyfunc`.
# MAGIC
# MAGIC * Compute predictions using `pyfunc` APIs.
# MAGIC
# MAGIC * Perform batch inference using Feature Engineering's `score_batch` method.
# MAGIC
# MAGIC * Materialize predictions into inference tables (Delta Lake).
# MAGIC
# MAGIC * Perform common write optimizations like liquid clustering, predictive optimization to maximize data skipping and on inference tables.

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
# MAGIC **🚨 Prerequisites:** 
# MAGIC * **Feature Engineering** and **Feature Store** are not the focus of this lesson. This course expect that you already know these topics. If not, you can check the **Data Preparation for Machine Learning** course.
# MAGIC
# MAGIC * Model development with MLFlow is not in the scope of this course. If you need to refresh your knowledge about model tracking and logging, you can check the **Machine Learning Model Development** course.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;"><div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">  <img src="../Includes/Images/genie-code.png" style="height: 64px; margin-bottom: 10px;">  <div style="font-size: 15pt; color: #0B2026; line-height: 1.7; margin-bottom: 16px;">    Want to know more about batch inference and Delta Lake optimizations in Databricks? Ask Genie Code. Click on the genie icon <img src="../Includes/Images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.  </div>  <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0B2026;">    <span id="genie-query-2-1" style="flex: 1;">How do I perform batch inference in Databricks using an MLflow model with pyfunc and the Feature Engineering score_batch method? How do Liquid Clustering and Predictive Optimization improve query performance and data skipping on the resulting Delta inference tables, and when should I use each?</span>    <button onclick="      var text = document.getElementById('genie-query-2-1').innerText;      var ta = document.createElement('textarea');      ta.value = text;      ta.style.position = 'fixed';      ta.style.opacity = '0';      document.body.appendChild(ta);      ta.select();      document.execCommand('copy');      document.body.removeChild(ta);      this.innerText = 'Copied!';      var btn = this;      setTimeout(function(){ btn.innerText = 'Copy'; }, 2000);    " style="background: #FF5F46; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-size: 13pt; cursor: pointer; white-space: nowrap;">Copy</button>  </div></div></div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-2.1

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
# MAGIC For this demonstration, we will utilize a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including gender, as well as internet subscription details such as subscription plans and payment methods.
# MAGIC
# MAGIC After loading the dataset, we will perform simple **data cleaning and feature selection**. 
# MAGIC
# MAGIC In the final step, we will split the dataset into **features** and **response** sets.

# COMMAND ----------

from pyspark.sql.functions import col

# Load dataset with spark
shared_volume_name = 'telco' # From Marketplace
csv_name = 'telco-customer-churn-missing' # CSV file name
dataset_p_telco = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

# features to use
primary_key = "customerID"
response = "Churn"
features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] # Keeping numerical only for simplicity and demo purposes

# Read dataset (and drop nan)
telco_df = spark.read.csv(dataset_p_telco, inferSchema=True, header=True, multiLine=True, escape='"')\
            .withColumn("TotalCharges", F.expr("try_cast(trim(TotalCharges) as double)"))\
            .withColumn("SeniorCitizen", col("SeniorCitizen").cast('double'))\
            .withColumn("Tenure", col("tenure").cast('double'))\
            .na.drop(how='any')

# Split with 80 percent of the data in train_df and 20 percent of the data in test_df
train_df, test_df = telco_df.randomSplit([.8, .2], seed=42)

# Separate features and ground-truth
features_df = train_df.select(primary_key, *features)
response_df = train_df.select(primary_key, response)

# review the features dataset
display(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Batch Deployment - Without Feature Store
# MAGIC
# MAGIC This demo will cover two main batch deployment methods. The first method is deploying models without a feature table. For the second method, we will use a feature table to train the model and later use the feature table for inference.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Setup Model Registry with UC
# MAGIC
# MAGIC Before we start model deployment, we need to fit and register a model. In this demo, **we will log models to Unity Catalog**, which means first we need to setup the **MLflow Model Registry URI**.

# COMMAND ----------

import mlflow

# Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# helper function that we will use for getting latest version of a model
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and Register a Model with UC

# COMMAND ----------

# Train a sklearn Decision Tree Classification model
from sklearn.tree import DecisionTreeClassifier
from mlflow.models import infer_signature

# Convert data to pandas dataframes
X_train_pdf = features_df.drop(primary_key).toPandas()
Y_train_pdf = response_df.drop(primary_key).toPandas()
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# Use 3-level namespace for model name
model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_model" 

with mlflow.start_run(run_name="Model-Batch-Deployment-Demo") as mlflow_run:

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        log_post_training_metrics=True,
        silent=True)
    
    clf.fit(X_train_pdf, Y_train_pdf)

    # Log model and push to registry
    signature = infer_signature(X_train_pdf, Y_train_pdf)
    mlflow.sklearn.log_model(
        clf,
        name="decision_tree",
        signature=signature,
        registered_model_name=model_name
    )

    # Set model alias (i.e. Baseline)
    client.set_registered_model_alias(model_name, "Baseline", get_latest_model_version(model_name))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Use the Model for Inference 
# MAGIC
# MAGIC Now that our model is ready in model registry, we can use it for inference. In this section we will use the model for inference directly on a spark dataframe, which is called **batch inference**.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load the Model
# MAGIC
# MAGIC Loading a model from UC-based model registry is done by getting a model using **alias** and **version**. 
# MAGIC
# MAGIC After loading the model, we use **`mlflow.pyfunc.load_model`** to load it as a generic Python function. We then wrap it in a **`pandas_udf`** to enable distributed inference directly on a Spark DataFrame.

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

latest_model_version = client.get_model_version_by_alias(name=model_name, alias="baseline").version
model_uri = f"models:/{model_name}/{latest_model_version}" # Should be version 1
# model_uri = f"models:/{model_name}@baseline" # uri can also point to @alias

# Load model as pyfunc 
loaded_model = mlflow.pyfunc.load_model(model_uri)

@pandas_udf(StringType())
def predict_func(*cols):
    input_df = pd.DataFrame({name: col for name, col in zip(features, cols)})
    return pd.Series(loaded_model.predict(input_df).astype(str))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference
# MAGIC
# MAGIC Next, we will simply use the created function for inference.

# COMMAND ----------

# prepare test dataset
test_features_df = test_df.select(primary_key, *features)

# make prediction
prediction_df = test_features_df.withColumn("prediction", predict_func(*test_features_df.drop(primary_key).columns))

display(prediction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Deployment - With Feature Store 
# MAGIC
# MAGIC In the previous section we trained and registered a model using Spark dataframe. In some cases, you will need to use features from a feature store for training and inference. 
# MAGIC
# MAGIC In this section we will demonstrate how to train and deploy a model using Feature Store.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Feature Table
# MAGIC
# MAGIC Let's create a feature table based on the `features_df` that we created before. Please note that we will be using **Feature Store with Unity Catalog**, which means we need to use **`FeatureEngineeringClient`**.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# prepare feature set
features_df_all = telco_df.select(primary_key, *features)

# feature table definition
fe = FeatureEngineeringClient()
feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"

#drop table if exists
try:
    fe.drop_table(name=feature_table_name)
except:
    pass

# Create feature table
fe.create_table(
    name=feature_table_name,
    df=features_df_all,
    primary_keys=[primary_key],
    description="Example feature table"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Feature Lookups
# MAGIC
# MAGIC In order to create a training set from the feature table, we need to define a *feature lookup*. This will be used for creating training set from the feature table. 
# MAGIC
# MAGIC Note that the **`lookup_key`** is used for matching records in feature table.

# COMMAND ----------

# Create training set based on feature lookup
from databricks.feature_engineering import FeatureLookup

fl_handle = FeatureLookup(
    table_name=feature_table_name,
    lookup_key=[primary_key]
)

training_set_spec = fe.create_training_set(
    df=response_df,
    label=response,
    feature_lookups=[fl_handle],
    exclude_columns=[primary_key]
)

# Load training dataframe based on defined feature-lookup specification
training_df = training_set_spec.load_df()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and Register a Model with UC using Feature Table
# MAGIC
# MAGIC After creating the training set, **model training and registering is the same as the previous step**.

# COMMAND ----------

import warnings
from mlflow.types.utils import _infer_schema
    
    
# Convert data to pandas dataframes
X_train_pdf2 = training_df.drop(primary_key, response).toPandas()
Y_train_pdf2 = training_df.select(response).toPandas()
clf2 = DecisionTreeClassifier(max_depth=3, random_state=42)


with mlflow.start_run(run_name="Model-Batch-Deployment-Demo-With-FS") as mlflow_run:

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        log_post_training_metrics=True,
        silent=True)
    
    clf2.fit(X_train_pdf2, Y_train_pdf2)

    # Infer output schema
    try:
      output_schema = _infer_schema(Y_train_pdf2)
    except Exception as e:
      warnings.warn(f"Could not infer model output schema: {e}")
      output_schema = None
    
    # Log using feature engineering client and push to registry
    fe.log_model(
        model=clf2,
        artifact_path="decision_tree",
        flavor=mlflow.sklearn,
        training_set=training_set_spec,
        output_schema=output_schema,
        registered_model_name=model_name
    )

    # Set model alias (i.e. Champion)
    client.set_registered_model_alias(model_name, "Champion", get_latest_model_version(model_name))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Use the Model for Inference
# MAGIC
# MAGIC Inference for models that are registered with a Feature Store table are different than inference with a Spark DataFrame. We **join the lookup DataFrame with the feature table** on `customerID` to reconstruct the full feature set, then use **`mlflow.pyfunc.load_model`** wrapped in a **`pandas_udf`** to run distributed batch inference on the enriched DataFrame.
# MAGIC
# MAGIC > **How does the model know which features to use?** The feature table is joined explicitly on the primary key, ensuring each record is enriched with the correct feature values before being passed to the model.
# MAGIC

# COMMAND ----------

champion_model_uri = f"models:/{model_name}@champion"

# COMMAND ----------

# prepare lookup dataset
lookup_df = test_df.select("customerID")

# Retrieve features from feature table (replicates score_batch feature lookup)
feature_table_df = spark.table(feature_table_name)
lookup_with_features_df = lookup_df.join(feature_table_df, on="customerID", how="left")

# clf2 is the trained sklearn model in memory — picklable and works as a UDF closure variable
@pandas_udf(StringType())
def predict_champion_func(*cols):
    import pandas as pd
    input_df = pd.DataFrame({name: col for name, col in zip(features, cols)})
    return pd.Series(clf2.predict(input_df).astype(str))

# predict in batch using feature-enriched lookup df
prediction_fe_df = lookup_with_features_df.withColumn(
    "prediction",
    predict_champion_func(*[lookup_with_features_df[f] for f in features])
)

# COMMAND ----------

# MAGIC %md
# MAGIC Join with the `test_df` DataFrame to compare the `prediction` and `churn` columns. Remember, we are _predicting_ churn in this scenario.

# COMMAND ----------

# Join prediction_fe_df and test_df on customerID and only keep columns from prediction_fe_df
prediction_fe_df = prediction_fe_df.join(test_df.select(["customerID","Churn"]), on = "customerID", how = "left")
display(prediction_fe_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Considerations
# MAGIC
# MAGIC There are many possible (write) optimizations that Delta Lake can offer such as:
# MAGIC - **Partitioning:** stores data associated with different categorical values in different directories.
# MAGIC - **Z-Ordering (Legacy):** colocates related information in the same set of files.
# MAGIC - **Liquid Clustering (Recommended):** replaces both above-mentioned  methods to simplify data layout decisions and optimize query performance.
# MAGIC - **Predictive Optimizations:** removes the need to manually manage maintenance operations for Delta tables on Databricks.
# MAGIC
# MAGIC In this demo, we will show the last two options; liquid clustering and predictive optimization.

# COMMAND ----------

spark.sql(f"USE CATALOG {DA.catalog_name}")
spark.sql(f"USE SCHEMA {DA.schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Enable Predictive Optimization** at the schema level (can also be done at catalog level) is automatically enabled as a part of the Workspace setup and you do not have the proper permissions to enable it or disable it at the schema level as a user of this demo. However, the code to do so is provided here for completeness: 
# MAGIC
# MAGIC ```spark.sql(f"ALTER SCHEMA {DA.catalog_name}.{DA.schema_name} ENABLE PREDICTIVE OPTIMIZATION;")```

# COMMAND ----------

# MAGIC %md
# MAGIC Create inference table (where batch scoring jobs would be materialized) and enable liquid clustering on using `CLUSTER BY`

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE batch_inference(
# MAGIC   customerID STRING
# MAGIC  ,Churn STRING
# MAGIC  ,SeniorCitizen DOUBLE
# MAGIC  ,tenure DOUBLE
# MAGIC  ,MonthlyCharges DOUBLE
# MAGIC  ,TotalCharges DOUBLE
# MAGIC  ,prediction STRING)
# MAGIC CLUSTER BY (customerID, tenure)

# COMMAND ----------

(
  prediction_fe_df.write
  .mode("append")
  .option("mergeSchema", True)
  .saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.batch_inference")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Manually optimize table

# COMMAND ----------

# MAGIC %sql
# MAGIC ANALYZE TABLE batch_inference COMPUTE STATISTICS FOR ALL COLUMNS;
# MAGIC OPTIMIZE batch_inference

# COMMAND ----------

# MAGIC %md
# MAGIC Review the `batch_inference` table.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC * EXCEPT(Churn),
# MAGIC Churn
# MAGIC FROM batch_inference

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we presented two main batch deployment methods using MLflow for model tracking and logging with Unity Catalog. In the first approach, we trained and registered a model without a feature table, loading it with `mlflow.pyfunc.load_model` and wrapping it in a `pandas_udf` for distributed batch inference. The second method involved training a model with a feature table, registering it in the model registry, and joining the feature table on the primary key to retrieve features before running inference with the same `pandas_udf` pattern.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>