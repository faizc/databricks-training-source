# Databricks notebook source
# MAGIC %pip install -U mlflow importlib-metadata cloudpickle zipp
# MAGIC %pip install --ignore-installed Jinja2 markupsafe

# COMMAND ----------

# MAGIC %md
# MAGIC **🚨 Warning: Please don't run this notebook directly. This notebook must be used when creating the pipeline. Follow the instructions listed in the "3.1.a - Pipeline Deployment" notebook.**
# MAGIC
# MAGIC **🚨 Warning:** For this notebook to successfully run, you must have;
# MAGIC * Trained and logged a model to the registry (e.g. `ml_model`)
# MAGIC * Set the `catalog` and `schema` to point to your own. 
# MAGIC * Create pipeline parameters for input data path and model name (e.g. `mlpipeline.bronze_dataset_path`& `mlpipeline.model_name`)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Inference Pipeline
# MAGIC
# MAGIC  MLflow-trained models can be used in Lakeflow Spark Declarative Pipelines (SDP, formerly Delta Live Tables). MLflow models are treated as transformations in Databricks, meaning they act upon a Spark DataFrame input and return results as a Spark DataFrame. Because Lakeflow SDP defines datasets against DataFrames, you can convert Apache Spark workloads that leverage MLflow to a Lakeflow pipeline with just a few lines of code.
# MAGIC
# MAGIC If you already have a Python notebook calling an MLflow model, you can adapt the code to Lakeflow SDP by using the `@dp.materialized_view` decorator (for batch sources) and ensuring functions are defined to return transformation results. For an introduction to Lakeflow SDP syntax, see the Develop a pipeline with Python tutorial ([AWS](https://docs.databricks.com/aws/en/dlt/tutorial-pipelines) | [Azure](https://learn.microsoft.com/azure/databricks/dlt/tutorial-pipelines) | [GCP](https://docs.databricks.com/gcp/en/dlt/tutorial-pipelines)).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Pipeline configs

# COMMAND ----------

bronze_dataset_path = spark.conf.get("mlpipeline.bronze_dataset_path")
model_name = spark.conf.get("mlpipeline.model_name")
alias_name = spark.conf.get("mlpipeline.model_alias")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference configs

# COMMAND ----------

import mlflow


mlflow.set_registry_uri("databricks-uc")
model_uri=f"models:/{model_name}@{alias_name}" 
loaded_model_udf = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri=model_uri, 
    result_type="string",
    env_manager = "local"
    )

primary_key = "customerID"
features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference code

# COMMAND ----------

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.functions import col, struct

@dp.materialized_view(
  name="raw_inputs",
  comment="Raw inputs table",
  table_properties={"quality": "bronze"}
)
def raw_inputs():
  return (
    spark.read.csv(
      bronze_dataset_path,
      inferSchema=True,
      header=True,
      multiLine=True,
      escape='"'
    )
  )

@dp.materialized_view(
  name="features_input",
  comment="Features table",
  table_properties={"quality": "silver"}
)
def features_input():
  return (
    spark.read.table("raw_inputs")
      .select(primary_key, *features)
      .withColumn("SeniorCitizen", col("SeniorCitizen").cast("double"))
      .withColumn("tenure", col("tenure").cast("double"))
      .withColumn("TotalCharges", F.expr("try_cast(trim(TotalCharges) as double)"))
      .na.drop(how="any")
  )

@dp.materialized_view(
  name="model_predictions",
  comment="Inference table",
  table_properties={"quality": "gold"}
)
def model_predictions():
  return (
    spark.read.table("features_input")
      # pack feature columns into a struct so the UDF receives a single argument
      .withColumn("prediction", loaded_model_udf(struct(*[col(c) for c in features])))
  )