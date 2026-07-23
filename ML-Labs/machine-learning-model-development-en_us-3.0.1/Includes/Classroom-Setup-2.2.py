# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

def clear_spark_ml_cache():
    """
    Clears all ML model references held in the current Python session
    so Spark Connect can release them from its in-memory cache.
    Useful when hitting ML_CACHE_SIZE_OVERFLOW_EXCEPTION.
    """
    import gc
    import mlflow
    from pyspark.ml import PipelineModel
 
    # Disable autologging to prevent MLflow from holding model refs
    mlflow.autolog(disable=True)
 
    # Collect all PipelineModel / Transformer objects in the global scope and delete them
    to_delete = [
        name for name, obj in globals().items()
        if isinstance(obj, PipelineModel)
    ]
    for name in to_delete:
        print(f"Deleting cached model reference: {name}")
        del globals()[name]
 
    # Force Python garbage collection
    gc.collect()
    print("ML cache references cleared.")
 
# Call this BEFORE DA.init() if you are re-running the notebook
clear_spark_ml_cache()

# COMMAND ----------

@DBAcademyHelper.add_init
def create_features_table(self):
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    table_name = "customer_churn"
    shared_volume_name = "telco"
    csv_name = "telco-customer-churn"

    full_table_path = f"{DA.catalog_name}.{DA.schema_name}.{table_name}"
    dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv"

    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE SCHEMA {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    telco_df = (
        spark.read.format("csv").option("header", True).load(dataset_path)
    )

    telco_df = telco_df.select(
        "gender", "SeniorCitizen", "Partner", "tenure", "InternetService",
        "Contract", "PaperlessBilling", "PaymentMethod", "TotalCharges", "Churn"
    )

    telco_df = (
        telco_df
        .withColumn("Gender", F.col("gender"))
        .withColumn("Tenure", F.col("tenure").cast("double"))
        .withColumn("TotalCharges", F.expr("try_cast(trim(TotalCharges) as double)"))
        .fillna(0)
    )

    # Label-encode every string column WITHOUT Spark ML
    categorical_cols = [c for c, t in telco_df.dtypes if t == "string"]
    for c in categorical_cols:
        window = Window.orderBy(c)
        telco_df = telco_df.withColumn(
            f"{c}_encoded", (F.dense_rank().over(window) - 1).cast("double")
        )

    telco_df.write.format("delta") \
        .option("overwriteSchema", "true") \
        .mode("overwrite") \
        .saveAsTable(table_name)

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()