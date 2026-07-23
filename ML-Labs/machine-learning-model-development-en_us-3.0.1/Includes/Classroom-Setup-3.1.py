# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

@DBAcademyHelper.add_init
def create_features_table(self):
    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql.functions import monotonically_increasing_id, col

    table_name = 'customer_churn'
    features_table_name = "customer_churn_features"

    shared_volume_name = "telco"
    csv_name = "telco-customer-churn"

    # Full path to tables in Unity Catalog
    full_table_path = f"{DA.catalog_name}.{DA.schema_name}.{table_name}"
    features_table_path = f"{DA.catalog_name}.{DA.schema_name}.{features_table_name}"

    # Path to CSV file
    dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv"

    # Define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(f"DROP TABLE IF EXISTS {features_table_name}")
    
    # Read the CSV file into a Spark DataFrame
    telco_df = (
        spark
        .read
        .format('csv')
        .option('header', True)
        .load(dataset_path)
        )

    # basic clean-up
    telco_df = telco_df.withColumn("CustomerID",  monotonically_increasing_id())
    telco_df = telco_df.withColumn("Gender", col("gender"))
    telco_df = telco_df.withColumn("Tenure", col("tenure").cast("double"))
    telco_df = telco_df.withColumn("TotalCharges", F.expr("try_cast(trim(TotalCharges) as double)"))
    telco_df = telco_df.withColumn("AverageMonthlyCharges", col("TotalCharges")/col("Tenure"))

    # Select columns of interest
    telco_df_clean = telco_df.select("CustomerID", "Gender", "SeniorCitizen", "Partner", "InternetService", "Contract", "PaperlessBilling", "PaymentMethod", "Churn")

    # save df as delta table
    telco_df_clean.write.format("delta").option("overwriteSchema", 'true').mode("overwrite").saveAsTable("customer_churn")

    # Create features table
    df_features = telco_df.select("CustomerID", "AverageMonthlyCharges")
    fe = FeatureEngineeringClient()
    fe.create_table(
        name=features_table_path,
        primary_keys=["CustomerID"],
        df=df_features,
        description="Customer Churn Feature Table",
        tags={"source": "gold", "format": "delta"}
    )

# COMMAND ----------

@DBAcademyHelper.add_init
def create_assistant_instructions(self):
    """
    Writes .assistant_instructions.md to the workspace folder containing this notebook.
    Genie Code reads this file automatically to stay project-aware across all prompts.
    Uses overwrite=True so re-running setup is always safe (create or replace).
    """
    content = f"""You are assisting with a binary classification task to predict customer churn.

Dataset: customer_churn (joined with customer_churn_features via CustomerID)
Join key: CustomerID (left join from customer_churn to customer_churn_features)
Target column: Churn (Yes = churned, No = retained) — encode as 1/0
Catalog: {DA.catalog_name}
Schema: {DA.schema_name}
MLflow experiment: /Users/{DA.username}/churn_prediction_genie
Modeling library: scikit-learn (Pipeline-based preprocessing preferred)
All experiments must be logged to MLflow with explicit parameter and metric logging.
Register final models to Unity Catalog using mlflow.set_registry_uri("databricks-uc").
"""

    try:
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        folder_path = "/".join(notebook_path.split("/")[:-1])
        file_path = f"/Workspace{folder_path}/.assistant_instructions.md"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ .assistant_instructions.md written to: {file_path}")
        print("   Genie Code will load this automatically when you open the Genie panel.")
    except Exception as e:
        print(f"⚠️  Could not write .assistant_instructions.md: {e}")
        print("   You can manually paste the custom instructions into the Genie Code sidebar.")

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper()
DA.init()