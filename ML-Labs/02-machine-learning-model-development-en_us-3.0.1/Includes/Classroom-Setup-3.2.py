# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

@DBAcademyHelper.add_init
def create_features_table(self):
    from pyspark.sql.functions import monotonically_increasing_id, col
    from databricks.feature_engineering import FeatureEngineeringClient
    
    table_name = "bank_loan"
    features_table_name = "bank_loan_features"

    shared_volume_name = "banking"
    csv_name = "loan-clean"

    # Full path to tables in Unity Catalog
    full_table_path = f"{DA.catalog_name}.{DA.schema_name}.{table_name}"
    features_table_path = f"{DA.catalog_name}.{DA.schema_name}.{features_table_name}"

    # Path to CSV file
    #dataset_path = f"{DA.paths.datasets.banking}/{shared_volume_name}/{csv_name}.csv"
    dataset_path = f"/Volumes/dbacademy_banking/v01/banking"
    # Define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(f"DROP TABLE IF EXISTS {features_table_name}")
    
    # Read the CSV file into a Spark DataFrame
    loan_df = (
        spark
        .read
        .format('csv')
        .option('header', True)
        .load(dataset_path)
        )

    # Select columns of interest and replace spaces with underscores
    loan_df_clean = loan_df.selectExpr("ID", "Age", "`ZIP Code` as ZIP_Code", "Family", "CCAvg", "Education", "Mortgage", "`Personal Loan` as Personal_Loan", "`Securities Account` as Securities_Account", "`CD Account` as CD_Account", "Online", "CreditCard")

    # Save df as delta table using Delta API
    loan_df_clean.write.format("delta").mode("overwrite").saveAsTable("bank_loan")

    # Create features table
    load_df_features = loan_df.select("ID", "Experience", "Income")
    fe = FeatureEngineeringClient()
    fe.create_table(
        name = features_table_path,
        primary_keys = ["ID"],
        df = load_df_features,
        description="Bank Loan Feature Table",
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
    content = f"""You are assisting with a binary classification task to predict bank personal loan acceptance.

Dataset: bank_loan (joined with bank_loan_features via ID)
Join key: ID (left join from bank_loan to bank_loan_features)
Target column: Personal_Loan (1 = accepted, 0 = declined) — already encoded as binary
Catalog: {DA.catalog_name}
Schema: {DA.schema_name}
MLflow experiment: /Users/{DA.username}/loan_model_comparison
Modeling library: scikit-learn (Pipeline-based preprocessing preferred)
All experiments must be logged to MLflow with explicit parameter and metric logging.
Register final models to Unity Catalog using mlflow.set_registry_uri("databricks-uc").
Lab tasks: business-driven feature engineering, multi-model comparison (Logistic Regression vs Random Forest), decision threshold optimization using precision-recall curve, and deployment to a real-time Databricks serving endpoint.
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