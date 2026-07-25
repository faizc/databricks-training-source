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
# MAGIC # LAB - Data Cleaning and Feature Engineering
# MAGIC
# MAGIC In this lab, you will clean the telco customer churn data and perform feature engineering. This is the first step in the MLOps workflow, preparing the data for model training.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC _This lab contains the following tasks:_
# MAGIC - **Task 1:** Load and explore the raw dataset
# MAGIC - **Task 2:** Clean the dataset and handle missing values
# MAGIC - **Task 3:** Engineer features for model training
# MAGIC - **Task 4:** Create and load a feature table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **17.3.x-cpu-ml-scala2.13**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup-1.1a

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this lab, we'll refer to the object DA. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Load and Explore the Raw Dataset
# MAGIC In this step, you will load the raw telco customer churn dataset and display its contents. This will help you understand the structure and content of the data.
# MAGIC
# MAGIC **Instructions:**
# MAGIC > - Load the raw telco customer churn dataset from a CSV file.
# MAGIC > - Display the dataset to understand its structure and content.

# COMMAND ----------

# Define the path to the dataset using an f-string for dynamic path construction
dataset_path = f"{DA.paths.datasets.telco}/telco/telco-customer-churn-missing.csv"

# Read the dataset into a Spark DataFrame. The 'csv' format and 'true' header option are specified
telcoDF = spark.read.format("csv").option("header", "true").load(dataset_path)

# Display the DataFrame
display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Define data cleaning and feature engineering logic
# MAGIC In this step, you will define a function to clean the data and perform feature engineering. The function will:
# MAGIC - Convert specific columns to appropriate data types.
# MAGIC - Handle missing values by filling them with suitable default values.
# MAGIC - Create a new feature that counts the number of optional services a customer has subscribed to.

# COMMAND ----------

# MAGIC %md
# MAGIC **Instructions:**
# MAGIC > - Define the data cleaning and feature engineering function.

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

def clean_churn_features(dataDF: DataFrame) -> DataFrame:
    df = dataDF

    def _to_double(colname):
        c = F.col(colname)
        return F.when(F.trim(c) == "", None).otherwise(c).cast("double")

    df = (df
          .withColumn("tenure", _to_double("tenure"))
          .withColumn("MonthlyCharges", _to_double("MonthlyCharges"))
          .withColumn("TotalCharges", _to_double("TotalCharges"))
         )

    df = df.withColumn(
        "SeniorCitizen",
        F.expr("try_cast(SeniorCitizen as double)")
    ).withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))

    binary_cols = ["Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity",
                   "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                   "PaperlessBilling","Churn"]

    def yesno(col):
        return (F.when(F.col(col) == "Yes", 1)
                 .when(F.col(col).isin("No", "No internet service", "No phone service"), 0)
                 .otherwise(None).cast("int"))

    for c in binary_cols:
        if c in df.columns:
            df = df.withColumn(c, yesno(c))

    gender_map   = F.create_map(F.lit("Male"), F.lit(1), F.lit("Female"), F.lit(0))
    internet_map = F.create_map(F.lit("Fiber optic"), F.lit(1), F.lit("DSL"), F.lit(2),
                                F.lit("None"), F.lit(0), F.lit("No"), F.lit(0))
    contract_map = F.create_map(F.lit("Month-to-month"), F.lit(1),
                                F.lit("One year"), F.lit(2), F.lit("Two year"), F.lit(3))
    pay_map      = F.create_map(F.lit("Credit card (automatic)"), F.lit(1),
                                F.lit("Mailed check"), F.lit(2),
                                F.lit("Bank transfer (automatic)"), F.lit(3),
                                F.lit("Electronic check"), F.lit(4))

    if "gender" in df.columns:
        df = df.withColumn("gender", gender_map[F.col("gender")].cast("int"))
    if "InternetService" in df.columns:
        df = df.withColumn("InternetService", internet_map[F.col("InternetService")].cast("int"))
    if "Contract" in df.columns:
        df = df.withColumn("Contract", contract_map[F.col("Contract")].cast("int"))
    if "PaymentMethod" in df.columns:
        df = df.withColumn("PaymentMethod", pay_map[F.col("PaymentMethod")].cast("int"))

    optional_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                     "TechSupport","StreamingTV","StreamingMovies"]
    df = df.fillna({c: 0 for c in optional_cols})
    df = df.withColumn("num_optional_services", sum(F.col(c) for c in optional_cols))

    df = df.fillna({"tenure": 0.0, "MonthlyCharges": 0.0, "TotalCharges": 0.0})

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2.2: Clean the data
# MAGIC
# MAGIC In this step, you will apply the data cleaning and feature engineering function to the raw dataset. 
# MAGIC
# MAGIC This will produce a cleaned dataset with new features that are ready for model training.
# MAGIC
# MAGIC > - Apply the data cleaning and feature engineering function to the raw dataset.
# MAGIC > - Display the cleaned dataset to verify the changes.
# MAGIC

# COMMAND ----------

# Apply the data cleaning function
cleaned_data = clean_churn_features(telcoDF)

# Display the cleaned data
display(cleaned_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Save the cleaned and feature-engineered data
# MAGIC
# MAGIC In this step, you will save the cleaned and feature-engineered data as a Delta table. This table will be used in subsequent labs for model training and evaluation.
# MAGIC
# MAGIC **Instructions:**
# MAGIC > - Save the cleaned and feature-engineered data as a Delta table.
# MAGIC > - Print the path where the cleaned data is saved.
# MAGIC

# COMMAND ----------

# Specify the table name
cleaned_data_path = f"/Users/{DA.username}/cleaned_telco_data"

# Save the cleaned_data DataFrame as a Delta table
cleaned_data.write.mode("overwrite").format("delta").save(cleaned_data_path)

# Print a success message
print(f"Cleaned data saved to: {cleaned_data_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4: Create and Load Feature Table
# MAGIC In this final step, you will create a feature table from the cleaned data and load it for further analysis.
# MAGIC
# MAGIC **Instructions:**
# MAGIC > - Create a feature table from the cleaned data.
# MAGIC > - Load the feature table and display its contents.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# Initialize the FeatureEngineeringClient
fe = FeatureEngineeringClient()

# Load the data into a DataFrame
df = spark.read.format("delta").load(cleaned_data_path)

# Define the name of the feature table using dynamic path construction
table_name = f"{DA.catalog_name}.{DA.schema_name}.telco_cleaned_table"

# Create a feature table from the dataset
fe.create_table(
    name=table_name,
    primary_keys=["customerID"],
    df=df,
    description="Telco customer features",
    tags={"source": "bronze", "format": "delta"}
)

# Retrieve the feature table by name
ft = fe.get_table(name=table_name)

# Print the description of the feature table
print(f"Feature Table description: {ft.description}")

# Display the data from the feature table
display(fe.read_table(name=table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC In this lab, you learned how to load, clean, and perform feature engineering on a raw dataset. These prepared features will be used in subsequent labs for model training and evaluation.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>