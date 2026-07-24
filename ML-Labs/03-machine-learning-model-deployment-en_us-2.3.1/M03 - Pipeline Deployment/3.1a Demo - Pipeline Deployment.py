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
# MAGIC # Pipeline Deployment
# MAGIC
# MAGIC In this demo, we will show how to use a model as part of a data pipeline for inference. In the first section of the demo, we will prepare data and perform some basic feature engineering. Then, we will fit and register the model to model registry. Please note that these two steps are already covered in other courses and they are not the main focus of this demo. In the last section, which is the main focus of this demo, we will create a Lakeflow Spark Declarative Pipeline (SDP, formerly Delta Live Tables) and use the registered model as part of the pipeline.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Describe steps for deploying a model within a pipeline.
# MAGIC
# MAGIC * Develop a simple pipeline that performs batch inference in its final step.
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

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;"><div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">  <img src="../Includes/Images/genie-code.png" style="height: 64px; margin-bottom: 10px;">  <div style="font-size: 15pt; color: #0B2026; line-height: 1.7; margin-bottom: 16px;">    Want to know more about deploying models inside Lakeflow Spark Declarative Pipelines in Databricks? Ask Genie Code. Click on the genie icon <img src="../Includes/Images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.  </div>  <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0B2026;">    <span id="genie-query-3-1a" style="flex: 1;">How do I use a registered MLflow model for inference inside a Lakeflow Spark Declarative Pipeline (formerly Delta Live Tables)? How do streaming tables and materialized views work, and how do I apply a model as the final step of a pipeline to generate predictions?</span>    <button onclick="      var text = document.getElementById('genie-query-3-1a').innerText;      var ta = document.createElement('textarea');      ta.value = text;      ta.style.position = 'fixed';      ta.style.opacity = '0';      document.body.appendChild(ta);      ta.select();      document.execCommand('copy');      document.body.removeChild(ta);      this.innerText = 'Copied!';      var btn = this;      setTimeout(function(){ btn.innerText = 'Copy'; }, 2000);    " style="background: #FF5F46; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-size: 13pt; cursor: pointer; white-space: nowrap;">Copy</button>  </div></div></div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.1a

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

# Dataset specs
primary_key = "customerID"
response = "Churn"
features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] # Keeping numerical only for simplicity and demo purposes

# Read dataset (and drop nan)
telco_df = spark.read.csv(dataset_p_telco, inferSchema=True, header=True, multiLine=True, escape='"')\
            .withColumn("TotalCharges", F.expr("try_cast(trim(TotalCharges) as double)"))\
            .na.drop(how='any')

# Separate features and ground-truth
features_df = telco_df.select(primary_key, *features)
response_df = telco_df.select(primary_key, response)

# Train a sklearn Decision Tree Classification model
# Convert data to pandas dataframes
X_train_pdf = features_df.drop(primary_key).toPandas()
Y_train_pdf = response_df.drop(primary_key).toPandas()

for col in X_train_pdf.select_dtypes("int32"):
    X_train_pdf[col] = X_train_pdf[col].astype("double")

# COMMAND ----------

print(X_train_pdf.info())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Preparation
# MAGIC
# MAGIC **Note:** This section is not the main focus of this course. We are just repeating the model development and registration process here.

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


def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and Register a Model with UC

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
from mlflow.models import infer_signature

# Use 3-level namespace for model name
model_name = f"{DA.catalog_name}.{DA.schema_name}.model_3_1a_demo" 

alias_name = "pipeline"

# model to use for classification
clf = DecisionTreeClassifier(max_depth=4, random_state=10)

with mlflow.start_run(run_name="Model-Deployment demo") as mlflow_run:

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

    # Set model alias
    client.set_registered_model_alias(model_name, alias_name, get_latest_model_version(model_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Pipeline to Run Batch Inference
# MAGIC
# MAGIC Now that our model is registered and ready, we can move on the most important part; using the model for inference inside a pipeline. 
# MAGIC
# MAGIC **Note: The pipeline is already defined in `3.1b` notebook.**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Config Variables
# MAGIC
# MAGIC While defining the pipeline, you will need to use the following variables. Run the code block below first. Then, use the output in the next section while creating the pipeline.
# MAGIC

# COMMAND ----------

print(f"mlpipeline.bronze_dataset_path: {dataset_p_telco}")
print(f"mlpipeline.model_name: {model_name}")
print(f"mlpipeline.model_alias: {alias_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create the ETL Pipeline
# MAGIC
# MAGIC - This Vocareum environment has be configured so that **Lakeflow Spark Declarative Pipelines (SDP)** has been enabled.
# MAGIC - > ****Note:**** To enable the Lakeflow Pipelines Editor: Open your user settings, go to Developer, and enable ****Lakeflow Pipelines Editor****.
# MAGIC
# MAGIC ### Instructions
# MAGIC 1. Navigate to **Jobs & Pipelines** in the left sidebar. Click **Create**, select **ETL Pipeline**.
# MAGIC     - If you see a prompt saying **"Could not find a valid default catalog"**, manually select `dbacademy` as the catalog and your labuser schema (e.g., `labuserXXXXXXXX_XXXXXXXXXX`) to continue.
# MAGIC        - **Note** : If you are unable to find your lab user schema, deselect `dbacademy` and then select it again. You should then be able to see your lab user schema.
# MAGIC 2. At the top of the editor, give the pipeline the name `<labuserXXXXXXXX_XXXXXXXXXX>-pipeline`, where you replace `<labuserXXXXXXXX_XXXXXXXXXX>` with your labuser name.
# MAGIC     - Click on the profile icon at the top right to copy your labuser name or see the output to cell 8 above.
# MAGIC 3. Next to the pipeline name, click the **catalog and schema** display to change it. Make sure the catalog `dbacademy` is selected and select your labuser schema (e.g., `<labuserXXXXXXXX_XXXXXXXXXX>`).
# MAGIC 4. In the pipeline editor, expand the **⋮ menu** (to the right of the **Use sample code** button) and select **Add existing source code**.
# MAGIC </br>
# MAGIC <img src="../Includes/Images/etl-pipeline-1.png" width="500"/>
# MAGIC </br>
# MAGIC 6. In **Pipeline root folder**, locate and open the folder **M03 - Pipeline Deployment/Pipeline**. 
# MAGIC 7. In **Source code paths**, click on the folder icon and select **3.1b Demo - Inference Pipeline** and click **Select**. 
# MAGIC 8. Back in the **Add existing assets** select **Add** at the bottom right. 
# MAGIC 9. Click on the **Pipeline** menu item at the top left and select the notebook **3.1b Demo - Inference Pipeline**. 
# MAGIC 10. This new editor will display the notebook in the center of the screen and the **Pipeline graph** on the right of the screen. We will need to configure the variables shown in the notebook **3.1b Demo - Inference Pipeline** in the **Pipeline settings**. To do this, click on the **settings** icon next to **Pipeline configuration** to open the pipeline settings. Then, scroll down to the **Configuration** section and click **Add configuration** to set up the necessary variables for the pipeline.
# MAGIC </br>
# MAGIC <img src="../Includes/Images/add-config.png" width="500"/>
# MAGIC </br>
# MAGIC 11. The config variable values are defined in the section **Config Variables** in this notebook (**3.1a Demo - Pipeline Deployment**). Copy and paste the key-value pair into the configuration and click **Save**.
# MAGIC 12. Back in **Pipeline settings**, navigate to and click **Dry run** at the top right. 
# MAGIC     - Dry-run mode allows you to test your policy configuration and monitor outbound connections without disrupting access to resources. This will not create or update any tables. 
# MAGIC 13. Once the dry run is is completed, click **Run pipeline**. This will now create or update any tables in our pipeline. 
# MAGIC
# MAGIC > Note we did not use classic compute for this pipeline run. We left **Serverless** as our compute by default.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Resources and Trainings
# MAGIC This demo is not a comprehensive introduction to **Lakeflow Spark Declarative Pipelines**. For a deeper dive into this Databricks feature, check out our course **[Build Data Pipeline with Lakeflow Spark Declarative Pipelines](https://www.databricks.com/training/catalog?search=build+data+pipelines+with+lakeflow+spark+declarative+pipelines)**.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demonstration, we walked through the sequential process of training, registering, and deploying a model within a pipeline. Following the standard procedure of fitting and registering the model, we then established a Lakeflow Spark Declarative Pipeline. This pipeline not only ingests data from a source file but also implements necessary data transformations, culminating in the utilization of the registered model as the final step in the pipeline. While your specific project requirements may vary, this example illustrates how to set up and integrate a model for inference within a Lakeflow Spark Declarative Pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>