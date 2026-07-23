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
# MAGIC # LAB - Model Development Tracking with **MLflow**
# MAGIC
# MAGIC In this lab, you will learn how to leverage MLflow to track and manage your model development process. First, you'll load data from a feature table and create train/test splits. Next, you'll train a classification model and track its training process using MLflow. Although MLflow supports autologging of metadata and artifacts, you will manually log parameters and artifacts to gain hands-on experience with its logging API.
# MAGIC
# MAGIC 📌 **Note:** In previous demos, we covered both supervised and unsupervised model training. In this lab, however, **you will focus exclusively on fitting and tracking a supervised model.**
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC - **Task 1:** Load the dataset from the feature store table.
# MAGIC
# MAGIC - **Task 2:** Define the model hyperparameters.
# MAGIC
# MAGIC - **Task 3:** Track the model using MLflow.
# MAGIC
# MAGIC - **Task 4:** Log a custom figure.
# MAGIC
# MAGIC - **Task 5:** Review the model details via the MLflow Experiment runs UI.
# MAGIC
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
# MAGIC <li><strong>Serverless Compute, Version 5</strong> — <a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976D2; text-decoration: underline;">How to select an environment version</a></li>
# MAGIC </ul>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V5</strong>. Other compute options may work but are not guaranteed to behave the same or support all features demonstrated.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;"><div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">  <img src="../Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">  <div style="font-size: 15pt; color: #0B2026; line-height: 1.7; margin-bottom: 16px;">    Starting the MLflow tracking lab? Brush up on manual logging before you begin. Click on the genie icon <img src="../Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and ask Genie Code. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.  </div>  <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0B2026;">    <span id="genie-query-1-3" style="flex: 1;">I am completing an MLflow tracking lab on a telco customer churn dataset using a Decision Tree classifier. I need to manually log model hyperparameters, training and test metrics (accuracy, precision, recall, F1), a confusion matrix figure, and then register the model to Unity Catalog. What are the key MLflow API calls I need for each step — log_params, log_metrics, log_figure, log_model, and register_model — and how do I structure them inside a single mlflow.start_run() context?</span>    <button onclick="      var text = document.getElementById('genie-query-1-3').innerText;      var ta = document.createElement('textarea');      ta.value = text;      ta.style.position = 'fixed';      ta.style.opacity = '0';      document.body.appendChild(ta);      ta.select();      document.execCommand('copy');      document.body.removeChild(ta);      this.innerText = 'Copied!';      var btn = this;      setTimeout(function(){ btn.innerText = 'Copy'; }, 2000);    " style="background: #FF5F46; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-size: 13pt; cursor: pointer; white-space: nowrap;">Copy</button>  </div></div></div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classroom Setup
# MAGIC
# MAGIC Run the following cell to configure your working environment for this course.
# MAGIC
# MAGIC This setup will:
# MAGIC - Initialize the `DA` object (Databricks Academy helper)
# MAGIC - Configure your **default catalog** and **schema**
# MAGIC - Provision any supporting configuration needed for this lab
# MAGIC
# MAGIC **NOTE:** The `DA` object is only available in Databricks Academy courses.

# COMMAND ----------

# MAGIC %md
# MAGIC Next, this script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-1.3

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
# MAGIC ## Task 1 - Load Dataset from Feature Store Table
# MAGIC
# MAGIC Use the feature store to load a dataset from a specific table.
# MAGIC    - **Load Dataset:** Utilize MLflow's `load_delta` function to seamlessly retrieve and load the dataset from the Feature Store table named **`"telco"`** in the specified catalog and schema `("DA.catalog_name" and "DA.schema_name")`. 
# MAGIC    - Convert dataset to `pandas` dataframe and explore the loaded dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

## Import the necessary library for MLflow
import mlflow

## Load the feature dataset using mlflow.data
feature_dataset = mlflow.<FILL_IN>(
    table_name=<FILL_IN>,
    name=<FILL_IN>
)

## convert the dataset to pandas df and drop the customerID column
feature_data_pd = <FILL_IN>

## Convert all feature_data_pd columns to float
feature_data_pd = feature_data_pd.astype(float)

## inspect final dataset
display(<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 1. Load Data
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ## Import the necessary library for MLflow
# MAGIC import mlflow
# MAGIC
# MAGIC ## Load the feature dataset using mlflow.data
# MAGIC feature_dataset = mlflow.data.load_delta(
# MAGIC     table_name=f"{DA.catalog_name}.{DA.schema_name}.telco",
# MAGIC     name="telco"
# MAGIC )
# MAGIC
# MAGIC ## convert the dataset to pandas df and drop the customerID column
# MAGIC feature_data_pd = feature_dataset.df.drop("customerID").toPandas()
# MAGIC
# MAGIC ## Convert all feature_data_pd columns to float
# MAGIC feature_data_pd = feature_data_pd.astype(float)
# MAGIC
# MAGIC ## inspect final dataset
# MAGIC display(feature_data_pd)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split
# MAGIC
# MAGIC Split the dataset into training and testing sets.

# COMMAND ----------

## Import necessary libraries
import mlflow.sklearn  # For MLflow integration
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets

## Split the dataset into training and testing sets
target_col = <FILL_IN>
X_all = <FILL_IN>
y_all = <FILL_IN>
X_train, X_test, y_train, y_test = <FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 1. Train / Test Split
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ## Import necessary libraries
# MAGIC import mlflow.sklearn  # For MLflow integration
# MAGIC from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
# MAGIC
# MAGIC ## Split the dataset into training and testing sets
# MAGIC target_col = "Churn"
# MAGIC X_all = feature_data_pd.drop(labels=target_col, axis=1)
# MAGIC y_all = feature_data_pd[target_col]
# MAGIC X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.8, random_state=42)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 - Define Model Hyperparameters
# MAGIC
# MAGIC In this lab, you will train a classification model. In this task define parameters for a Decision Tree Model.

# COMMAND ----------

## Define Decision Tree Classifier parameters
dtc_params = {
  'criterion': <FILL_IN>,
  'max_depth': <FILL_IN>,
  'min_samples_split': <FILL_IN>,
  'min_samples_leaf': <FILL_IN>
}

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 2. Define Model Hyperparameters
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ## Define Decision Tree Classifier parameters
# MAGIC dtc_params = {
# MAGIC     'criterion': 'gini',
# MAGIC     'max_depth': 50,
# MAGIC     'min_samples_split': 20,
# MAGIC     'min_samples_leaf': 5
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 3 - Track the Model Development  with MLflow
# MAGIC
# MAGIC Initialize an MLflow run.
# MAGIC    - **Initialize MLflow Run:** Start an MLflow run to track the model development process. This allows for systematic recording of parameters, metrics, and artifacts associated with the model.
# MAGIC
# MAGIC    - **Logging Model Details:** Utilize MLflow tracking to log essential information about the model, including parameters, metrics, and other relevant artifacts.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from mlflow.models.signature import infer_signature

## set the path for mlflow experiment
mlflow.set_experiment(f"/Users/{DA.username}/LAB-1-Model-Development-Tracking-with-MLflow")

## Turn off autologging as we want to log the model manually
mlflow.autolog(disable=True)

## Start an MLFlow run
with mlflow.start_run(run_name="Model Developing Tracking with MLflow Lab") as run:
   # Log the dataset
   mlflow.log_input(feature_dataset, context="source")
   mlflow.log_input(mlflow.data.from_pandas(<FILL_IN>, source=feature_dataset.source), context="training")
   mlflow.log_input(mlflow.data.from_pandas(<FILL_IN> source=feature_dataset.source), context="test")

   ## Log parameters
   mlflow.log_params(<FILL_IN>)

   ## Fit the model
   dtc = DecisionTreeClassifier(<FILL_IN>)
   dtc_mdl = dtc.fit(<FILL_IN>)

   ## Define model signature
   signature = infer_signature(X_all, y_all)
    
   ## Log the model
   ## Define the model name based on the feature store catalog and schema
   model_name = f"{DA.catalog_name}.{DA.schema_name}.churnmodel"
   mlflow.sklearn.log_model(
       sk_model=<FILL_IN>,
       artifact_path="model-artifacts",
       signature=<FILL_IN>,
       registered_model_name=<FILL_IN>
   )

   ## Evaluate on the training set
   y_pred_train = <FILL_IN>
   mlflow.log_metric("train_accuracy", <FILL_IN>)
   mlflow.log_metric("train_precision", <FILL_IN>)
   mlflow.log_metric("train_recall", <FILL_IN>)
   mlflow.log_metric("train_f1", <FILL_IN>)

   ## Evaluate on the test set
   y_pred_test = dtc_mdl.predict(X_test)
   mlflow.log_metric("test_accuracy", <FILL_IN>)
   mlflow.log_metric("test_precision", <FILL_IN>)
   mlflow.log_metric("test_recall", <FILL_IN>)
   mlflow.log_metric("test_f1", <FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 3. Track Model Development with MLflow
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
# MAGIC from sklearn.tree import DecisionTreeClassifier
# MAGIC
# MAGIC from mlflow.models.signature import infer_signature
# MAGIC
# MAGIC ## set the path for mlflow experiment
# MAGIC mlflow.set_experiment(f"/Users/{DA.username}/LAB-1-Model-Development-Tracking-with-MLflow")
# MAGIC
# MAGIC ## Turn off autologging as we want to log the model manually
# MAGIC mlflow.autolog(disable=True)
# MAGIC
# MAGIC ## Start an MLFlow run
# MAGIC with mlflow.start_run(run_name="Model Developing Tracking with MLflow Lab") as run:
# MAGIC     # Log the dataset
# MAGIC     mlflow.log_input(feature_dataset, context="source")
# MAGIC     mlflow.log_input(mlflow.data.from_pandas(X_train, source=feature_dataset.source), context="training")
# MAGIC     mlflow.log_input(mlflow.data.from_pandas(X_test, source=feature_dataset.source), context="test")
# MAGIC
# MAGIC     ## Log parameters
# MAGIC     mlflow.log_params(dtc_params)
# MAGIC
# MAGIC     ## Fit the model
# MAGIC     dtc = DecisionTreeClassifier(**dtc_params)
# MAGIC     dtc_mdl = dtc.fit(X_train, y_train)
# MAGIC
# MAGIC     ## Define model signature
# MAGIC     signature = infer_signature(X_all, y_all)
# MAGIC
# MAGIC     ## Log the model
# MAGIC     ## Define the model name based on the feature store catalog and schema
# MAGIC     model_name = f"{DA.catalog_name}.{DA.schema_name}.churnmodel"
# MAGIC     mlflow.sklearn.log_model(
# MAGIC         sk_model=dtc_mdl,
# MAGIC         artifact_path="model-artifacts",
# MAGIC         signature=signature,
# MAGIC         registered_model_name=model_name
# MAGIC     )
# MAGIC
# MAGIC     ## Evaluate on the training set
# MAGIC     y_pred_train = dtc_mdl.predict(X_train)
# MAGIC     mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_pred_train))
# MAGIC     mlflow.log_metric("train_precision", precision_score(y_train, y_pred_train))
# MAGIC     mlflow.log_metric("train_recall", recall_score(y_train, y_pred_train))
# MAGIC     mlflow.log_metric("train_f1", f1_score(y_train, y_pred_train))
# MAGIC
# MAGIC     ## Evaluate on the test set
# MAGIC     y_pred_test = dtc_mdl.predict(X_test)
# MAGIC     mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
# MAGIC     mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
# MAGIC     mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
# MAGIC     mlflow.log_metric("test_f1", f1_score(y_test, y_pred_test))
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4 - Log Custom Figure
# MAGIC
# MAGIC **Log Custom Figure/Visualization:** Include the logging of a custom figure, such as a confusion matrix or any relevant visualization, to further illustrate the model's behavior. This visual representation can be valuable for model evaluation and interpretation.

# COMMAND ----------

## Import necessary libraries for creating and displaying a confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mlflow.client import MlflowClient
client = MlflowClient()

## Compute the confusion matrix
cm = confusion_matrix(<FILL_IN>)

## Create a figure object and axes for the confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))

## Create a ConfusionMatrixDisplay object with the computed confusion matrix
disp = ConfusionMatrixDisplay(<FILL_IN>)

## Plot the confusion matrix using the created axes and specified color map
disp.plot(<FILL_IN>)

## Set the title of the plot
ax.set_title('Confusion Matrix')

## Log the confusion matrix figure to MLflow
client.log_figure(<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Task 4. Log Custom Figure
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ## Import necessary libraries for creating and displaying a confusion matrix
# MAGIC import matplotlib.pyplot as plt
# MAGIC from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# MAGIC
# MAGIC from mlflow.client import MlflowClient
# MAGIC client = MlflowClient()
# MAGIC
# MAGIC ## Compute the confusion matrix
# MAGIC cm = confusion_matrix(y_test, y_pred_test, labels=[1, 0])
# MAGIC
# MAGIC ## Create a figure object and axes for the confusion matrix plot
# MAGIC fig, ax = plt.subplots(figsize=(8, 6))
# MAGIC
# MAGIC ## Create a ConfusionMatrixDisplay object with the computed confusion matrix
# MAGIC disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
# MAGIC
# MAGIC ## Plot the confusion matrix using the created axes and specified color map
# MAGIC disp.plot(cmap=plt.cm.Blues, ax=ax)
# MAGIC
# MAGIC ## Set the title of the plot
# MAGIC ax.set_title('Confusion Matrix')
# MAGIC
# MAGIC ## Log the confusion matrix figure to MLflow
# MAGIC client.log_figure(run.info.run_id, figure=fig, artifact_file="confusion_matrix.png")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5 - Review model details via the UI
# MAGIC To review the model details via the MLflow UI in the Experiment runs, follow these steps:
# MAGIC
# MAGIC + Step 1: Go to the "Experiments" Section
# MAGIC
# MAGIC + Step 2: Locate Your Experiment
# MAGIC
# MAGIC + Step 3: Review Run Details
# MAGIC
# MAGIC + Step 4: Reviewing Artifacts and Model metrics
# MAGIC
# MAGIC + Step 5: Viewing Confusion Matrix Image
# MAGIC
# MAGIC + Step 6: Retrieve Model Details

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC
# MAGIC In conclusion, this lab showcased the effectiveness of MLflow in seamlessly managing the model development process. Leveraging MLflow's features, such as experiment tracking, custom metric logging, and artifact storage, enhances collaboration and ensures reproducibility. The ability to review model details through the MLflow UI provides valuable insights into model performance and aids in making informed decisions.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>