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
# MAGIC # LAB - Hyperparameter Tuning with Optuna
# MAGIC
# MAGIC Welcome to the Hyperparameter Tuning with Optuna lab! In this hands-on session, you'll gain practical insights into **optimizing machine learning models using Optuna**. Throughout the lab, we'll cover key steps, from loading the dataset and creating training/test sets to **defining a hyperparameter search space and running optimization trials with Spark**. The primary objective is to equip you with the skills to fine-tune models effectively using Spark, Optuna, and MLflow.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC 1. Load the dataset and create training/test sets for a scikit-learn model. 
# MAGIC 1. Define the hyperparameter search space for optimization.
# MAGIC 1. Define the optimization function to fine-tune the model.
# MAGIC 1. Run hyperparameter tuning trials. 
# MAGIC 1. Search for runs using the MLflow API and visualize all runs within the MLflow experiment.
# MAGIC 1. Identify the best run based on the model's precision value programmatically and visually.
# MAGIC 1. Register the model with Unity Catalog.

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
# MAGIC <div style="width: 100%; font-family: sans-serif;"><div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">  <img src="../Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">  <div style="font-size: 15pt; color: #0B2026; line-height: 1.7; margin-bottom: 16px;">    New to Optuna hyperparameter tuning? Ask Genie Code before you start. Click on the genie icon <img src="../Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.  </div>  <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0B2026;">    <span id="genie-query-2-2" style="flex: 1;">How do I define a hyperparameter search space in Optuna and integrate it with MLflow tracking in Databricks? What samplers does Optuna support and when should I use TPE versus random or grid search?</span>    <button onclick="      var text = document.getElementById('genie-query-2-2').innerText;      var ta = document.createElement('textarea');      ta.value = text;      ta.style.position = 'fixed';      ta.style.opacity = '0';      document.body.appendChild(ta);      ta.select();      document.execCommand('copy');      document.body.removeChild(ta);      this.innerText = 'Copied!';      var btn = this;      setTimeout(function(){ btn.innerText = 'Copy'; }, 2000);    " style="background: #FF5F46; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-size: 13pt; cursor: pointer; white-space: nowrap;">Copy</button>  </div></div></div>
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

# MAGIC %pip install -U -qq optuna
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-2.2

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
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Dataset
# MAGIC
# MAGIC In this lab, you will be using a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including gender, as well as internet subscription details such as subscription plans and payment methods.
# MAGIC
# MAGIC In this lab, we will create and tune a model that will predict customer churn based on the **`Churn`** field. 
# MAGIC
# MAGIC A table with all features is already created for you.
# MAGIC
# MAGIC **Table name: `customer_churn`**

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

## Load the table from Unity Catalog called customer_churn
table_name = <FILL_IN>

## Read into a PySpark DataFrame and convert to Pandas DataFrame
diabetes_dataset = <FILL_IN>
customer_pd = <FILL_IN>

## Split dataset between features and targets. The target variable is Churn
target_col = <FILL_IN>
X_all = <FILL_IN>

## Encode target: map 'Yes' -> 1, 'No' -> 0
y_all = <FILL_IN>

## Encode categorical feature columns using one-hot encoding
X_all = <FILL_IN>

## Test / train split using 95% train / 5% test
X_train, X_test, y_train, y_test = train_test_split(<FILL_IN>)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Prepare Dataset
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC
# MAGIC ## Load the table from Unity Catalog called customer_churn
# MAGIC table_name = f"{DA.catalog_name}.{DA.schema_name}.customer_churn"
# MAGIC
# MAGIC ## Read into a PySpark DataFrame and convert to Pandas DataFrame
# MAGIC diabetes_dataset = spark.read.table(table_name)
# MAGIC customer_pd = diabetes_dataset.drop('CustomerID').toPandas()
# MAGIC
# MAGIC ## Split dataset between features and targets. The target variable is Churn
# MAGIC target_col = "Churn"
# MAGIC X_all = customer_pd.drop(labels=target_col, axis=1)
# MAGIC
# MAGIC ## Encode target: map 'Yes' -> 1, 'No' -> 0
# MAGIC y_all = customer_pd[target_col].map({'Yes': 1, 'No': 0})
# MAGIC
# MAGIC ## Encode categorical feature columns using one-hot encoding
# MAGIC ## The customer_churn dataset contains string columns (Gender, Partner, InternetService, etc.)
# MAGIC ## that must be converted to numeric before passing to DecisionTreeClassifier
# MAGIC X_all = pd.get_dummies(X_all, drop_first=True)
# MAGIC
# MAGIC ## Test / train split using 95% train / 5% test
# MAGIC X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.95, random_state=42)
# MAGIC print(f"We have {X_train.shape[0]} records in our training dataset")
# MAGIC print(f"We have {X_test.shape[0]} records in our test dataset")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define the Search Space and Optimization Function
# MAGIC
# MAGIC Define the parameter search space for Optuna.
# MAGIC
# MAGIC Your objective function should meet the following requirements:
# MAGIC
# MAGIC 1. Define the search space using the hyperparameters `max_depth` and `max_features`. For `max_depth`, the search range should be between 5 and 50, while `max_features` should be between 5 and 10. Additionally, for the `criterion` parameter, search based on `gini`, `entropy`, and `log_loss`. 
# MAGIC 1. Enable MLflow run as a nested experiment.
# MAGIC 1. For each run, log the cross-validation results for `accuracy`, `precision`, `recall`, and `f1`.
# MAGIC 1. Use **3-fold** cross-validation. Be sure to average the fold results using `.mean()`.
# MAGIC 1. The objective will be to _maximize_ **`precision`**.

# COMMAND ----------

import optuna
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from mlflow.models.signature import infer_signature

## Define the objective function
def optuna_objective_function(<FILL_IN>):
    params = {
        'criterion': <FILL_IN>,
        'max_depth': <FILL_IN>,
        'max_features': <FILL_IN>
    }
    
    with mlflow.start_run(nested=True, run_name=f"Optuna Trial {trial.number}"):
        
        ## Train model
        dtc = <FILL_IN>

        ## Perform cross-validation
        scoring_metrics = [<FILL_IN>]
        cv_results = cross_validate(<FILL_IN>)

        ## Create input signature using the first row of X_train
        input_example = X_train.iloc[[0]]
        signature = <FILL_IN>

        ## Compute and log average scores
        cv_results_avg = {metric: cv_results[f'test_{metric}'].mean() for metric in scoring_metrics}
        mlflow.log_metrics(<FILL_IN>)
        mlflow.log_params(<FILL_IN>)
        mlflow.sklearn.log_model(<FILL_IN>)

        ## Return precision to maximize it
        return <FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Step 1. Define Search Space and Optimization Function
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import optuna
# MAGIC import mlflow
# MAGIC import mlflow.sklearn
# MAGIC from sklearn.tree import DecisionTreeClassifier
# MAGIC from sklearn.model_selection import cross_validate
# MAGIC from mlflow.models.signature import infer_signature
# MAGIC
# MAGIC ## Define the objective function
# MAGIC def optuna_objective_function(trial):
# MAGIC     params = {
# MAGIC         'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
# MAGIC         'max_depth': trial.suggest_int('max_depth', 5, 50),
# MAGIC         'max_features': trial.suggest_int('max_features', 5, 10)
# MAGIC     }
# MAGIC
# MAGIC     with mlflow.start_run(nested=True, run_name=f"Optuna Trial {trial.number}"):
# MAGIC
# MAGIC         ## Train model
# MAGIC         dtc = DecisionTreeClassifier(**params)
# MAGIC         dtc.fit(X_train, y_train)
# MAGIC
# MAGIC         ## Perform cross-validation
# MAGIC         scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
# MAGIC         cv_results = cross_validate(dtc, X_train, y_train, cv=3, scoring=scoring_metrics)
# MAGIC
# MAGIC         ## Create input signature using the first row of X_train
# MAGIC         input_example = X_train.iloc[[0]]
# MAGIC         signature = infer_signature(input_example, dtc.predict(input_example))
# MAGIC
# MAGIC         ## Compute and log average scores
# MAGIC         cv_results_avg = {metric: cv_results[f'test_{metric}'].mean() for metric in scoring_metrics}
# MAGIC         mlflow.log_metrics(cv_results_avg)
# MAGIC         mlflow.log_params(params)
# MAGIC         mlflow.sklearn.log_model(dtc, "lab_optuna_decision_tree_model", signature = signature, input_example=input_example)
# MAGIC
# MAGIC         ## Return precision to maximize it
# MAGIC         return cv_results_avg['precision']
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create an Optuna Study and Log with MLflow
# MAGIC
# MAGIC First, we will delete all previous runs to keep our workspace and experiment tidy. Second, you will create an Optuna study and run the experiment with MLflow.

# COMMAND ----------

## Set the MLflow experiment name and get the id
experiment_name = f"/Users/{DA.username}/Lab_Optuna_Experiment_{DA.schema_name}"
print(f"Experiment Name: {experiment_name}")
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
print(f"Experiment ID: {experiment_id}")

print("Clearing out old runs (If you want to add more runs, change the n_trial parameter in the next cell) ...")
## Get all runs
runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="pandas")

if runs.empty:
    print("No runs found in the experiment.")
else:
    ## Iterate and delete each run
    for run_id in runs["run_id"]:
        mlflow.delete_run(run_id)
        print(f"Deleted run: {run_id}")

    print("All runs have been deleted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Study and Log with MLflow
# MAGIC
# MAGIC #### Instructions:
# MAGIC
# MAGIC 1. Create an Optuna study with name `lab_optuna_hpo`.
# MAGIC 1. Maximize the objective function. 
# MAGIC 1. Give the parent run the name `Lab_Optuna_Hyperparameter_Optimization`.
# MAGIC 1. Only run 10 trials with Optuna.

# COMMAND ----------

study = <FILL_IN>

with mlflow.start_run(run_name='Lab_Optuna_Hyperparameter_Optimization') as parent_run:
    <FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Step 2. Create Optuna Study and Log with MLflow
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC study = optuna.create_study(
# MAGIC     study_name="lab_optuna_hpo",
# MAGIC     direction="maximize"
# MAGIC )
# MAGIC
# MAGIC with mlflow.start_run(run_name='Lab_Optuna_Hyperparameter_Optimization') as parent_run:
# MAGIC     ## Run optimization
# MAGIC     study.optimize(
# MAGIC         optuna_objective_function, 
# MAGIC         n_trials=10,
# MAGIC         )
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Visual Inspection of Precision Values
# MAGIC
# MAGIC Here, we can view all 10 runs. After completing the code and running the following cell, scroll to the right and locate the column `metrics.precision`. Use the UI to order and order by descending. This will locate the largest precision score. Next, you will create a visual to also help understand the distribution of scores by trial. 
# MAGIC
# MAGIC
# MAGIC ### Creating a precision score visual
# MAGIC
# MAGIC 1. **Run the next cell** to generate the table output.  
# MAGIC 1. Click on the **plus (+) symbol** in the output cell.  
# MAGIC 1. Select **Visualization** from the options.  
# MAGIC 1. In the visualization settings, choose 
# MAGIC **Bar** and ensure **Horizontal Chart** toggle is **on**.  
# MAGIC 1. Configure the **Y-axis**:  
# MAGIC    - Set **Y Column** to `tags.mlflow.runName`.  
# MAGIC 1. Configure the **X-axis**:  
# MAGIC    - Set **X Columns** to `metrics.precision`.  
# MAGIC    - Choose **Sum** as the aggregation method.  
# MAGIC 1. Click on the **Y-axis tab**:  
# MAGIC    - Ensure **Show Labels** is **on**.  
# MAGIC 1. Apply the settings and visualize the data.
# MAGIC
# MAGIC
# MAGIC After following the above instructions, visually inspect which trial had the best run according to `precision`.

# COMMAND ----------

import mlflow
import pandas as pd

## Define your experiment name or ID
experiment_id = parent_fun.experiment_id

## Fetch all runs from the experiment using the MLflow API
df_runs = mlflow.search_runs(<FILL_IN>)

display(df_runs)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Step 3. Visual Inspection of Precision Values
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC import pandas as pd
# MAGIC
# MAGIC ## Define your experiment name or ID
# MAGIC experiment_id = parent_run.info.experiment_id # Replace with your actual experiment ID
# MAGIC
# MAGIC ## Fetch all runs from the experiment
# MAGIC df_runs = mlflow.search_runs(
# MAGIC   experiment_ids=[experiment_id]
# MAGIC   )
# MAGIC
# MAGIC display(df_runs)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4. Find the Best Run Programmatically
# MAGIC
# MAGIC In this step you will find the best scores using the Optuna library to find the best value and parameter values. Additionally, you will use MLflow to find these values. 
# MAGIC
# MAGIC #### Instructions
# MAGIC 1. Use the Optuna study to find the best precision score. 
# MAGIC 1. Use the Optuna study to find the best hyperparameter values. 
# MAGIC 1. Use the MLflow API to find the best run based on precision score.

# COMMAND ----------

## Display the best hyperparameters and metric
print(f"Best hyperparameters: {<FILL_IN>}")
print(f"Best precision score: {<FILL_IN>}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Step 4. Find the Best Run — Optuna
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ## Display the best hyperparameters and metric
# MAGIC print(f"Best hyperparameters: {study.best_params}")
# MAGIC print(f"Best precision score: {study.best_value}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

search_runs_pd = (mlflow.search_runs(<FILL_IN>))

## convert search_runs_pd to pyspark dataframe
search_runs_sd = <FILL_IN>
display(search_runs_pd)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Step 4. Find the Best Run — MLflow
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC search_runs_pd = (mlflow.search_runs(
# MAGIC     experiment_ids=[experiment_id],
# MAGIC     order_by=["metrics.precision DESC"],
# MAGIC     max_results=1))
# MAGIC
# MAGIC ## convert search_runs_pd to pyspark dataframe
# MAGIC search_runs_sd = spark.createDataFrame(search_runs_pd)
# MAGIC display(search_runs_pd)
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Best Model and Parameters and Register to Unity Catalog
# MAGIC
# MAGIC #### Instructions:
# MAGIC 1. Either use the results from above to copy and paste the run_id and experiment_id below or perform this task programmatically using `.collect()` on the `search_runs` PySpark DataFrame. 
# MAGIC 1. Load the model from MLflow.
# MAGIC 1. Display the results for the best model and parameters.

# COMMAND ----------

## Get the string value from run_id and experiment_id from PySpark DataFrame hpo_runs_df
run_id = <FILL_IN>
experiment_id = <FILL_IN>

print(f"Run ID: {run_id}")
print(f"Experiment ID: {experiment_id}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Load Best Model — Get Run ID
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC ## Get the string value from run_id and experiment_id from PySpark DataFrame hpo_runs_df
# MAGIC run_id = search_runs_sd.select("run_id").collect()[0][0]
# MAGIC experiment_id = search_runs_sd.select("experiment_id").collect()[0][0]
# MAGIC
# MAGIC print(f"Run ID: {run_id}")
# MAGIC print(f"Experiment ID: {experiment_id}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

import mlflow
import json
from mlflow.models import Model

## Grab an input example from the test set (pandas DataFrame)
input_example = X_test.iloc[[0]]

## You logged the model as: mlflow.sklearn.log_model(dtc, "lab_optuna_decision_tree_model", ...)
model_uri = <FILL_IN>

## Load the model
loaded_model = <FILL_IN>

## Retrieve model parameters MLflow client and get_run() method
client = <FILL_IN>
params = <FILL_IN>

## Display model parameters
print("Best Model Parameters:")
print(json.dumps(params, indent=4))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Load Best Model — Model Parameters
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC import json
# MAGIC
# MAGIC ## Grab an input example from the test set (pandas DataFrame)
# MAGIC input_example = X_test.iloc[[0]]
# MAGIC
# MAGIC ## You logged the model as: mlflow.sklearn.log_model(dtc, "lab_optuna_decision_tree_model", ...)
# MAGIC model_uri = f"runs:/{run_id}/lab_optuna_decision_tree_model"
# MAGIC
# MAGIC ## Load the model
# MAGIC loaded_model = mlflow.pyfunc.load_model(model_uri)
# MAGIC
# MAGIC ## Retrieve model parameters via MLflow client
# MAGIC client = mlflow.tracking.MlflowClient()
# MAGIC params = client.get_run(run_id).data.params
# MAGIC
# MAGIC ## Display model parameters
# MAGIC print("Best Model Parameters:")
# MAGIC print(json.dumps(params, indent=4))
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Model to Unity Catalog
# MAGIC
# MAGIC Register your model to Unity Catalog under the name `lab_optuna_model`. 
# MAGIC
# MAGIC > _You can get the catalog name and schema name using `DA.catalog_name` and `DA.schema_name`, respectively._

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_uri = <FILL_IN>
mlflow.register_model(<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">ANSWER</span> Register Model to Unity Catalog
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC model_uri = f'runs:/{run_id}/lab_optuna_decision_tree_model'
# MAGIC mlflow.register_model(model_uri=model_uri, name=f"{DA.catalog_name}.{DA.schema_name}.lab_optuna_model")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you learned about Optuna and how to integrate Optuna trials and studies with MLflow. You also demonstrated the ability to programmatically and visually inspect the best trial. Finally, you showed how to load the MLflow model and register it to Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>