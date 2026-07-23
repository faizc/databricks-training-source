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
# MAGIC # Supervised Learning
# MAGIC
# MAGIC In this demo, we will guide you through two supervised learning models: a **Linear Regression** model and a **Decision Tree** model. You will learn how to retrieve data, fit and evaluate each model using the scikit-learn API, interpret their results, and observe how MLflow automatically captures training metrics via Autologging.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC * Fit a linear regression model on modeling data using the sklearn API.
# MAGIC
# MAGIC * Interpret the fit of an sklearn linear model’s coefficients and intercept.
# MAGIC
# MAGIC * Fit a decision tree model using sklearn API and training data.
# MAGIC
# MAGIC * Visualize an sklearn tree’s split points.
# MAGIC
# MAGIC * Identify which metrics are tracked by MLflow.
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
# MAGIC <div style="width: 100%; font-family: sans-serif;"><div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">  <img src="../Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">  <div style="font-size: 15pt; color: #0B2026; line-height: 1.7; margin-bottom: 16px;">    Want to know more about supervised learning and classification in Databricks? Ask Genie Code. Click on the genie icon <img src="../Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.  </div>  <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0B2026;">    <span id="genie-query-1-1a" style="flex: 1;">What is supervised learning? What are the main classification algorithms available in scikit-learn, and how do I choose the right one for a binary classification task like customer churn prediction?</span>    <button onclick="      var text = document.getElementById('genie-query-1-1a').innerText;      var ta = document.createElement('textarea');      ta.value = text;      ta.style.position = 'fixed';      ta.style.opacity = '0';      document.body.appendChild(ta);      ta.select();      document.execCommand('copy');      document.body.removeChild(ta);      this.innerText = 'Copied!';      var btn = this;      setTimeout(function(){ btn.innerText = 'Copy'; }, 2000);    " style="background: #FF5F46; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-size: 13pt; cursor: pointer; white-space: nowrap;">Copy</button>  </div></div></div>
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
# MAGIC - Provision any supporting configuration needed for this demo
# MAGIC
# MAGIC **NOTE:** The `DA` object is only available in Databricks Academy courses.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-1.1a

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
# MAGIC In this section, we are going to prepare the dataset for our machine learning models. The dataset we'll be working with is the **California housing dataset**. 
# MAGIC
# MAGIC The dataset has been loaded, cleaned and saved to a **feature table**. We will read data directly from this table.
# MAGIC
# MAGIC Then, we will split the dataset into **train and test** sets.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Dataset
# MAGIC
# MAGIC This dataset contains information about housing districts in California and **aims to predict the median house value** for California districts, based on various features.
# MAGIC
# MAGIC While data cleaning and feature engineering is out of the scope of this demo, we will only map the `ocean_proximity` field. 
# MAGIC

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# read data from the feature store
table_name = f"{DA.catalog_name}.{DA.schema_name}.ca_housing"
feature_data_pd = fe.read_table(name=table_name).toPandas()
feature_data_pd = feature_data_pd.drop(columns=['unique_id'])

# COMMAND ----------

ocean_proximity_mapping = {
    'NEAR BAY': 1,
    '<1H OCEAN': 2,
    'INLAND': 3,
    'NEAR OCEAN': 4,
    'ISLAND': 5  
}

# Replace values in the DataFrame
feature_data_pd['ocean_proximity'] = feature_data_pd['ocean_proximity'].replace(ocean_proximity_mapping).astype(float)

# Print the updated DataFrame
feature_data_pd = feature_data_pd.fillna(0)

display(feature_data_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split
# MAGIC
# MAGIC Split the dataset into training and testing sets. This is essential for evaluating the performance of machine learning models.

# COMMAND ----------

from sklearn.model_selection import train_test_split

print(f"We have {feature_data_pd.shape[0]} records in our source dataset")

# split target variable into its own dataset
target_col = "median_house_value"
X_all = feature_data_pd.drop(labels=target_col, axis=1)
y_all = feature_data_pd[target_col]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.8, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine for Potential Co-linearity
# MAGIC
# MAGIC Now, let's examine the correlations between predictors to identify potential co-linearity. Understanding the relationships between different features can provide insights into the dataset and help us make informed decisions during the modeling process.
# MAGIC
# MAGIC Let's review the **correlation matrix** in **tabular format**. Also, we can create a **graph based on the correlation matrix** to easily inspect the matrix.

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Combine X and y into a single DataFrame for simplicity
data = pd.concat([X_train, y_train], axis=1)

# Calculate correlation matrix
corr = data.corr()

# display correlation matrix
pd.set_option('display.max_columns', 10)
print(corr)

# COMMAND ----------

# display correlation matrix visually

# Initialize figure
plt.figure(figsize=(8, 8))
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        # Determine the color based on positive or negative correlation
        color = 'blue' if corr.iloc[i, j] > 0 else 'red'

        # don't fill in circles on the diagonal
        fill = not( i == j )

        # Plot the circle with size corresponding to the absolute value of correlation
        plt.gca().add_patch(plt.Circle((j, i), 
                                       0.5 * np.abs(corr.iloc[i, j]), 
                                       color=color, 
                                       edgecolor=color,
                                       fill=fill,
                                       alpha=0.5))



plt.xlim(-0.5, len(corr.columns) - 0.5)
plt.ylim(-0.5, len(corr.columns) - 0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(np.arange(len(corr.columns)), corr.columns)
plt.title('Correlogram')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit a Regression Model
# MAGIC
# MAGIC To enhance the performance of our regression model, we'll scale our input variables so that they are on a common (standardized) scale. **Standardization ensures that each feature has a mean of 0 and a standard deviation of 1**, which can be beneficial for certain algorithms, including linear regression.

# COMMAND ----------

from math import sqrt

import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# turn on autologging
mlflow.sklearn.autolog(log_input_examples=True)

# apply the Standard Scaler to all our input columns
std_ct = ColumnTransformer(transformers=[("scaler", StandardScaler(), ["total_bedrooms", "total_rooms", "housing_median_age", "latitude", "longitude", "median_income", "population", "ocean_proximity", "households"])])

# pipeline to transform inputs and then pass results to the linear regression model
lr_pl = Pipeline(steps=[
  ("tx_inputs", std_ct),
  ("lr", LinearRegression() )
])

# fit our model
lr_mdl = lr_pl.fit(X_train, y_train)

# evaluate the test set
predicted = lr_mdl.predict(X_test)
test_r2 = r2_score(y_test, predicted)
test_mse = mean_squared_error(y_test, predicted)
test_rmse = sqrt(test_mse)
test_mape = mean_absolute_percentage_error(y_test, predicted)
print("Test evaluation summary:")
print(f"R^2: {test_r2}")
print(f"MSE: {test_mse}")
print(f"RMSE: {test_rmse}")
print(f"MAPE: {test_mape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine Model Result
# MAGIC
# MAGIC Now, let's inspect the results of our linear regression model. We'll examine both the intercept and the coefficients of the fitted model. Additionally, we'll perform a **t-test on each coefficient to assess its significance in contributing to the overall model**.

# COMMAND ----------

lr_mdl

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy import stats

# Extracting coefficients and intercept
coefficients = np.append([lr_mdl.named_steps['lr'].intercept_], lr_mdl.named_steps['lr'].coef_)
coefficient_names = ['Intercept'] + X_train.columns.to_list()

# Calculating standard errors and other statistics (this is a simplified example)
# In a real scenario, you might need to calculate these values more rigorously
n_rows, n_cols = X_train.shape
X_with_intercept = np.append(np.ones((n_rows, 1)), X_train, axis=1)
var_b = test_mse * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept)).diagonal()
standard_errors = np.sqrt(var_b)
t_values = coefficients / standard_errors
p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(X_with_intercept) - 1))) for i in t_values]

# Creating a DataFrame for display
summary_df = pd.DataFrame({'Coefficient': coefficients,
                           'Standard Error': standard_errors,
                           't-value': t_values,
                           'p-value': p_values},
                          index=coefficient_names)

# Print the DataFrame
print(summary_df)

# COMMAND ----------

import matplotlib.pyplot as plt

# Plotting the feature importances
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(coefficient_names))
plt.bar(y_pos, coefficients, align='center', alpha=0.7)
plt.xticks(y_pos, coefficient_names, rotation=45)
plt.ylabel('Coefficient Size')
plt.title('Coefficients in Linear Regression')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Fit a Decision Tree Model
# MAGIC
# MAGIC Decision trees learn by recursively splitting the training data on the feature that gives the
# MAGIC best reduction in impurity. Unlike linear regression, they make no assumptions about feature
# MAGIC relationships and can capture non-linear patterns. We limit the tree to **three levels deep**
# MAGIC (`max_depth=3`) to keep it interpretable and avoid overfitting.

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

dt_mdl = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_mdl.fit(X_train, y_train)

dt_train_r2 = dt_mdl.score(X_train, y_train)
dt_test_r2  = dt_mdl.score(X_test,  y_test)
print(f"Decision Tree — Train R²: {dt_train_r2:.4f}")
print(f"Decision Tree — Test  R²: {dt_test_r2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize the Tree's Split Points
# MAGIC
# MAGIC `plot_tree` renders every node in the fitted tree, showing the **split feature**, **threshold**,
# MAGIC **impurity (MSE)**, and **sample count** at each node. Reading the top levels reveals which
# MAGIC features drive the biggest variance reductions — in the California Housing dataset you will
# MAGIC typically see **median income (`MedInc`)** dominate the root split.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(16, 6))
plot_tree(
    dt_mdl,
    feature_names=X_train.columns.tolist(),
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    max_depth=3,
)
ax.set_title(
    "Decision Tree Regressor — Top 3 Split Levels (California Housing)",
    fontsize=12,
)
plt.tight_layout()
plt.show()

print(f"Root split feature : {X_train.columns[dt_mdl.tree_.feature[0]]}")
print(f"Tree depth         : {dt_mdl.get_depth()}")
print(f"Leaf nodes         : {dt_mdl.get_n_leaves()}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we trained and evaluated two supervised learning models on the
# MAGIC California Housing dataset. The **Linear Regression** model offered a transparent,
# MAGIC coefficient-based view of each feature's contribution, while the **Decision Tree**
# MAGIC model captured non-linear relationships through recursive feature splits. Comparing
# MAGIC their R² scores highlights the bias-variance trade-off: linear models are stable but
# MAGIC may underfit complex data, whereas shallow trees balance flexibility and
# MAGIC interpretability. MLflow Autologging recorded all parameters and metrics automatically,
# MAGIC giving you a reproducible experiment record in the Databricks UI.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>