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
# MAGIC # LAB - Feature Engineering with Feature Store
# MAGIC
# MAGIC Welcome to the "Feature Engineering with Feature Store" lab! In this lesson, you will learn how to load and prepare a dataset for feature selection, explore and manipulate a feature table through the Databricks UI, perform feature selection on specific columns, create a new feature table, access feature table details using both the UI and API, merge two feature tables based on a common identifier, and efficiently delete unnecessary feature tables.
# MAGIC
# MAGIC **Lab Objectives:**
# MAGIC
# MAGIC In this Lab, you will learn how to:
# MAGIC
# MAGIC 1. Load and Prepare Dataset for Feature Selection
# MAGIC 2. Create a Feature Table
# MAGIC 3. Explore Feature Table through UI
# MAGIC 4. Access Feature Table Information
# MAGIC 5. Create Feature Table from Existing UC Table
# MAGIC 6. Enhance Feature Table with New Features
# MAGIC 7. Delete a Feature Table

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

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC Run the following cell to configure your working environment for this course.
# MAGIC
# MAGIC This setup will:
# MAGIC - Initialize the `DA` object (Databricks Academy helper)
# MAGIC - Configure your **default catalog** and **schema**
# MAGIC - Provision any supporting configuration needed for this lab
# MAGIC
# MAGIC **NOTE:** The `DA` object is only available in Databricks Academy courses.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.2

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains **variables such as your username, catalog name, schema name, working directory, and dataset locations**. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation

# COMMAND ----------

## Set the path of the dataset
shared_volume_name = 'cdc-diabetes' ## From Marketplace
csv_name = 'diabetes_binary_5050split_BRFSS2015' ## CSV file name
dataset_path = f"{DA.paths.datasets.cdc_diabetes}/{shared_volume_name}/{csv_name}.csv" ## Full path
silver_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')
display(silver_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Feature Selection
# MAGIC
# MAGIC The dataset is loaded and ready. We are assuming that most of the data cleaning and feature computation is already done and data is saved to a "silver" table.
# MAGIC
# MAGIC Select these features from the dataset: **"HighBP", "HighChol", "BMI", "Stroke", "PhysActivity", "GenHlth", "Sex", "Age", "Education", "Income".**
# MAGIC
# MAGIC Then drop the target column (`Diabetes_binary`) and create a `UID` column using `monotonically_increasing_id()` to serve as the primary key for the feature table.

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

## Select features we are interested in
silver_df = <FILL_IN>

## Drop the target column
silver_df = <FILL_IN>

## Create a UID column to be used as primary key
silver_df = <FILL_IN>

display(silver_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 1 — Feature Selection — Solution
# MAGIC
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT1()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t1" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC from pyspark.sql.functions import monotonically_increasing_id
# MAGIC
# MAGIC ## Select relevant feature columns
# MAGIC silver_df = silver_df.select(
# MAGIC     "Diabetes_binary",
# MAGIC     "HighBP",
# MAGIC     "HighChol",
# MAGIC     "BMI",
# MAGIC     "Stroke",
# MAGIC     "PhysActivity",
# MAGIC     "GenHlth",
# MAGIC     "Sex",
# MAGIC     "Age",
# MAGIC     "Education",
# MAGIC     "Income"
# MAGIC )
# MAGIC
# MAGIC ## Drop the target column
# MAGIC silver_df = silver_df.drop("Diabetes_binary")
# MAGIC
# MAGIC ## Create a unique identifier column (UID)
# MAGIC silver_df = silver_df.withColumn("UID", monotonically_increasing_id())
# MAGIC
# MAGIC display(silver_df)
# MAGIC </code>
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT1() {
# MAGIC   const el = document.getElementById("copy-block-t1");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(() => fallbackT1(text));
# MAGIC   } else {
# MAGIC     fallbackT1(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT1(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text;
# MAGIC   ta.style.position = "fixed";
# MAGIC   ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta);
# MAGIC   ta.select();
# MAGIC
# MAGIC   try {
# MAGIC     document.execCommand("copy");
# MAGIC     alert("Copied to clipboard");
# MAGIC   } catch (err) {
# MAGIC     alert("Could not copy. Please copy manually.");
# MAGIC   } finally {
# MAGIC     document.body.removeChild(ta);
# MAGIC   }
# MAGIC }
# MAGIC </script>
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task 2: Create a Feature Table
# MAGIC
# MAGIC Create a feature table from the `silver_df` dataset using the `FeatureEngineeringClient`. Define a description and tags as you wish.
# MAGIC
# MAGIC The new feature table name must be **`diabetes_features`**.
# MAGIC
# MAGIC **Note:** Do not define a partition column for this table.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = <FILL_IN>

diabetes_table_name = <FILL_IN>

fe.create_table(<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 2 — Create Feature Table — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAnsT2()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">Copy to clipboard</button>
# MAGIC <pre id="copy-block-t2" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC
# MAGIC fe = FeatureEngineeringClient()
# MAGIC
# MAGIC diabetes_table_name = f"{DA.catalog_name}.{DA.schema_name}.diabetes_features"
# MAGIC
# MAGIC fe.create_table(
# MAGIC     name=diabetes_table_name,
# MAGIC     primary_keys=["UID"],
# MAGIC     df=silver_df,
# MAGIC     description="Diabetes customer features",
# MAGIC     tags={"source": "bronze", "format": "delta", "owner": DA.username}
# MAGIC )
# MAGIC </code></pre>
# MAGIC <script>
# MAGIC function copyAnsT2() {
# MAGIC   const el = document.getElementById("copy-block-t2");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => { console.error("Clipboard write failed:", err); fallbackT2(text); });
# MAGIC   } else { fallbackT2(text); }
# MAGIC }
# MAGIC function fallbackT2(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta); ta.select();
# MAGIC   try { document.execCommand("copy"); alert("Copied to clipboard"); }
# MAGIC   catch (err) { alert("Could not copy. Please copy manually."); }
# MAGIC   finally { document.body.removeChild(ta); }
# MAGIC }
# MAGIC </script>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task 3: Explore Feature Table with the UI
# MAGIC
# MAGIC Now that the feature table is created, visit the **Features** page from the left panel and review the following information:
# MAGIC
# MAGIC * Check table columns — identify the **primary key** and **partition** columns.
# MAGIC
# MAGIC * View **sample data**.
# MAGIC
# MAGIC * View table **details**.
# MAGIC
# MAGIC * View **history**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Retrieve Feature Table Details
# MAGIC
# MAGIC Another way to access feature table information is through the API. Use the `FeatureEngineeringClient` to retrieve and print the **`features`** and **`primary_keys`** of the table.

# COMMAND ----------

ft = fe.<FILL_IN>
print(f"Features: {ft.<FILL_IN>}")
print(f"Primary Keys: {ft.<FILL_IN>}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 4 — Retrieve Feature Table Details — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAnsT4()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">Copy to clipboard</button>
# MAGIC <pre id="copy-block-t4" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC ft = fe.get_table(name=diabetes_table_name)
# MAGIC print(f"Features: {ft.features}")
# MAGIC print(f"Primary Keys: {ft.primary_keys}")
# MAGIC </code></pre>
# MAGIC <script>
# MAGIC function copyAnsT4() {
# MAGIC   const el = document.getElementById("copy-block-t4");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => { console.error("Clipboard write failed:", err); fallbackT4(text); });
# MAGIC   } else { fallbackT4(text); }
# MAGIC }
# MAGIC function fallbackT4(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta); ta.select();
# MAGIC   try { document.execCommand("copy"); alert("Copied to clipboard"); }
# MAGIC   catch (err) { alert("Could not copy. Please copy manually."); }
# MAGIC   finally { document.body.removeChild(ta); }
# MAGIC }
# MAGIC </script>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Create a Feature Table from an Existing UC Table
# MAGIC
# MAGIC A table called **`diet_features`** has already been created for you, containing diet-related features. Convert this existing Delta table into a feature table by:
# MAGIC
# MAGIC 1. Setting the `UID` column to `NOT NULL`
# MAGIC 2. Adding a `PRIMARY KEY` constraint on `UID`

# COMMAND ----------

display(spark.sql("SELECT * FROM diet_features"))

# COMMAND ----------

# MAGIC %sql
# MAGIC Set UID column to NOT NULL
# MAGIC <FILL_IN>
# MAGIC
# MAGIC Add PRIMARY KEY constraint on UID
# MAGIC <FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 5 — Create Feature Table from Existing UC Table — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAnsT5()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">Copy to clipboard</button>
# MAGIC <pre id="copy-block-t5" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC -- Run as a %sql cell
# MAGIC ALTER TABLE diet_features ALTER COLUMN UID SET NOT NULL;
# MAGIC ALTER TABLE diet_features ADD CONSTRAINT diet_features_pk_constraint PRIMARY KEY(UID);
# MAGIC </code></pre>
# MAGIC <script>
# MAGIC function copyAnsT5() {
# MAGIC   const el = document.getElementById("copy-block-t5");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => { console.error("Clipboard write failed:", err); fallbackT5(text); });
# MAGIC   } else { fallbackT5(text); }
# MAGIC }
# MAGIC function fallbackT5(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta); ta.select();
# MAGIC   try { document.execCommand("copy"); alert("Copied to clipboard"); }
# MAGIC   catch (err) { alert("Could not copy. Please copy manually."); }
# MAGIC   finally { document.body.removeChild(ta); }
# MAGIC }
# MAGIC </script>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6: Add New Features to Existing Table
# MAGIC
# MAGIC Read the `diet_features` table and merge it into the existing `diabetes_features` table. Both tables share `UID` as a unique identifier — use `fe.write_table()` with `mode="merge"` to combine them, then display the updated feature table.

# COMMAND ----------

diet_features = spark.sql("SELECT * FROM diet_features")

## Merge diet features into the diabetes feature table
fe.<FILL_IN>

## Read and display the merged feature table
display(fe.<FILL_IN>)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 6 — Add New Features to Existing Table — Solution
# MAGIC
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT6()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t6" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC
# MAGIC diet_features = spark.sql("SELECT *, CAST(UID AS LONG) AS uniqueID FROM diet_features")
# MAGIC
# MAGIC ## Merge the diet features into the existing diabetes feature table
# MAGIC fe.write_table(
# MAGIC     name=diabetes_table_name,
# MAGIC     df=diet_features,
# MAGIC     mode="merge"
# MAGIC )
# MAGIC
# MAGIC ## Read and display the updated feature table
# MAGIC display(fe.read_table(name=diabetes_table_name))
# MAGIC </code>
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT6() {
# MAGIC   const el = document.getElementById("copy-block-t6");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(() => fallbackT6(text));
# MAGIC   } else {
# MAGIC     fallbackT6(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT6(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text;
# MAGIC   ta.style.position = "fixed";
# MAGIC   ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta);
# MAGIC   ta.select();
# MAGIC
# MAGIC   try {
# MAGIC     document.execCommand("copy");
# MAGIC     alert("Copied to clipboard");
# MAGIC   } catch (err) {
# MAGIC     alert("Could not copy. Please copy manually.");
# MAGIC   } finally {
# MAGIC     document.body.removeChild(ta);
# MAGIC   }
# MAGIC }
# MAGIC </script>
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7: Delete a Feature Table
# MAGIC
# MAGIC After merging, the `diet_features` table is no longer needed. Use `fe.drop_table()` to delete it from the feature store.

# COMMAND ----------

diet_table_name = f"{DA.catalog_name}.{DA.schema_name}.diet_features"

## Drop the feature table
fe.<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 7 — Delete a Feature Table — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAnsT7()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">Copy to clipboard</button>
# MAGIC <pre id="copy-block-t7" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC diet_table_name = f"{DA.catalog_name}.{DA.schema_name}.diet_features"
# MAGIC
# MAGIC fe.drop_table(
# MAGIC     name=diet_table_name
# MAGIC )
# MAGIC </code></pre>
# MAGIC <script>
# MAGIC function copyAnsT7() {
# MAGIC   const el = document.getElementById("copy-block-t7");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => { console.error("Clipboard write failed:", err); fallbackT7(text); });
# MAGIC   } else { fallbackT7(text); }
# MAGIC }
# MAGIC function fallbackT7(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta); ta.select();
# MAGIC   try { document.execCommand("copy"); alert("Copied to clipboard"); }
# MAGIC   catch (err) { alert("Could not copy. Please copy manually."); }
# MAGIC   finally { document.body.removeChild(ta); }
# MAGIC }
# MAGIC </script>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you demonstrated the use of the Databricks Feature Store to perform feature engineering tasks. You loaded, prepared, and selected features from a dataset, created a feature table, explored and accessed table details through both the UI and API, merged two feature tables using a shared identifier, and deleted unnecessary tables.
# MAGIC
# MAGIC This hands-on experience has enhanced your feature engineering skills on the Databricks platform.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC