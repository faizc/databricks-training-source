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
# MAGIC # LAB - Build a Feature Engineering Pipeline
# MAGIC
# MAGIC In this lab, you will build a complete feature engineering pipeline using the **CDC Diabetes Health Indicators** dataset. You will load and clean the data, create a Spark ML pipeline that handles missing values, encodes categorical features, and scales numerical features, then apply it consistently to both training and test sets. Finally, you will prepare the target column and save the pipeline for future reuse.
# MAGIC
# MAGIC **Lab Objectives**
# MAGIC
# MAGIC In this Lab, you will learn how to:
# MAGIC * **Task 1:** Load Dataset and Data Preparation
# MAGIC   * **1.1.** Load Dataset
# MAGIC   * **1.2.** Data Preparation — Type Casting, Missing Columns, Outlier Removal, Save Silver Table
# MAGIC * **Task 2:** Split Dataset into Training and Testing Sets
# MAGIC * **Task 3:** Create Feature Engineering Pipeline
# MAGIC   * **3.1.** Analyze Data Types and Missing Values
# MAGIC   * **3.2.** Define Pipeline Stages and Build Pipeline
# MAGIC * **Task 4:** Fit the Pipeline
# MAGIC * **Task 5:** Transform Datasets and Prepare the Target Column
# MAGIC * **Task 6:** Save and Load Pipeline

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
# MAGIC ## ⚠️ SERVERLESS: RESTART YOUR SESSION IF NEEDED
# MAGIC
# MAGIC <div style="border-left: 4px solid #FF9800; background: #FFF3E0; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #E65100; font-size: 1.1em;">Restart the Serverless Session</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Because this notebook runs on <strong>Serverless</strong>, you may occasionally need to start a fresh session for state to reset cleanly. To do this, either:</p>
# MAGIC <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC <li><strong>Restart the session</strong> — open the compute drop-down (top right) and choose <strong>New session</strong> to begin a new session, <em>or</em></li>
# MAGIC <li><strong>Terminate and start</strong> — <strong>terminate</strong> the current Serverless compute, then <strong>start</strong> it again before re-running the cells.</li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #F44336; background: #FFEBEE; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #C62828; font-size: 1.1em;">If you hit this error</strong>
# MAGIC <p style="margin: 8px 0 0 0;"><code style="background: #FCE4EC; color: #B71C1C; padding: 2px 6px; border-radius: 3px;">[CONNECT_ML.MODEL_SIZE_EXCEEDED_EXCEPTION] Spark Connect ML error: The fitted or loaded model size is about … bytes. Fit or load a model smaller than 268435456 bytes. SQLSTATE: 54000</code></p>
# MAGIC <p style="margin: 12px 0 0 0; color: #333;"><strong>Fix:</strong> <strong>Restart the session</strong> (compute drop-down → <strong>New session</strong>) <em>or</em> <strong>terminate and start</strong> the Serverless compute, then re-run the notebook cells from the top.</p>
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

# MAGIC %run ../Includes/Classroom-Setup-2.3

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
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Load Dataset and Data Preparation
# MAGIC
# MAGIC In this task, you will load the **CDC Diabetes Health Indicators** dataset and prepare it for machine learning. This health survey dataset contains demographic information and health indicators about respondents, including whether they have been diagnosed with diabetes.
# MAGIC
# MAGIC You will:
# MAGIC - Load the raw CSV file into a Spark DataFrame
# MAGIC - Perform initial data preparation: type casting, removing columns with excessive missing values, and filtering outliers
# MAGIC - Save the cleaned dataset as a **Delta silver table** for downstream use

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1.1: Load the Dataset
# MAGIC
# MAGIC Load the CDC diabetes dataset from the provided path. Use the following options:
# MAGIC - `.option("nullValue", "null")` — read the string `"null"` as a SQL `null`
# MAGIC - `header="true"` — the CSV file includes a header row
# MAGIC - `inferSchema="true"` — let Spark automatically detect column data types
# MAGIC - `multiLine="true"` — handle multi-line CSV fields correctly
# MAGIC
# MAGIC Once loaded, display the DataFrame to inspect its structure and column types.

# COMMAND ----------

## Set the path of the dataset
dataset_path = f"{DA.paths.datasets.cdc_diabetes}/cdc-diabetes/diabetes_binary_5050_raw.csv"

## Read the CSV file using Spark
## Set the header, inferSchema, multiLine, and nullValue options
cdc_df = <FILL_IN>

## Display the resulting DataFrame
<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 1.1 — Load Dataset — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT1a()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t1a" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC # Set the path of the dataset
# MAGIC dataset_path = f"{DA.paths.datasets.cdc_diabetes}/cdc-diabetes/diabetes_binary_5050_raw.csv"
# MAGIC
# MAGIC # Read the CSV file using Spark
# MAGIC cdc_df = spark.read.option("nullValue", "null").csv(
# MAGIC     dataset_path,
# MAGIC     header="true",
# MAGIC     inferSchema="true",
# MAGIC     multiLine="true",
# MAGIC     escape='"'
# MAGIC )
# MAGIC
# MAGIC # Display the resulting DataFrame
# MAGIC display(cdc_df)
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT1a() {
# MAGIC   const el = document.getElementById("copy-block-t1a");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackT1a(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackT1a(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT1a(text) {
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
# MAGIC ### Task 1.2: Data Preparation
# MAGIC
# MAGIC With the data loaded, the next step is to prepare it for modeling. You will:
# MAGIC - **Cast data types** — Convert integer and boolean columns to `double` for compatibility with Spark ML
# MAGIC - **Remove columns with too many missing values** — Drop any column where more than 60% of values are missing
# MAGIC - **Remove outliers** — Filter records with invalid or extreme values
# MAGIC - **Save the cleaned data** — Write to a Delta silver table for reuse

# COMMAND ----------

# MAGIC %md
# MAGIC **1.2a — Convert Data Types**
# MAGIC
# MAGIC Spark ML requires all feature columns to be numeric. Identify any `IntegerType` or `BooleanType` columns in the DataFrame and cast them to `DoubleType`. This ensures numerical consistency across all features.
# MAGIC
# MAGIC > Print the schema after casting to verify the changes.

# COMMAND ----------

from pyspark.sql.types import IntegerType, BooleanType
from pyspark.sql.functions import col

## Get a list of integer and boolean columns
integer_cols = <FILL_IN>

## Cast each to double for Spark ML compatibility
for column in integer_cols:
    cdc_df = cdc_df.withColumn(column, <FILL_IN>)

## Print the schema to verify the changes
cdc_df.<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 1.2a — Type Casting — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT1b()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t1b" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC from pyspark.sql.types import IntegerType, BooleanType
# MAGIC from pyspark.sql.functions import col
# MAGIC
# MAGIC # Get a list of integer and boolean columns
# MAGIC integer_cols = [
# MAGIC     c.name for c in cdc_df.schema.fields
# MAGIC     if isinstance(c.dataType, (IntegerType, BooleanType))
# MAGIC ]
# MAGIC
# MAGIC # Cast each to double for Spark ML compatibility
# MAGIC for column in integer_cols:
# MAGIC     cdc_df = cdc_df.withColumn(column, col(column).cast("double"))
# MAGIC
# MAGIC # Print the schema to verify the changes
# MAGIC cdc_df.printSchema()
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT1b() {
# MAGIC   const el = document.getElementById("copy-block-t1b");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackT1b(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackT1b(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT1b(text) {
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
# MAGIC **1.2b — Remove Columns with Too Many Missing Values**
# MAGIC
# MAGIC Columns with a very high proportion of missing values are unlikely to be useful for modeling and can cause issues during pipeline fitting. Count the missing values in each column, then drop any column where more than **60%** of rows are null.
# MAGIC
# MAGIC > Display the missing value counts before dropping, then display the cleaned DataFrame after.

# COMMAND ----------

from pyspark.sql.functions import col, when, sum as spark_sum

## Count missing values per column
missing_counts = cdc_df.agg(*[
    <FILL_IN>
    for c in cdc_df.columns
]).first().asDict()

## Display missing value counts as a summary DataFrame
missing_df = spark.createDataFrame(
    <FILL_IN>,
    ["column", "missing_count"]
)
display(missing_df.orderBy("missing_count", ascending=False))

## Set threshold: drop columns with more than 60% missing data
per_thresh = 0.6
N = cdc_df.<FILL_IN>
to_drop_missing = <FILL_IN>

print(f"Dropping columns with >{per_thresh * 100}% missing data: {to_drop_missing}")
cdc_no_missing_df = cdc_df.drop(*to_drop_missing)
display(cdc_no_missing_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 1.2b — Remove Missing-Heavy Columns — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT1c()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t1c" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC from pyspark.sql.functions import col, when, sum as spark_sum
# MAGIC
# MAGIC # Count missing values per column
# MAGIC missing_counts = cdc_df.agg(*[
# MAGIC     spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
# MAGIC     for c in cdc_df.columns
# MAGIC ]).first().asDict()
# MAGIC
# MAGIC # Display missing value counts as a summary DataFrame
# MAGIC missing_df = spark.createDataFrame(
# MAGIC     [(c, int(v)) for c, v in missing_counts.items()],
# MAGIC     ["column", "missing_count"]
# MAGIC )
# MAGIC display(missing_df.orderBy("missing_count", ascending=False))
# MAGIC
# MAGIC # Set threshold: drop columns with more than 60% missing data
# MAGIC per_thresh = 0.6
# MAGIC N = cdc_df.count()
# MAGIC
# MAGIC to_drop_missing = [
# MAGIC     c for c, v in missing_counts.items()
# MAGIC     if v / N >= per_thresh
# MAGIC ]
# MAGIC
# MAGIC print(f"Dropping columns with >{per_thresh * 100}% missing data: {to_drop_missing}")
# MAGIC
# MAGIC cdc_no_missing_df = cdc_df.drop(*to_drop_missing)
# MAGIC display(cdc_no_missing_df)
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT1c() {
# MAGIC   const el = document.getElementById("copy-block-t1c");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackT1c(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackT1c(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT1c(text) {
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
# MAGIC **1.2c — Remove Outliers**
# MAGIC
# MAGIC Before modeling, we remove records with values that fall outside plausible ranges. For this dataset:
# MAGIC - `MentHlth` — number of days of poor mental health; negative values are invalid (cutoff: `>= 0`)
# MAGIC - `BMI` — body mass index; values above 50 are considered extreme outliers (cutoff: `<= 50`)
# MAGIC
# MAGIC Apply both filters in a single step, then print the row count before and after.

# COMMAND ----------

## Define cutoff values
MentHlth_cutoff = 0   ## MentHlth cannot be negative
BMI_cutoff = 50       ## Reasonable upper limit for BMI

## Apply both filters in a single step
cdc_no_outliers_df = cdc_no_missing_df.filter(
    <FILL_IN>
)

## Print count before and after
print(<FILL_IN>)

## Display the filtered DataFrame
<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 1.2c — Remove Outliers — Solution
# MAGIC
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT1c()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t1c" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC ## Define cutoff values
# MAGIC MentHlth_cutoff = 0   ## MentHlth cannot be negative
# MAGIC BMI_cutoff = 50       ## Reasonable upper limit for BMI
# MAGIC
# MAGIC ## Apply both filters in a single step
# MAGIC cdc_no_outliers_df = cdc_no_missing_df.filter(
# MAGIC     (col("MentHlth") >= MentHlth_cutoff) & (col("BMI") <= BMI_cutoff)
# MAGIC )
# MAGIC
# MAGIC ## Print count before and after
# MAGIC print(f"Row count — Before: {cdc_no_missing_df.count()} / After: {cdc_no_outliers_df.count()}")
# MAGIC
# MAGIC ## Display the filtered DataFrame
# MAGIC display(cdc_no_outliers_df)
# MAGIC </code>
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT1c() {
# MAGIC   const el = document.getElementById("copy-block-t1c");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => fallbackT1c(text));
# MAGIC   } else {
# MAGIC     fallbackT1c(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT1c(text) {
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
# MAGIC **1.2d — Save the Cleaned Dataset as a Silver Table**
# MAGIC
# MAGIC Save the cleaned DataFrame as a Delta table. This creates a **silver table** — a cleaned, structured version of the raw data — that can be reused in downstream tasks without re-running the data preparation steps.

# COMMAND ----------

cdc_df_full = "cdc_df_full"

## Build the silver table name
cdc_df_full_silver = <FILL_IN>

## Save as Delta table (overwrite if exists)
cdc_no_outliers_df.write.mode("overwrite").option("mergeSchema", True).<FILL_IN>

print(f"Saved silver table: {cdc_df_full_silver}")
display(cdc_no_outliers_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 1.2d — Save Silver Table — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT1e()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t1e" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC # Base name for the dataset
# MAGIC cdc_df_full = "cdc_df_full"
# MAGIC
# MAGIC # Build the silver table name
# MAGIC cdc_df_full_silver = f"{cdc_df_full}_silver"
# MAGIC
# MAGIC # Save as Delta table (overwrite if exists)
# MAGIC cdc_no_outliers_df.write.mode("overwrite").option("mergeSchema", True).saveAsTable(cdc_df_full_silver)
# MAGIC
# MAGIC print(f"Saved silver table: {cdc_df_full_silver}")
# MAGIC
# MAGIC # Display the resulting DataFrame
# MAGIC display(cdc_no_outliers_df)
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT1e() {
# MAGIC   const el = document.getElementById("copy-block-t1e");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackT1e(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackT1e(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT1e(text) {
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
# MAGIC ## Task 2: Split Dataset into Training and Testing Sets
# MAGIC
# MAGIC Split the cleaned dataset into training and testing sets using an **80/20 split**. This ensures that:
# MAGIC - The pipeline is **fitted only on training data** (preventing data leakage)
# MAGIC - The test set remains unseen during pipeline fitting, for an unbiased evaluation
# MAGIC
# MAGIC After splitting, save both sets as Delta tables for reproducibility.

# COMMAND ----------

## Split the dataset: 80% training, 20% testing
train_df, test_df = cdc_no_outliers_df.<FILL_IN>

## Save each split as a Delta table
train_df.write.mode("overwrite").option("overwriteSchema", True).<FILL_IN>
test_df.write.mode("overwrite").option("overwriteSchema", True).<FILL_IN>

print(f"Train rows: {train_df.count()}, Test rows: {test_df.count()}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 2 — Split Dataset — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT2()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t2" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC # Split the dataset: 80% training, 20% testing
# MAGIC train_df, test_df = cdc_no_outliers_df.randomSplit([0.8, 0.2], seed=42)
# MAGIC
# MAGIC # Save each split as a Delta table
# MAGIC train_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(
# MAGIC     f"{DA.catalog_name}.{DA.schema_name}.cdc_df_train"
# MAGIC )
# MAGIC
# MAGIC test_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(
# MAGIC     f"{DA.catalog_name}.{DA.schema_name}.cdc_df_baseline"
# MAGIC )
# MAGIC
# MAGIC # Print row counts
# MAGIC print(f"Train rows: {train_df.count()}, Test rows: {test_df.count()}")
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT2() {
# MAGIC   const el = document.getElementById("copy-block-t2");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackT2(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackT2(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT2(text) {
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
# MAGIC ## Task 3: Create Feature Engineering Pipeline
# MAGIC
# MAGIC Now you will build a **Spark ML Pipeline** that automates the full feature transformation process. A pipeline ensures that every transformation learned from training data is applied identically to new data — this is the foundation of reproducible, production-ready ML workflows.
# MAGIC
# MAGIC **The pipeline will include the following stages:**
# MAGIC
# MAGIC | Step | Transformer | Purpose |
# MAGIC |------|------------|---------|
# MAGIC | 1 | `StringIndexer` | Convert string categories to numeric indices |
# MAGIC | 2 | `Imputer` | Fill missing numerical values using the mean |
# MAGIC | 3 | `VectorAssembler` | Combine numerical columns into a single vector |
# MAGIC | 4 | `RobustScaler` | Normalize numerical features, robust to outliers |
# MAGIC | 5 | `OneHotEncoder` | Convert categorical indices to binary sparse vectors |
# MAGIC | 6 | `VectorAssembler` | Combine all features into a final feature vector |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3.1: Analyze Data Types and Missing Values
# MAGIC
# MAGIC Before building the pipeline, analyze the training set to understand which columns need which treatment:
# MAGIC
# MAGIC - Cast any remaining `IntegerType` or `BooleanType` columns to `double` (apply to both `train_df` and `test_df`)
# MAGIC - Identify **categorical (string) columns** — these will go through `StringIndexer` → `OneHotEncoder`
# MAGIC - Identify **numerical (double) columns** — these will be imputed and scaled
# MAGIC - Identify **which numeric columns have missing values** — only those need imputation
# MAGIC
# MAGIC > **Important:** Exclude the target column `Diabetes_binary` from the feature columns. It is our prediction target, not an input feature.

# COMMAND ----------

from pyspark.sql.types import IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.functions import col, count, when

## Cast any remaining integer and boolean columns to double
## Apply to BOTH train_df and test_df
integer_cols = [c.name for c in train_df.schema.fields if <FILL_IN>]
for column in integer_cols:
    train_df = train_df.withColumn(column, col(column).cast(<FILL_IN>))
    test_df = test_df.withColumn(column, col(column).cast(<FILL_IN>))

## Define the target column — exclude it from all feature lists
target_col = "Diabetes_binary"

## Identify string (categorical) columns — excluding target
string_cols = [c.name for c in train_df.schema.fields if <FILL_IN>]

## Identify numeric columns — excluding target
num_cols = [c.name for c in train_df.schema.fields if <FILL_IN>]

## Find numeric columns with missing values (only these need Imputer)
num_missing_values_logic = [count(when(col(c).isNull(), c)).alias(c) for c in num_cols]
row_dict_num = train_df.select(num_missing_values_logic).first().<FILL_IN>
num_missing_cols = [c for c in row_dict_num if row_dict_num[c] > 0]

print(f"Categorical (string) columns: {string_cols}")
print(f"Numeric columns: {num_cols}")
print(f"Numeric columns with missing values: {num_missing_cols}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 3.1 — Analyze Data Types and Missing Values — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT3a()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t3a" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC from pyspark.sql.types import IntegerType, BooleanType, StringType, DoubleType
# MAGIC from pyspark.sql.functions import col, count, when
# MAGIC
# MAGIC # Cast any remaining integer and boolean columns to double
# MAGIC integer_cols = [
# MAGIC     c.name for c in train_df.schema.fields
# MAGIC     if isinstance(c.dataType, (IntegerType, BooleanType))
# MAGIC ]
# MAGIC
# MAGIC for column in integer_cols:
# MAGIC     train_df = train_df.withColumn(column, col(column).cast("double"))
# MAGIC     test_df = test_df.withColumn(column, col(column).cast("double"))
# MAGIC
# MAGIC # Define the target column — exclude it from all feature lists
# MAGIC target_col = "Diabetes_binary"
# MAGIC
# MAGIC # Identify string (categorical) columns — excluding target
# MAGIC string_cols = [
# MAGIC     c.name for c in train_df.schema.fields
# MAGIC     if isinstance(c.dataType, StringType) and c.name != target_col
# MAGIC ]
# MAGIC
# MAGIC # Identify numeric columns — excluding target
# MAGIC num_cols = [
# MAGIC     c.name for c in train_df.schema.fields
# MAGIC     if isinstance(c.dataType, DoubleType) and c.name != target_col
# MAGIC ]
# MAGIC
# MAGIC # Find numeric columns with missing values
# MAGIC num_missing_values_logic = [
# MAGIC     count(when(col(c).isNull(), c)).alias(c)
# MAGIC     for c in num_cols
# MAGIC ]
# MAGIC
# MAGIC row_dict_num = train_df.select(num_missing_values_logic).first().asDict()
# MAGIC
# MAGIC num_missing_cols = [
# MAGIC     c for c in row_dict_num
# MAGIC     if row_dict_num[c] > 0
# MAGIC ]
# MAGIC
# MAGIC print(f"Categorical (string) columns: {string_cols}")
# MAGIC print(f"Numeric columns: {num_cols}")
# MAGIC print(f"Numeric columns with missing values: {num_missing_cols}")
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT3a() {
# MAGIC   const el = document.getElementById("copy-block-t3a");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackT3a(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackT3a(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT3a(text) {
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
# MAGIC ### Task 3.2: Define Pipeline Stages and Build the Pipeline
# MAGIC
# MAGIC Using the column lists you identified in Task 3.1, define each pipeline stage and assemble them into a `Pipeline`.
# MAGIC
# MAGIC Follow these steps in order:
# MAGIC 1. **`StringIndexer`** — convert string category columns to numeric indices. Use `handleInvalid="keep"` so null values are treated as a separate category rather than causing an error.
# MAGIC 2. **`Imputer`** — fill missing values in numeric columns using the `mean` strategy. Only include columns that actually have missing values (`num_missing_cols`).
# MAGIC 3. **`VectorAssembler`** — assemble all numeric feature columns into a single vector called `numerical_assembled`.
# MAGIC 4. **`RobustScaler`** — scale the assembled numeric vector. `RobustScaler` uses the interquartile range (IQR) and is less sensitive to remaining outliers than `StandardScaler`.
# MAGIC 5. **`OneHotEncoder`** — convert the indexed categorical columns into binary sparse vectors. Use `handleInvalid="keep"`.
# MAGIC 6. **`VectorAssembler`** (final) — combine the scaled numeric features and the one-hot encoded categorical features into a single `features` column.
# MAGIC 7. **`Pipeline`** — pass all stages to a `Pipeline` in the correct order.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler, RobustScaler

## Step 1: StringIndexer — string categories to numeric indices
categorical_cols_indexed = [c + "_index" for c in string_cols]
string_indexer = StringIndexer(inputCols=<FILL_IN>, outputCols=<FILL_IN>, handleInvalid=<FILL_IN>)

## Step 2: Imputer — fill missing numeric values with mean
## Only impute columns that have missing values (num_missing_cols)
imputer = Imputer(inputCols=<FILL_IN>, outputCols=<FILL_IN>, strategy=<FILL_IN>)

## Step 3: VectorAssembler — combine numeric columns into one vector
numerical_assembler = VectorAssembler(inputCols=<FILL_IN>, outputCol=<FILL_IN>)

## Step 4: RobustScaler — normalize the numeric vector
numerical_scaler = RobustScaler(inputCol=<FILL_IN>, outputCol=<FILL_IN>)

## Step 5: OneHotEncoder — convert indices to binary sparse vectors
ohe_cols = [c + "_ohe" for c in string_cols]
one_hot_encoder = OneHotEncoder(inputCols=<FILL_IN>, outputCols=<FILL_IN>, handleInvalid=<FILL_IN>)

## Step 6: Final VectorAssembler — combine all features into one vector
feature_cols = ["numerical_scaled"] + <FILL_IN>
vector_assembler = VectorAssembler(inputCols=<FILL_IN>, outputCol=<FILL_IN>)

## Step 7: Build the Pipeline with all stages in order
stages_list = [string_indexer, imputer, numerical_assembler, numerical_scaler, one_hot_encoder, vector_assembler]
pipeline = <FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 3.2 — Define Pipeline Stages — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT3b()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t3b" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC from pyspark.ml import Pipeline
# MAGIC from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler, RobustScaler
# MAGIC
# MAGIC # Define the target column
# MAGIC target_col = "Diabetes_binary"
# MAGIC
# MAGIC # ---- Build stages list conditionally ----
# MAGIC stages_list = []
# MAGIC
# MAGIC # Step 1: StringIndexer — only if there are string columns
# MAGIC categorical_cols_indexed = [c + "_index" for c in string_cols]
# MAGIC if string_cols:
# MAGIC     string_indexer = StringIndexer(
# MAGIC         inputCols=string_cols,
# MAGIC         outputCols=categorical_cols_indexed,
# MAGIC         handleInvalid="keep"
# MAGIC     )
# MAGIC     stages_list.append(string_indexer)
# MAGIC
# MAGIC # Step 2: Imputer — only if there are numeric columns with missing values
# MAGIC if num_missing_cols:
# MAGIC     imputer = Imputer(
# MAGIC         inputCols=num_missing_cols,
# MAGIC         outputCols=num_missing_cols,
# MAGIC         strategy="mean"
# MAGIC     )
# MAGIC     stages_list.append(imputer)
# MAGIC
# MAGIC # Step 3: VectorAssembler — combine numeric columns into one vector
# MAGIC numerical_assembler = VectorAssembler(
# MAGIC     inputCols=num_cols,
# MAGIC     outputCol="numerical_assembled"
# MAGIC )
# MAGIC stages_list.append(numerical_assembler)
# MAGIC
# MAGIC # Step 4: RobustScaler — normalize the numeric vector
# MAGIC numerical_scaler = RobustScaler(
# MAGIC     inputCol="numerical_assembled",
# MAGIC     outputCol="numerical_scaled"
# MAGIC )
# MAGIC stages_list.append(numerical_scaler)
# MAGIC
# MAGIC # Step 5: OneHotEncoder — only if there are categorical columns
# MAGIC ohe_cols = [c + "_ohe" for c in string_cols]
# MAGIC if string_cols:
# MAGIC     one_hot_encoder = OneHotEncoder(
# MAGIC         inputCols=categorical_cols_indexed,
# MAGIC         outputCols=ohe_cols,
# MAGIC         handleInvalid="keep"
# MAGIC     )
# MAGIC     stages_list.append(one_hot_encoder)
# MAGIC
# MAGIC # Step 6: Final VectorAssembler — combine all available features
# MAGIC feature_cols = ["numerical_scaled"] + ohe_cols  # ohe_cols will be [] if no string cols
# MAGIC vector_assembler = VectorAssembler(
# MAGIC     inputCols=feature_cols,
# MAGIC     outputCol="features"
# MAGIC )
# MAGIC stages_list.append(vector_assembler)
# MAGIC
# MAGIC # Step 7: Build the Pipeline
# MAGIC pipeline = Pipeline(stages=stages_list)
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT3b() {
# MAGIC   const el = document.getElementById("copy-block-t3b");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackT3b(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackT3b(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT3b(text) {
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
# MAGIC ## Task 4: Fit the Pipeline
# MAGIC
# MAGIC Fit the pipeline on the **training dataset**. During fitting, each Estimator stage (e.g., `StringIndexer`, `Imputer`, `RobustScaler`) learns parameters from `train_df` — such as category mappings, mean values for imputation, and scaling factors. These learned parameters are stored in the resulting `PipelineModel`.
# MAGIC
# MAGIC > **Why fit on training data only?** Statistics computed from the test set (e.g., test means used for imputation) would introduce data leakage, making your evaluation results unreliable.

# COMMAND ----------

# Fit the pipeline on the training dataset
pipeline_model = <FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 4 — Fit the Pipeline — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT4()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t4" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC # Fit the pipeline on the training dataset
# MAGIC pipeline_model = pipeline.fit(train_df)
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT4() {
# MAGIC   const el = document.getElementById("copy-block-t4");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackT4(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackT4(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT4(text) {
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
# MAGIC ## Task 5: Transform Datasets and Prepare the Target Column
# MAGIC
# MAGIC Apply the fitted pipeline to both datasets, then prepare the target column for modeling.
# MAGIC
# MAGIC **5.1 — Transform datasets:** Use `pipeline_model.transform()` to apply all learned transformations to `train_df` and `test_df`. The result will include the final `features` vector column.
# MAGIC
# MAGIC **5.2 — Prepare the target column:** Spark ML models require the target label to be a numeric column named `label`. The `Diabetes_binary` column already contains `0.0` (no diabetes) and `1.0` (diabetes). Create a `label` column from it in both the training and test transformed DataFrames.
# MAGIC
# MAGIC > Display the final result showing both `features` and `label` to confirm the dataset is ready for modeling.

# COMMAND ----------

## Transform both datasets using the fitted pipeline
train_transformed_df = pipeline_model.<FILL_IN>
test_transformed_df = <FILL_IN>

## Create the 'label' column from 'Diabetes_binary' (already 0.0 / 1.0)
from pyspark.sql.functions import col
train_prepared_df = train_transformed_df.withColumn(<FILL_IN>)
test_prepared_df = test_transformed_df.withColumn(<FILL_IN>)

## Display features and label from the training set
display(train_prepared_df.select(<FILL_IN>))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 5 — Transform and Prepare Target — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT5()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t5" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC # Transform both datasets using the fitted pipeline
# MAGIC train_transformed_df = pipeline_model.transform(train_df)
# MAGIC test_transformed_df = pipeline_model.transform(test_df)
# MAGIC
# MAGIC # Create the 'label' column from 'Diabetes_binary' (already 0.0 / 1.0)
# MAGIC from pyspark.sql.functions import col
# MAGIC
# MAGIC train_prepared_df = train_transformed_df.withColumn("label", col("Diabetes_binary"))
# MAGIC test_prepared_df = test_transformed_df.withColumn("label", col("Diabetes_binary"))
# MAGIC
# MAGIC # Display features and label from the training set
# MAGIC display(train_prepared_df.select("features", "label"))
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT5() {
# MAGIC   const el = document.getElementById("copy-block-t5");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackT5(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackT5(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT5(text) {
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
# MAGIC ## Task 6: Save and Load the Pipeline
# MAGIC
# MAGIC Saving the fitted pipeline model allows you to reuse the exact same transformations in future sessions or production deployments — without re-fitting from scratch. In this task you will save the pipeline to the working directory, then load it back and inspect its stages.

# COMMAND ----------

# MAGIC %md
# MAGIC **6a — Save the Pipeline**

# COMMAND ----------

## Save the fitted pipeline model to the working directory
pipeline_model.<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 6a — Save Pipeline — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAnsT6a()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">Copy to clipboard</button>
# MAGIC <pre id="copy-block-t6a" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC ## Save the fitted pipeline model to the working directory
# MAGIC pipeline_model.write().overwrite().save(f"{DA.paths.working_dir}/spark_pipelines")
# MAGIC print(f"Pipeline saved to: {DA.paths.working_dir}/spark_pipelines")
# MAGIC </code></pre>
# MAGIC <script>
# MAGIC function copyAnsT6a() {
# MAGIC   const el = document.getElementById("copy-block-t6a");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => { console.error("Clipboard write failed:", err); fallbackT6a(text); });
# MAGIC   } else { fallbackT6a(text); }
# MAGIC }
# MAGIC function fallbackT6a(text) {
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
# MAGIC **6b — Load the Saved Pipeline and Inspect Its Stages**

# COMMAND ----------

from pyspark.ml import PipelineModel

## Load the saved pipeline model
loaded_pipeline = <FILL_IN>

## Display the pipeline stages to confirm the transformation order
<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 6b — Load Pipeline — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC
# MAGIC <button onclick="copyAnsT6b()" style="background:#1976d2; color:white; border:none; padding:6px 14px; border-radius:6px; cursor:pointer; font-size:0.85rem; margin: 8px 0 4px 0; display:inline-block;">
# MAGIC Copy to clipboard
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-block-t6b" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;"><code>
# MAGIC from pyspark.ml import PipelineModel
# MAGIC
# MAGIC # Load the saved pipeline model
# MAGIC loaded_pipeline = PipelineModel.load(f"{DA.paths.working_dir}/spark_pipelines")
# MAGIC
# MAGIC # Display the pipeline stages to confirm the transformation order
# MAGIC loaded_pipeline.stages
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAnsT6b() {
# MAGIC   const el = document.getElementById("copy-block-t6b");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => alert("Copied to clipboard"))
# MAGIC       .catch(err => {
# MAGIC         console.error("Clipboard write failed:", err);
# MAGIC         fallbackT6b(text);
# MAGIC       });
# MAGIC   } else {
# MAGIC     fallbackT6b(text);
# MAGIC   }
# MAGIC }
# MAGIC
# MAGIC function fallbackT6b(text) {
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
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you built an end-to-end feature engineering pipeline for the CDC Diabetes Health Indicators dataset using Spark ML.
# MAGIC
# MAGIC You performed key data preparation steps, including:
# MAGIC - Casting columns to appropriate data types  
# MAGIC - Removing columns with excessive missing values  
# MAGIC - Filtering out invalid or outlier values  
# MAGIC
# MAGIC You then split the dataset into training and test sets and constructed a reusable Spark ML pipeline that:
# MAGIC - Encodes categorical features using `StringIndexer` and `OneHotEncoder`  
# MAGIC - Imputes missing numerical values using `Imputer`  
# MAGIC - Scales numerical features using `RobustScaler`  
# MAGIC - Assembles all features into a single vector for modeling  
# MAGIC
# MAGIC You also prepared the target variable by converting `Diabetes_binary` into a numeric `label` column compatible with Spark ML models. Finally, you saved the pipeline for reuse, enabling consistent and reproducible data transformations.
# MAGIC
# MAGIC These core practices—structured data preparation, leakage-free transformations, and reusable pipelines—form the foundation for building reliable machine learning workflows in Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC