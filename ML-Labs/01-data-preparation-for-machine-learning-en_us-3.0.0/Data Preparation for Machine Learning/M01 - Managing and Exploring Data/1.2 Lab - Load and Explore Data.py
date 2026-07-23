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
# MAGIC # Lab - Load and Explore Data
# MAGIC
# MAGIC In this lab, you will build hands-on experience working with data in a Databricks environment using PySpark. You will load data, explore its structure, compute summary statistics, and use **Genie Code** to generate visualizations that support machine learning workflows.
# MAGIC
# MAGIC You will also work with Delta Lake to manage data, control access, and explore versioned data using time travel.
# MAGIC
# MAGIC **Lab Objectives**
# MAGIC
# MAGIC In this lab, you will:
# MAGIC
# MAGIC 1. Load data from a CSV file and store it as a Delta table  
# MAGIC 2. Manage access permissions on a Delta table  
# MAGIC 3. Compute and interpret summary statistics  
# MAGIC 4. Use **Genie Code** to explore the dataset, including:
# MAGIC    - Assessing class balance and target distribution  
# MAGIC    - Identifying missing value patterns  
# MAGIC    - Exploring feature distributions  
# MAGIC    - Comparing key features against the target variable  
# MAGIC 5. Modify the dataset by removing a column  
# MAGIC 6. Use Delta Lake time travel to access a previous version of the data  
# MAGIC 7. View and interpret the version history of a Delta table

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

# MAGIC %run ../Includes/Classroom-Setup-1.2

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this lab, we'll make use of the object `DA`, which provides critical variables. Execute the code block below to see various variables that will be used in this notebook:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Read Data from Delta Table
# MAGIC
# MAGIC
# MAGIC + Use Spark to read the CSV file and write it as a Delta table into a DataFrame.

# COMMAND ----------

## Load dataset with spark
shared_volume_name = 'telco' ## From Marketplace
csv_name = 'telco-customer-churn-missing' ## CSV file name
dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" ## Full path

## Read dataset with spark
telco_df = <FILL_IN>

table_name = "telco_missing"
table_name_bronze = f"{table_name}_bronze"

## Write it as delta table
telco_df.write.<FILL_IN>
telco_df.show()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 1: Read Data from Delta Table — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAns1()" style="background:#1976d2; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:600; margin: 8px 0;">Copy to clipboard</button>
# MAGIC
# MAGIC <pre id="copy-ans-1" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC ## Load dataset with spark
# MAGIC shared_volume_name = 'telco'
# MAGIC csv_name = 'telco-customer-churn-missing'
# MAGIC dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv"
# MAGIC
# MAGIC ## Read dataset with spark
# MAGIC telco_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')
# MAGIC
# MAGIC table_name = "telco_missing"
# MAGIC table_name_bronze = f"{table_name}_bronze"
# MAGIC
# MAGIC ## Write it as delta table
# MAGIC telco_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name_bronze)
# MAGIC telco_df.show()
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAns1() {
# MAGIC   const el = document.getElementById("copy-ans-1");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text).then(() => alert("Copied to clipboard")).catch(err => fallbackAns1(text));
# MAGIC   } else { fallbackAns1(text); }
# MAGIC }
# MAGIC function fallbackAns1(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta); ta.select();
# MAGIC   try { document.execCommand("copy"); alert("Copied to clipboard"); }
# MAGIC   catch (e) { alert("Could not copy. Please copy manually."); }
# MAGIC   finally { document.body.removeChild(ta); }
# MAGIC }
# MAGIC </script>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Manage Data Permissions
# MAGIC
# MAGIC Establish controlled access to the Telco Delta table by granting specific permissions for essential actions.
# MAGIC
# MAGIC + Grant permissions for specific actions (e.g., read) on the Delta table.

# COMMAND ----------

# MAGIC %sql
# MAGIC ---- Write query to Grant Permission to all the users to access Delta Table
# MAGIC <FILL_IN>;

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 2: Manage Data Permissions — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAns2()" style="background:#1976d2; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:600; margin: 8px 0;">Copy to clipboard</button>
# MAGIC
# MAGIC <pre id="copy-ans-2" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC %sql
# MAGIC GRANT SELECT ON TABLE telco_missing_bronze TO `account users`;
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAns2() {
# MAGIC   const el = document.getElementById("copy-ans-2");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text).then(() => alert("Copied to clipboard")).catch(err => fallbackAns2(text));
# MAGIC   } else { fallbackAns2(text); }
# MAGIC }
# MAGIC function fallbackAns2(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta); ta.select();
# MAGIC   try { document.execCommand("copy"); alert("Copied to clipboard"); }
# MAGIC   catch (e) { alert("Could not copy. Please copy manually."); }
# MAGIC   finally { document.body.removeChild(ta); }
# MAGIC }
# MAGIC </script>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Show Summary Statistics
# MAGIC
# MAGIC Compute and present key statistical metrics to gain a comprehensive understanding of the Telco dataset.
# MAGIC
# MAGIC + Use PySpark to compute and display summary statistics for the Telco dataset.
# MAGIC + Include key metrics such as mean, standard deviation, min, max, and more.

# COMMAND ----------

## Show summary of the Data
<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 3: Show Summary Statistics — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAns3()" style="background:#1976d2; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:600; margin: 8px 0;">Copy to clipboard</button>
# MAGIC
# MAGIC <pre id="copy-ans-3" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC ## Show summary of the Data
# MAGIC display(telco_df.summary())
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAns3() {
# MAGIC   const el = document.getElementById("copy-ans-3");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text).then(() => alert("Copied to clipboard")).catch(err => fallbackAns3(text));
# MAGIC   } else { fallbackAns3(text); }
# MAGIC }
# MAGIC function fallbackAns3(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta); ta.select();
# MAGIC   try { document.execCommand("copy"); alert("Copied to clipboard"); }
# MAGIC   catch (e) { alert("Could not copy. Please copy manually."); }
# MAGIC   finally { document.body.removeChild(ta); }
# MAGIC }
# MAGIC </script>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Explore the Dataset Using Genie Code
# MAGIC
# MAGIC In this task, you will use **Genie Code** to perform a guided exploratory analysis of the Telco dataset.
# MAGIC
# MAGIC Instead of writing code manually, you will provide a **single, well-structured prompt** and allow Genie Code to generate multiple steps of analysis. You may need to review, run, and refine the generated code as you proceed.
# MAGIC
# MAGIC The goal is to use Genie Code to understand:
# MAGIC - the target variable (`Churn`)
# MAGIC - missing data patterns
# MAGIC - key feature relationships
# MAGIC - potential signals useful for machine learning
# MAGIC
# MAGIC Run the cell below to display the dataset, then use the prompt that follows.

# COMMAND ----------

## Display the data — run this cell before using the Genie Code prompts below
display(telco_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Use Genie Code to Perform Exploratory Analysis
# MAGIC
# MAGIC **Copy the prompt below into Genie Code and execute it step by step.**
# MAGIC
# MAGIC <button id="copy-genie-btn-t4" onclick="copyGenieT4()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-genie-t4" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.4; white-space:pre-wrap;">
# MAGIC You are a data analyst working in a Databricks notebook.
# MAGIC
# MAGIC We have a Spark DataFrame called `telco_df`.
# MAGIC
# MAGIC Your task is to perform a structured exploratory data analysis for a machine learning use case (predicting customer churn).
# MAGIC
# MAGIC Follow these steps and generate code in separate cells:
# MAGIC
# MAGIC 1. Convert the Spark DataFrame to a pandas DataFrame called `telco_pdf`.
# MAGIC
# MAGIC 2. Analyze the target variable:
# MAGIC    - Show the distribution of the `Churn` column
# MAGIC    - Display both counts and percentages
# MAGIC    - Comment on whether the dataset is balanced or imbalanced
# MAGIC
# MAGIC 3. Analyze missing values:
# MAGIC    - Calculate the percentage of missing values per column
# MAGIC    - Visualize the result using a bar chart
# MAGIC    - Highlight columns with significant missing data
# MAGIC
# MAGIC 4. Explore feature relationships:
# MAGIC    - Compare `MonthlyCharges` and `tenure` across churn categories using boxplots
# MAGIC    - Identify any visible patterns between these features and churn
# MAGIC
# MAGIC 5. Explore categorical features:
# MAGIC    - Analyze how `Contract` and `InternetService` relate to churn
# MAGIC    - Use countplots or similar visualizations
# MAGIC
# MAGIC 6. Based on the analysis:
# MAGIC    - Print a short summary of key insights that could be useful for building a machine learning model
# MAGIC
# MAGIC Use matplotlib or seaborn for visualizations.
# MAGIC Keep the code clean and readable for learners.
# MAGIC Display all charts inline.
# MAGIC
# MAGIC Break the output into multiple logical steps (multiple code cells if needed).
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyGenieT4() {
# MAGIC   const el = document.getElementById("copy-genie-t4");
# MAGIC   const btn = document.getElementById("copy-genie-btn-t4");
# MAGIC   if (!el || !btn) return;
# MAGIC
# MAGIC   const text = el.innerText;
# MAGIC   const originalText = "Copy Genie Prompt";
# MAGIC
# MAGIC   function showCopied() {
# MAGIC     btn.innerText = "✅ Copied";
# MAGIC     setTimeout(() => {
# MAGIC       btn.innerText = originalText;
# MAGIC     }, 2000);
# MAGIC   }
# MAGIC
# MAGIC   function fallbackGenieT4(text) {
# MAGIC     const ta = document.createElement("textarea");
# MAGIC     ta.value = text;
# MAGIC     ta.style.position = "fixed";
# MAGIC     ta.style.left = "-9999px";
# MAGIC     document.body.appendChild(ta);
# MAGIC     ta.select();
# MAGIC     try {
# MAGIC       document.execCommand("copy");
# MAGIC       showCopied();
# MAGIC     } catch (e) {
# MAGIC       alert("Could not copy automatically. Please copy manually.");
# MAGIC     } finally {
# MAGIC       document.body.removeChild(ta);
# MAGIC     }
# MAGIC   }
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => showCopied())
# MAGIC       .catch(() => fallbackGenieT4(text));
# MAGIC   } else {
# MAGIC     fallbackGenieT4(text);
# MAGIC   }
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Drop the Column
# MAGIC Remove a specific column to enhance data cleanliness and focus.
# MAGIC
# MAGIC + Identify the column that needs to be dropped. For this task, drop the `SeniorCitizen` column.
# MAGIC
# MAGIC + Use the appropriate command to drop the identified column from the Telco dataset.
# MAGIC
# MAGIC + Verify that the column has been successfully dropped by displaying the updated dataset.

# COMMAND ----------

## Drop SeniorCitizen Column
telco_dropped_df = <FILL_IN>

## Overwrite the Delta table
telco_dropped_df.write.mode("overwrite")<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 5: Drop the Column — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAns5()" style="background:#1976d2; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:600; margin: 8px 0;">Copy to clipboard</button>
# MAGIC
# MAGIC <pre id="copy-ans-5" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC ## Drop SeniorCitizen Column
# MAGIC telco_dropped_df = telco_df.drop("SeniorCitizen")
# MAGIC
# MAGIC ## Overwrite the Delta table
# MAGIC telco_dropped_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name_bronze)
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAns5() {
# MAGIC   const el = document.getElementById("copy-ans-5");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text).then(() => alert("Copied to clipboard")).catch(err => fallbackAns5(text));
# MAGIC   } else { fallbackAns5(text); }
# MAGIC }
# MAGIC function fallbackAns5(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta); ta.select();
# MAGIC   try { document.execCommand("copy"); alert("Copied to clipboard"); }
# MAGIC   catch (e) { alert("Could not copy. Please copy manually."); }
# MAGIC   finally { document.body.removeChild(ta); }
# MAGIC }
# MAGIC </script>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6: Time-Travel to First Version
# MAGIC
# MAGIC Revert the Telco dataset back to its initial state using Delta time-travel.
# MAGIC
# MAGIC + Utilize time-travel capabilities to revert the dataset to its initial version.
# MAGIC
# MAGIC + Display and analyze the first version of the Telco dataset to understand its original structure and content.

# COMMAND ----------

## Extract timestamp of first version (can also be set manually)
timestamp_v0 = spark.sql(<FILL_IN>)
(spark
        .read
        .option(<FILL_IN>)
        .table(<FILL_IN>)
        .printSchema()
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 6: Time-Travel to First Version — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAns6()" style="background:#1976d2; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:600; margin: 8px 0;">Copy to clipboard</button>
# MAGIC
# MAGIC <pre id="copy-ans-6" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC ## Extract timestamp of first version
# MAGIC timestamp_v0 = spark.sql(f"DESCRIBE HISTORY telco_missing_bronze").orderBy("version").first().timestamp
# MAGIC (spark
# MAGIC         .read
# MAGIC         .option("timestampAsOf", timestamp_v0)
# MAGIC         .table("telco_missing_bronze")
# MAGIC         .printSchema()
# MAGIC )
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAns6() {
# MAGIC   const el = document.getElementById("copy-ans-6");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text).then(() => alert("Copied to clipboard")).catch(err => fallbackAns6(text));
# MAGIC   } else { fallbackAns6(text); }
# MAGIC }
# MAGIC function fallbackAns6(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta); ta.select();
# MAGIC   try { document.execCommand("copy"); alert("Copied to clipboard"); }
# MAGIC   catch (e) { alert("Could not copy. Please copy manually."); }
# MAGIC   finally { document.body.removeChild(ta); }
# MAGIC }
# MAGIC </script>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7: Read Previous Versions of the Delta Table
# MAGIC Demonstrate the ability to read data from a specific version of the Delta table.
# MAGIC
# MAGIC + Use SQL to display the full version history of the Delta table.

# COMMAND ----------

## Show table versions
<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Task 7: Read Previous Versions — Solution
# MAGIC <details>
# MAGIC <summary>EXPAND FOR SOLUTION CODE</summary>
# MAGIC <button onclick="copyAns7()" style="background:#1976d2; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:600; margin: 8px 0;">Copy to clipboard</button>
# MAGIC
# MAGIC <pre id="copy-ans-7" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre;">
# MAGIC <code>
# MAGIC %sql
# MAGIC DESCRIBE HISTORY telco_missing_bronze;
# MAGIC </code></pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyAns7() {
# MAGIC   const el = document.getElementById("copy-ans-7");
# MAGIC   if (!el) return;
# MAGIC   const text = el.innerText;
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text).then(() => alert("Copied to clipboard")).catch(err => fallbackAns7(text));
# MAGIC   } else { fallbackAns7(text); }
# MAGIC }
# MAGIC function fallbackAns7(text) {
# MAGIC   const ta = document.createElement("textarea");
# MAGIC   ta.value = text; ta.style.position = "fixed"; ta.style.left = "-9999px";
# MAGIC   document.body.appendChild(ta); ta.select();
# MAGIC   try { document.execCommand("copy"); alert("Copied to clipboard"); }
# MAGIC   catch (e) { alert("Could not copy. Please copy manually."); }
# MAGIC   finally { document.body.removeChild(ta); }
# MAGIC }
# MAGIC </script>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC In this lab, you demonstrated how to explore and manipulate the Telco dataset using Databricks, covering data loading, permissions management, summary statistics, ML-focused visualization with Genie Code, column management, and Delta time-travel capabilities. These foundational skills will carry directly into the data preparation and feature engineering modules ahead.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC