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
# MAGIC # Demo - Load and Explore Data
# MAGIC
# MAGIC The ability to efficiently handle and explore data is paramount for machine learning projects. In this demo, we'll delve into techniques such as reading from and writing to Delta tables, computing statistics for machine learning insights, and visually exploring data using **Genie Code** for a comprehensive understanding of your datasets.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC
# MAGIC 1. **Load data into a Spark DataFrame**
# MAGIC    - Read a dataset from a shared volume using Spark
# MAGIC
# MAGIC 2. **Explore and understand dataset structure**
# MAGIC    - Inspect schema and preview data
# MAGIC    - Compute summary statistics using built-in Spark functions
# MAGIC
# MAGIC 3. **Use Genie Code to perform ML-focused exploratory analysis**
# MAGIC    - Generate visualizations for target distribution, missing values, and feature relationships
# MAGIC    - Interpret patterns that are useful for machine learning workflows
# MAGIC
# MAGIC 4. **Write data to a Delta table**
# MAGIC    - Persist a DataFrame as a managed Delta table in Unity Catalog
# MAGIC
# MAGIC 5. **Use Delta Lake time travel**
# MAGIC    - Read a previous version of a Delta table
# MAGIC    - Understand how versioning supports reproducibility in data workflows

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
# MAGIC - Provision any supporting configuration needed for this demo
# MAGIC
# MAGIC **NOTE:** The `DA` object is only available in Databricks Academy courses.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-1.1

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
print(f"Datasets Location: {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Dataset
# MAGIC
# MAGIC In this step, we will read the `.csv` file from a shared volume and load it into a Spark DataFrame called `telco_df`.

# COMMAND ----------

# Load dataset with spark
shared_volume_name = 'telco' # From Marketplace
csv_name = 'telco-customer-churn-missing' # CSV file name
dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

schema_string = """
    customerID string,
    gender string,
    SeniorCitizen double,
    Partner string,
    Dependents string,
    tenure double,
    phoneService string,
    MultipleLines string,
    internetService string,
    OnlineSecurity string,
    OnlineBackup string,
    DeviceProtection string,
    TechSupport string,
    StreamingTV string,
    StreamingMovies string,
    Contract string,
    PaperlessBilling string,
    PaymentMethod string,
    MonthlyCharges double,
    TotalCharges double,
    Churn string
"""

telco_df = spark.read.csv(dataset_path, \
                          header=True,
                          schema=schema_string,
                          multiLine=True,
                          escape='"')

display(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore Data with Summary Stats
# MAGIC
# MAGIC While using notebooks, you have various options to view summary statistics for a dataset. Some of these options are:
# MAGIC
# MAGIC * Using Spark DataFrame's built-in method (e.g. `summary()`)
# MAGIC * Using Databricks' built-in data profiler and visualization editor
# MAGIC * Using Genie Code to generate targeted, ML-ready analyses
# MAGIC
# MAGIC In this section we will go over Spark's built-in features for summarizing data. In the next section, we will use Genie Code for ML-focused visualization.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The first and simplest way is using Spark's `summary` function.

# COMMAND ----------

# Display summary statistics with spark
display(telco_df.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The **`display`** function not only facilitates the viewing of detailed dataframes but also serves as a powerful tool for visualizing our dataset according to individual preferences. Whether opting for a Pie Chart or a Bar Chart, this functionality allows for a more insightful exploration, uncovering dependencies and patterns within the dataframe features.

# COMMAND ----------

display(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualization with Genie Code
# MAGIC
# MAGIC In this section, you will use **Genie Code** to perform exploratory analysis of the Telco dataset from a machine learning perspective.
# MAGIC
# MAGIC Instead of writing visualization code manually, you will use natural language prompts to generate charts and analyses that help you:
# MAGIC
# MAGIC - Understand feature distributions  
# MAGIC - Identify missing value patterns  
# MAGIC - Analyze class imbalance  
# MAGIC - Explore relationships between features and the target variable  
# MAGIC
# MAGIC Rather than creating one visualization at a time, you will provide a structured prompt that enables Genie Code to explore the dataset more holistically and generate multiple relevant insights.
# MAGIC
# MAGIC This approach reflects how data scientists interact with AI-assisted workflows in real-world scenarios.
# MAGIC
# MAGIC **Genie Code generates executable code directly in your notebook. You can review, run, and modify the generated code as needed.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Access Genie Code
# MAGIC
# MAGIC To use Genie Code in Databricks:
# MAGIC
# MAGIC 1. Open the **Genie** panel on the right side of the notebook.  
# MAGIC ![Databricks Genie Code](../Includes/images/1.1_Genie_intro.png)
# MAGIC
# MAGIC 2. Select **Agent** from the drop-down (instead of standard Genie mode).  
# MAGIC ![Agent Mode Drop-down](../Includes/images/1.1_Agene_drop_down.png)
# MAGIC
# MAGIC 3. Use prompts such as:
# MAGIC    - "Plot the distribution of the `Churn` column in `telco_df`"
# MAGIC    - "Show a heatmap of missing values in `telco_df`"
# MAGIC    - "Compare `MonthlyCharges` across `Churn` categories using a boxplot"
# MAGIC
# MAGIC Genie Code will generate executable code directly in the notebook, which you can run and refine as needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Best Practices for Genie Code Prompts
# MAGIC
# MAGIC To get reliable and useful results from Genie Code, write prompts that clearly specify:
# MAGIC
# MAGIC - **DataFrame name** — specify which dataset to use (for example, `telco_df`)
# MAGIC - **Column names** — explicitly mention the columns to analyze
# MAGIC - **Chart type** — define the visualization if you have a preference
# MAGIC - **Target column** — identify the label for ML-focused analysis
# MAGIC - **Objective** — clearly state the question you want to answer
# MAGIC
# MAGIC **Example:**
# MAGIC > "Using `telco_pdf`, create a countplot of the `Churn` column to show class balance.  
# MAGIC > Use seaborn or matplotlib. Add a title and axis labels."

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Important Considerations
# MAGIC
# MAGIC <div style="border-left: 4px solid #f57c00; background: #fff3e0; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC     <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC         <span style="font-size: 24px;">⚠️</span>
# MAGIC         <div>
# MAGIC             <strong style="color: #e65100; font-size: 1.1em;">Always Review Generated Code</strong>
# MAGIC             <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC                 Genie Code can accelerate development, but you remain responsible for reviewing the generated code and verifying correctness before running it.
# MAGIC             </p>
# MAGIC         </div>
# MAGIC     </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC     <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC         <span style="font-size: 24px;">🛠️</span>
# MAGIC         <div>
# MAGIC             <strong style="color: #0d47a1; font-size: 1.1em;">Troubleshooting the Genie Panel</strong>
# MAGIC             <p style="margin: 8px 0 0 0; color: #333;">
# MAGIC                 If you do not see the Genie panel, try the following:
# MAGIC             </p>
# MAGIC             <ul style="margin: 8px 0 0 18px; color: #333;">
# MAGIC                 <li>Confirm you are using a supported Databricks Runtime version</li>
# MAGIC                 <li>Ensure Genie Code is enabled in your workspace</li>
# MAGIC                 <li>Refresh your browser or reopen the notebook</li>
# MAGIC             </ul>
# MAGIC             <p style="margin: 10px 0 0 0; color: #333;">
# MAGIC                 <strong>Note:</strong> Your instructor may have already configured the required workspace settings for this course.
# MAGIC             </p>
# MAGIC         </div>
# MAGIC     </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### What to Analyze
# MAGIC
# MAGIC During this analysis, you will use Genie Code to explore the dataset and automatically generate both visualizations and insights.
# MAGIC
# MAGIC Focus on understanding:
# MAGIC
# MAGIC - The distribution of the target variable (`Churn`) and whether it is balanced  
# MAGIC - Which columns contain missing values and may require imputation  
# MAGIC - Relationships between features and the target variable  
# MAGIC - Patterns in the data that may influence model performance  
# MAGIC
# MAGIC
# MAGIC **Use Genie Code to Explore the Dataset**
# MAGIC
# MAGIC **_Use the prompt below with Genie Code to perform a complete exploratory analysis and generate insights:_**
# MAGIC
# MAGIC <button id="copy-genie-btn-task2" onclick="copyGenieTask2()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-genie-task2" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.35; white-space:pre-wrap;">
# MAGIC You are a data analyst working in a Databricks notebook.
# MAGIC
# MAGIC We have a Spark DataFrame called `telco_df`.
# MAGIC
# MAGIC Perform an exploratory data analysis focused on machine learning readiness.
# MAGIC
# MAGIC Steps:
# MAGIC 1. Convert `telco_df` to a pandas DataFrame called `telco_pdf`
# MAGIC
# MAGIC 2. Analyze and visualize:
# MAGIC    - Distribution of the target variable `Churn` (class balance with counts and percentages)
# MAGIC    - Missing value percentage across columns
# MAGIC    - Relationship between key numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) and `Churn`
# MAGIC    - Distribution of important categorical features (`Contract`, `PaymentMethod`, `InternetService`) grouped by `Churn`
# MAGIC
# MAGIC 3. Create appropriate visualizations using matplotlib or seaborn
# MAGIC 4. Add titles and labels to all charts
# MAGIC 5. Keep the code clean and readable
# MAGIC 6. Generate multiple code cells if needed
# MAGIC
# MAGIC After generating the charts, provide a clear summary that answers:
# MAGIC
# MAGIC - Is the dataset balanced or imbalanced?
# MAGIC - Which features appear to influence churn the most?
# MAGIC - What data quality issues should be addressed before modeling?
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyGenieTask2() {
# MAGIC   const textEl = document.getElementById("copy-genie-task2");
# MAGIC   const buttonEl = document.getElementById("copy-genie-btn-task2");
# MAGIC
# MAGIC   if (!textEl || !buttonEl) return;
# MAGIC
# MAGIC   const text = textEl.innerText;
# MAGIC   const originalText = "Copy Genie Prompt";
# MAGIC
# MAGIC   function showCopiedState() {
# MAGIC     buttonEl.innerText = "✅ Copied";
# MAGIC     setTimeout(() => {
# MAGIC       buttonEl.innerText = originalText;
# MAGIC     }, 2000);
# MAGIC   }
# MAGIC
# MAGIC   function fallbackCopy(text) {
# MAGIC     const ta = document.createElement("textarea");
# MAGIC     ta.value = text;
# MAGIC     ta.style.position = "fixed";
# MAGIC     ta.style.left = "-9999px";
# MAGIC     document.body.appendChild(ta);
# MAGIC     ta.focus();
# MAGIC     ta.select();
# MAGIC
# MAGIC     try {
# MAGIC       document.execCommand("copy");
# MAGIC       showCopiedState();
# MAGIC     } catch (err) {
# MAGIC       alert("Could not copy automatically. Please copy manually.");
# MAGIC     } finally {
# MAGIC       document.body.removeChild(ta);
# MAGIC     }
# MAGIC   }
# MAGIC
# MAGIC   if (navigator.clipboard && navigator.clipboard.writeText) {
# MAGIC     navigator.clipboard.writeText(text)
# MAGIC       .then(() => showCopiedState())
# MAGIC       .catch(() => fallbackCopy(text));
# MAGIC   } else {
# MAGIC     fallbackCopy(text);
# MAGIC   }
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Data to Delta Table
# MAGIC
# MAGIC In this step, we will persist the dataset as a **Delta table**.
# MAGIC
# MAGIC Storing data in Delta format enables features such as:
# MAGIC - ACID transactions  
# MAGIC - schema enforcement  
# MAGIC - time travel for versioning  
# MAGIC
# MAGIC We will save the dataset as a managed table in the default catalog and schema.

# COMMAND ----------

# Specify table name
table_name_bronze = "telco_missing_bronze"

# Write DataFrame to Delta table
telco_df.write.saveAsTable(table_name_bronze)

print(f"Table created: {table_name_bronze}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Time Travel with Delta Lake
# MAGIC
# MAGIC In this section, we will explore **Delta Lake time travel**, which allows you to access previous versions of a table.
# MAGIC
# MAGIC This capability enables you to:
# MAGIC - Reproduce past results  
# MAGIC - Audit data changes  
# MAGIC - Recover from unintended updates  
# MAGIC
# MAGIC Time travel is especially valuable in machine learning workflows, where consistent and versioned data is critical for experimentation.
# MAGIC
# MAGIC **Note:** While versioning improves reproducibility, it may introduce challenges in long-running ML workflows if data retention policies limit access to older versions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate an Accidental Change
# MAGIC
# MAGIC To demonstrate time travel, we will simulate a common scenario where columns are unintentionally removed from a table.

# COMMAND ----------

# Drop columns and overwrite table
to_drop_wrong = ["gender", "SeniorCitizen"]
telco_dropped_df = telco_df.drop(*to_drop_wrong)

telco_dropped_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name_bronze)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reverting Changes by Version
# MAGIC
# MAGIC What do we do if we drop a column by mistake?
# MAGIC
# MAGIC With Delta's time-travel feature, we can seamlessly revert to a previous version of our dataset. Let's use the **`DESCRIBE HISTORY`** SQL command to review the history of changes and pinpoint the version where `SeniorCitizen` was still part of our dataset.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY telco_missing_bronze

# COMMAND ----------

# MAGIC %md
# MAGIC Examine the schema to validate whether columns were dropped, specifically verifying if the `gender` and `SeniorCitizen` columns have been removed.

# COMMAND ----------

spark.table(table_name_bronze).printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC You can also use the SQL `DESCRIBE` command to view the table schema.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE `telco_missing_bronze`;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Restore a Previous Version
# MAGIC As we can see in the **`operation`** column in version 1, the table was overwritten, leading to the unintentional removal of columns. To rectify this, we perform a time-travel operation, reverting to version 0, to retrieve the table with all the original columns intact.

# COMMAND ----------

# MAGIC %md
# MAGIC Inspect the schema of the restored DataFrame to verify that the dropped columns have been recovered.

# COMMAND ----------

telco_df_v0 = (
  spark.read
      .option("versionAsOf", 0)
      .table(table_name_bronze)
)

telco_df_v0.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, you explored key techniques for data management and exploratory analysis using Delta Lake and Spark.
# MAGIC
# MAGIC You were able to:
# MAGIC - Load and inspect a dataset using Spark DataFrames  
# MAGIC - Compute summary statistics to understand data characteristics  
# MAGIC - Use **Genie Code** to generate ML-focused visualizations  
# MAGIC - Write data to a Delta table and leverage time travel for versioning  
# MAGIC
# MAGIC These capabilities provide a strong foundation for preparing data and building reliable machine learning workflows.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>
# MAGIC