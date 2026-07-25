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
# MAGIC # Demo: Setting Up and Managing LakeFlow Jobs using UI
# MAGIC
# MAGIC In this demo, we'll set up a Databricks Lakeflow Jobs that automates a series of MLOps tasks such as data quality assessment, feature importance analysis, and alerting on unusual patterns. These tasks help ensure data readiness and provide insights before moving to model training. We’ll also introduce a conditional path based on the detection of unusual patterns.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC In this demo, we will:
# MAGIC
# MAGIC - Create and configure a Databricks Lakeflow job with multiple Python script tasks.
# MAGIC - Set dependencies and conditional paths between tasks.
# MAGIC - Enable email notifications for successful job runs.
# MAGIC - Manually trigger the workflow.
# MAGIC - Monitor the job's execution and completion.

# COMMAND ----------

# MAGIC %md
# MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
# MAGIC Before executing cells in this notebook, please select your classic compute cluster in the lab. Be aware that **Serverless** is enabled by default.
# MAGIC Follow these steps to select the classic compute cluster:
# MAGIC 1. Navigate to the top-right of this notebook and click the drop-down menu to select your cluster. By default, the notebook will use **Serverless**.
# MAGIC 1. If your cluster is available, select it and continue to the next cell. If the cluster is not shown:
# MAGIC    - In the drop-down, select **More**.
# MAGIC    - In the **Attach to an existing compute resource** pop-up, select the first drop-down. You will see a unique cluster name in that drop-down. Please select that cluster.
# MAGIC
# MAGIC **NOTE:** If your cluster has terminated, you might need to restart it in order to select it. To do this:
# MAGIC 1. Right-click on **Compute** in the left navigation pane and select *Open in new tab*.
# MAGIC 1. Find the triangle icon to the right of your compute cluster name and click it.
# MAGIC 1. Wait a few minutes for the cluster to start.
# MAGIC 1. Once the cluster is running, complete the steps above to select your cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC - To run this notebook, you need to use one of the following Databricks runtime(s): `17.3.x-cpu-ml-scala2.13`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Using a Serverless Cluster for This Notebook
# MAGIC
# MAGIC Instructors have the flexibility to use a **Serverless cluster** for this notebook if preferred. Serverless clusters can provide faster startup times and simplified resource management, making them a convenient option for running the notebook efficiently.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Create a Databricks LakeFlow Jobs in the UI
# MAGIC
# MAGIC 1. **Navigate to Jobs & Pipelines**:
# MAGIC    - In your Databricks workspace, click on the **Jobs & Pipelines** icon in the left sidebar.
# MAGIC 2. **Create a New Job**:
# MAGIC    - Click on **Create** in the upper-right corner of the Jobs & Pipelines page and Select **Job**.
# MAGIC    - Name the job **"MLOps Workflow: Data Quality and Feature Analysis"** or something similar for easy identification.
# MAGIC
# MAGIC ![Job_UI](../Includes/images/Job_UI.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Add Tasks to the Job:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.1: Data Quality Assessment
# MAGIC
# MAGIC 1. **Create First Task**:
# MAGIC    - Select `Notebook`.
# MAGIC    - Name the task `Data_Quality_Assessment`.
# MAGIC    - Set **Type** to `Notebook`.
# MAGIC    - **Source** should be set to `Workspace`.
# MAGIC    - Set **Path** to the notebook for data quality assessment (e.g., `/1.1 Demo Pipeline - Data Quality and Feature Analysis/1.1a - Data Quality Assessment`).
# MAGIC    - Select an `Serverless` cluster for this task.
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This task will check for missing values, duplicates, and outliers in the dataset and generate a data quality report.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.2: Alert on Unusual Patterns
# MAGIC
# MAGIC 1. **Create Second Task**:
# MAGIC    - Click on **Add Task --> Notebook**.
# MAGIC    - Name the task `Alert_Unusual_Patterns`.
# MAGIC    - Set **Type** to `Notebook`.
# MAGIC    - **Source** should be set to `Workspace`.
# MAGIC    - Set **Path** to the notebook for alerting on unusual patterns (e.g., `/1.1 Demo Pipeline - Data Quality and Feature Analysis/1.1b - Alert on Unusual Patterns`).
# MAGIC    - Use the same `Serverless` cluster as the previous task.
# MAGIC    - Set **Depends on** to `Data_Quality_Assessment` to ensure this task runs after data quality checks.
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This task will check for unusual patterns, such as high cardinality and skewed distributions, setting a flag if any unusual patterns are detected.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.3: Conditional Path Setup
# MAGIC
# MAGIC 1. **Create Condition for Unusual Patterns**:
# MAGIC    - Click on **Add Task --> If/else condition.**
# MAGIC    - Name the condition task `Alert_Unusual_Patterns_True`.
# MAGIC    - Set **Type** to `If/else condition`.
# MAGIC    - **Condition**: Set the expression to **&lcub;&lcub;tasks.Alert_Unusual_Patterns.values.unusual_pattern_status&rcub;&rcub; == unusual_pattern_detected**.
# MAGIC    - Set **Depends on** to `Alert_Unusual_Patterns` to ensure this condition is evaluated after the unusual patterns check.
# MAGIC    - This condition will branch based on whether unusual patterns are detected (`True`) or not (`False`).
# MAGIC    - Click **Save Task**.
# MAGIC
# MAGIC This conditional setup directs the workflow to either investigate unusual patterns (if detected) or proceed with feature importance analysis if no patterns are detected.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.4: Investigate and Resolve Unusual Patterns and Analyse Feature Importance
# MAGIC
# MAGIC 1. **Create Third Task**:
# MAGIC    - Click on **Add Task --> Notebook**.
# MAGIC    - Name the task `Investigate_Unusual_Patterns`.
# MAGIC    - Set **Type** to `Notebook`.
# MAGIC    - **Source** should be set to `Workspace`.
# MAGIC    - Set **Path** to the notebook for investigating unusual patterns (e.g., `/1.1 Demo Pipeline - Data Quality and Feature Analysis/1.1c -Investigate and Resolve Unusual Patterns`).
# MAGIC    - Use the same cluster as the previous tasks.
# MAGIC    - Set **Depends on** to `Alert_Unusual_Patterns_True(true)`, ensuring it only runs if unusual patterns are detected.
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This task will execute only if unusual patterns are detected (when the condition is `True`).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.5: Feature Importance Analysis (False Path or After Investigation)
# MAGIC
# MAGIC 1. **Create Fourth Task**:
# MAGIC    - Click on **Add Task --> Notebook**.
# MAGIC    - Name the task `Feature_Importance`.
# MAGIC    - Set **Type** to `Notebook`.
# MAGIC    - **Source** should be set to `Workspace`.
# MAGIC    - Set **Path** to the notebook for feature importance analysis (e.g., `/1.1 Demo Pipeline - Data Quality and Feature Analysis/1.1d - Feature Importance Analysis`).
# MAGIC    - Use the same cluster as the previous tasks.
# MAGIC    - Set **Depends on** to:
# MAGIC      - `Alert_Unusual_Patterns_True(false)` (to run if no unusual patterns are detected).
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This task will run only if there are no unusual patterns or after the unusual patterns investigation is completed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.6: Save Report (Final Task)
# MAGIC
# MAGIC 1. **Create Fifth Task**:
# MAGIC    - Click on **Add Task --> Notebook**.
# MAGIC    - Name the task `Save_Report`.
# MAGIC    - Set **Type** to `Notebook`.
# MAGIC    - **Source** should be set to `Workspace`.
# MAGIC    - Set **Path** to the notebook for saving the final report (e.g., `/1.1 Demo Pipeline - Data Quality and Feature Analysis/1.1e - Save Report Notebook (Success Path)`).
# MAGIC    - Use the same cluster as the previous tasks.
# MAGIC    - Set **Depends on** to both:
# MAGIC      - `Feature_Importance` and `Investigate_Unusual_Patterns`.
# MAGIC    - Set **Run if dependencies** to "At least one succeeded" to ensure it saves the report regardless of the path taken.
# MAGIC    - Click **Create Task**.
# MAGIC
# MAGIC This task will save the final report once all prior steps are successfully completed, regardless of whether unusual patterns were detected.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2.7: Enable Email Notifications
# MAGIC
# MAGIC 1. **Set up Notifications**:
# MAGIC    - In the job's configuration, navigate to the **Job Notifications** section.
# MAGIC    - Enable email notifications by adding your email to receive updates on job completion.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Trigger the Job Manually
# MAGIC
# MAGIC 1. **Run the Job**:
# MAGIC    - Go to the job in the Databricks UI and click on **Run Now** in the top-right corner to manually trigger the job. This will execute all tasks in the Job according to their dependencies and conditions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Monitor the Job Execution
# MAGIC
# MAGIC 1. **Navigate to the Runs Tab**:
# MAGIC    - In the job interface, go to the **Runs** tab to view active and completed executions of the job.
# MAGIC
# MAGIC 2. **Observe Task Execution**:
# MAGIC    - Each task’s status is displayed in the **Runs** tab, where you can see which tasks are currently executing or have completed.
# MAGIC    - Click on each task to view its execution details and outputs, allowing you to troubleshoot and verify each stage.
# MAGIC    - Check the logs to see if the Job followed the correct path based on the unusual pattern detection condition.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, you learned how to:
# MAGIC - Configure and execute a Databricks LakeFlow Jobs with multiple tasks for data quality, feature importance, and unusual pattern checks.
# MAGIC - Use dependencies and conditional paths to control the flow of tasks based on the data conditions.
# MAGIC - Set up email notifications to stay updated on job execution.
# MAGIC - Trigger the Job manually and monitor its execution.
# MAGIC
# MAGIC This workflow serves as a preliminary step to ensure data quality and feature insights before moving on to model training. By automating these MLOps setup tasks and handling conditional paths, you can ensure a robust pipeline that adapts based on data characteristics, providing insights and addressing issues early in the MLOps process.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>