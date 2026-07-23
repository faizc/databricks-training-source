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
# MAGIC
# MAGIC
# MAGIC ## Machine Learning Model Development
# MAGIC
# MAGIC This comprehensive course provides a practical guide to developing machine learning models on Databricks, emphasizing hands-on demonstrations and workflows using popular ML libraries. This course focuses on executing common tasks efficiently with **`MLflow`**, **`Optuna`**, and **`Genie Code`** (Agent Mode). Participants will delve into key topics, including regression and classification models, tracking model training with MLflow, leveraging feature stores for model development, implementing hyperparameter tuning with Optuna, and using Genie Code in Agent Mode for agentic, end-to-end ML workflows — ensuring that participants gain practical, real-world skills for streamlined and effective machine learning model development in the Databricks environment.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC The content was developed for participants with these skills/knowledge/abilities:  
# MAGIC - Knowledge of fundamental concepts of regression and classification methods
# MAGIC - Familiarity with Databricks workspace and notebooks
# MAGIC - Intermediate level knowledge of Python
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Course Agenda  
# MAGIC The following modules are part of the **Machine Learning Model Development** course by **Databricks Academy**.
# MAGIC
# MAGIC | # | Module Name | Lesson Name |
# MAGIC |---|-------------|-------------|
# MAGIC | 1 | **[Model Development Workflow]($./M01 - Model Development Workflow)** | • *Lecture:* Model Development Workflow <br> • [**Demo:** Supervised Learning]($./M01 - Model Development Workflow/1.1a Demo - Supervised Learning) <br> • [**Demo:** Unsupervised Learning]($./M01 - Model Development Workflow/1.1b Demo - Unsupervised Learning) <br> • [**Demo:** Model Tracking with MLflow]($./M01 - Model Development Workflow/1.2 Demo - Model Tracking with MLflow) <br> • [**Lab:** Model Development Tracking with MLflow]($./M01 - Model Development Workflow/1.3 Lab - Model Development Tracking with MLflow) |
# MAGIC | 2 | **[Hyperparameter Tuning]($./M02 - Hyperparameter Tuning)** | • *Lecture:* Hyperparameter Tuning <br> • [**Demo:** Hyperparameter Tuning with Optuna]($./M02 - Hyperparameter Tuning/2.1 Demo - Hyperparameter Tuning with Optuna) <br> • [**Lab:** Hyperparameter Tuning with Optuna]($./M02 - Hyperparameter Tuning/2.2 Lab - Hyperparameter Tuning with Optuna) |
# MAGIC | 3 | **[Agentic Machine Learning]($./M03 - Agentic Machine Learning)** | • *Lecture:* Introduction to Genie Code <br> • [**Demo:** Model Development with Genie Code]($./M03 - Agentic Machine Learning/3.1 Demo - Model Development with Genie Code)  <br> • [**Lab:** Agentic ML with Genie Code]($./M03 - Agentic Machine Learning/3.2 Lab - Agentic ML with Genie Code) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * Use Databricks Runtime version: **`Serverless ML V5`** for running all demo and lab notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>