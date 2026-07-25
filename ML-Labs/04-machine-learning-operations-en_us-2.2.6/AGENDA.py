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
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Machine Learning Operations
# MAGIC
# MAGIC This course will guide participants through a comprehensive exploration of machine learning model operations, focusing on MLOps and model lifecycle management. The initial segment covers essential MLOps components and best practices, providing participants with a strong foundation for effectively operationalizing machine learning models. In the latter part of the course, we will delve into the basics of the model lifecycle, demonstrating how to navigate it seamlessly using the Model Registry in conjunction with the Unity Catalog for efficient model management. By the course's conclusion, participants will have gained practical insights and a well-rounded understanding of MLOps principles, equipped with the skills needed to navigate the intricate landscape of machine learning model operations.
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC The content was developed for participants with these skills/knowledge/abilities:  
# MAGIC - Basic knowledge of traditional machine learning concepts.
# MAGIC - Beginner experience with traditional machine learning development on Databricks.
# MAGIC - Intermediate knowledge of Python for machine learning projects.
# MAGIC - Recommended: Beginner experience with basic **`DevOps`** concepts like **`CI/CD`**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Course Agenda  
# MAGIC The following modules are part of the **Machine Learning Operations** course by **Databricks Academy**.
# MAGIC
# MAGIC | # | Module Name | Lesson Name |
# MAGIC |---|-------------|-------------|
# MAGIC | 1 | [Modern MLOps]($./M01 - Modern MLOps) | • *Lecture:* Defining MLOps <br> • *Lecture:* MLOps on Databricks <br> • [Demo: Setting Up and Managing LakeFlow Jobs using UI]($./M01 - Modern MLOps/1.1 Demo - Setting Up and Managing LakeFlow Jobs using UI) <br> • [Lab: Creating and Managing LakeFlow Jobs using UI]($./M01 - Modern MLOps/1.2 LAB - Creating and Managing LakeFlow Jobs using UI) |
# MAGIC | 2 | [Architecting MLOps Solutions]($./M02 - Architecting MLOps Solutions) | • *Lecture:* Opinionated MLOps Principles <br> • *Lecture:* Recommended MLOps Architectures <br> • [Setup: Generate Tokens]($./M02 - Architecting MLOps Solutions/0 - Generate Tokens) <br> • [Demo: Machine Learning Pipeline with Databricks API]($./M02 - Architecting MLOps Solutions/2.1 Demo - Machine Learning Pipeline with Databricks API) <br> • [Demo: Model Testing Job with the Databricks CLI]($./M02 - Architecting MLOps Solutions/2.2 Demo - Model Testing Job with the Databricks CLI) <br> • [Lab: Deploying Models with Jobs and the Databricks CLI]($./M02 - Architecting MLOps Solutions/2.3 LAB - Deploying Models with Jobs and the Databricks CLI) |
# MAGIC | 3 | [Implementation and Monitoring MLOps Solution]($./M03 - Implementation and Monitoring MLOps Solution) | • *Lecture:* Implementation of MLOps Stacks <br> • *Lecture:* Types of Model Monitoring <br> • *Lecture:* Monitoring in Machine Learning <br> • [Demo: Data Profiling Dashboard]($./M03 - Implementation and Monitoring MLOps Solution/3.1 Demo - Data Profiling Dashboard) <br> • [Lab: Model Monitoring]($./M03 - Implementation and Monitoring MLOps Solution/3.2 Lab - Model Monitoring) |
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * Use Databricks Runtime version: **`17.3.x-cpu-ml-scala2.13`** for running all demo and lab notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC **Instruction:** In case of any issues, please refer to the [Troubleshooting Content Notebook]($./TROUBLESHOOTING_CONTENT).

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>