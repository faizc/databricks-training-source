# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC        alt="Databricks Learning">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB - Agentic ML with Genie Code
# MAGIC
# MAGIC In this lab, you will independently build a bank personal loan prediction classifier using **Genie Code** in **Agent Mode**. Each task asks you to write a structured prompt, run it in Genie Code, review the generated code, and validate the output before moving on. The workflow deliberately extends the demo - you will apply the same Genie Code pattern to a new dataset and complete three skills the demo did not cover: business-driven feature engineering, multi-model comparison, and decision threshold optimization.
# MAGIC
# MAGIC 📌 **Note:** In the demo, a single Random Forest model was trained on a customer churn dataset. In this lab, you will engineer features from business reasoning, compare two models using the MLflow API, optimize the decision threshold using the precision-recall curve, and deploy the selected model to a real-time Databricks serving endpoint.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC - **Task 1:** Load the bank loan dataset from the Feature Store and perform EDA — class distribution, null check, and correlation heatmap.
# MAGIC
# MAGIC - **Task 2:** Apply business-driven feature engineering — remove a redundant feature identified during EDA and create a meaningful derived feature.
# MAGIC
# MAGIC - **Task 3:** Train and compare two classification models (Logistic Regression and Random Forest) in the same MLflow experiment and select the best performer programmatically.
# MAGIC
# MAGIC - **Task 4:** Evaluate the selected model and optimize the decision threshold using a precision-recall curve to maximize F1-score.
# MAGIC
# MAGIC - **Task 5:** Register the selected model to Unity Catalog with a `champion` alias and a model description.
# MAGIC
# MAGIC - **Task 6:** Deploy the registered model to a real-time Databricks serving endpoint and verify it with a live inference request.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## REQUIRED - SELECT A COMPUTE ENVIRONMENT
# MAGIC
# MAGIC <div style="border-left: 4px solid #F44336; background: #FFEBEE; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC   <div>
# MAGIC     <strong style="color: #C62828; font-size: 1.1em;">Select Compute</strong>
# MAGIC     <p style="margin: 8px 0 0 0; color: #333;">Before starting, select the required compute environment listed below.</p>
# MAGIC     <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC       <li><strong>Serverless Compute, Version 5</strong> — <a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976D2; text-decoration: underline;">How to select an environment version</a></li>
# MAGIC     </ul>
# MAGIC     <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V5</strong>.</p>
# MAGIC   </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;"><div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">  <img src="../Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">  <div style="font-size: 15pt; color: #0B2026; line-height: 1.7; margin-bottom: 16px;">    Starting the agentic ML lab on bank loan prediction? Ask Genie Code to orient yourself before you begin. Click on the genie icon <img src="../Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.  </div>  <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0B2026;">    <span id="genie-query-lab" style="flex: 1;">I am working on a bank personal loan prediction ML lab in Databricks. I need to compare two classification models (Logistic Regression and Random Forest), optimize the decision threshold, and deploy the best model to a serving endpoint. What workflow would you recommend?</span>    <button onclick="      var text = document.getElementById('genie-query-lab').innerText;      var ta = document.createElement('textarea');      ta.value = text;      ta.style.position = 'fixed';      ta.style.opacity = '0';      document.body.appendChild(ta);      ta.select();      document.execCommand('copy');      document.body.removeChild(ta);      this.innerText = 'Copied!';      var btn = this;      setTimeout(function(){ btn.innerText = 'Copy'; }, 2000);    " style="background: #FF5F46; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-size: 13pt; cursor: pointer; white-space: nowrap;">Copy</button>  </div></div></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## How to Use This Lab
# MAGIC
# MAGIC Each task follows the same pattern:
# MAGIC
# MAGIC 1. **Read the objective** — understand the goal and how it extends the demo
# MAGIC 2. **Study the prompt** — each prompt is structured with context, task, and expected output
# MAGIC 3. **Copy and paste into Genie Code Agent Mode** — use the copy button, paste, press Enter
# MAGIC 4. **Review the generated code** — read it before running; refine the prompt if needed
# MAGIC 5. **Run the cell** — execute and observe output
# MAGIC 6. **Validate** — check the criteria before moving on
# MAGIC
# MAGIC > 💡 **Prompts are starting points.** Try variations — observe how specificity changes the output. This is a key skill for agentic ML workflows.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.2

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Dataset Overview
# MAGIC
# MAGIC **Task:** Predict whether a bank customer will accept a personal loan offer.
# MAGIC **Target:** `Personal_Loan` (1 = accepted, 0 = declined) — already encoded as binary.
# MAGIC
# MAGIC **Tables available:**
# MAGIC - `bank_loan`: Age, Experience, Income, ZIP_Code, Family, CCAvg, Education, Mortgage, Securities_Account, CD_Account, Online, CreditCard, Personal_Loan
# MAGIC - `bank_loan_features`: Engineered features from the Feature Store — Loan_to_Income_Ratio, Monthly_Income
# MAGIC
# MAGIC > 💡 **Important note:** You will discover in Task 2 that `Age` and `Experience` are nearly perfectly correlated (r ≈ 0.99) — a real-world data quality issue that requires a deliberate engineering decision.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## How to Access Genie Code (Agent Mode)
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 10px 0;">
# MAGIC   <strong style="color: #0d47a1;">Step-by-step:</strong>
# MAGIC   <ol style="margin: 8px 0 0 18px; color: #333; line-height: 1.8;">
# MAGIC     <li>Click the <strong>Genie Code</strong> icon on the right-hand panel (AI wand or chat bubble).</li>
# MAGIC     <li>Switch the mode drop-down to <strong>Agent</strong> (not the default mode).</li>
# MAGIC     <li>Genie Code can now read notebook variables, insert cells, and reason across steps.</li>
# MAGIC     <li>For each task: click <strong>Copy Genie Prompt</strong>, paste, press Enter.</li>
# MAGIC     <li>Review the generated code before running it.</li>
# MAGIC   </ol>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Custom Instructions for This Lab
# MAGIC
# MAGIC The classroom setup cell above automatically wrote a `.assistant_instructions.md` file to this notebook's workspace folder. Genie Code reads this file when the Genie Code panel opens, so it is already grounded in the lab context before you send the first prompt.
# MAGIC
# MAGIC <div style="border-left: 4px solid #2e7d32; background: #e8f5e9; padding: 14px 18px; border-radius: 4px; margin: 10px 0;">
# MAGIC   <strong style="color: #1b5e20;">✅ Auto-configured — no action needed</strong>
# MAGIC   <p style="margin: 6px 0 0 0; color: #333;">The file was written by the setup script and will be loaded automatically each time you open Genie Code in this notebook. Re-running classroom setup rewrites it fresh (create or replace).</p>
# MAGIC </div>
# MAGIC
# MAGIC <p style="margin: 14px 0 6px 0; font-weight: 600; color: #263238;">Contents of <code>.assistant_instructions.md</code> for this lab:</p>
# MAGIC
# MAGIC <div style="background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px 20px; font-family: ui-monospace, monospace; font-size: 0.88rem; line-height: 1.6; margin: 0;">
# MAGIC You are assisting with a binary classification task to predict bank personal loan acceptance.<br><br>
# MAGIC Dataset: bank_loan (joined with bank_loan_features via ID)<br>
# MAGIC Join key: ID (left join from bank_loan to bank_loan_features)<br>
# MAGIC Target column: Personal_Loan (1 = accepted, 0 = declined) — already encoded as binary<br>
# MAGIC Catalog: {DA.catalog_name}<br>
# MAGIC Schema: {DA.schema_name}<br>
# MAGIC MLflow experiment: /Users/{DA.username}/loan_model_comparison<br>
# MAGIC Modeling library: scikit-learn (Pipeline-based preprocessing preferred)<br>
# MAGIC All experiments must be logged to MLflow with explicit parameter and metric logging.<br>
# MAGIC Register final models to Unity Catalog using mlflow.set_registry_uri("databricks-uc").<br>
# MAGIC Lab tasks: business-driven feature engineering, multi-model comparison (Logistic Regression vs Random Forest), decision threshold optimization using precision-recall curve, and deployment to a real-time Databricks serving endpoint.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Task 1: Load Data and Perform Exploratory Data Analysis
# MAGIC
# MAGIC ### Objective
# MAGIC Load the bank loan dataset from both tables, join them, and explore the data. Pay particular attention to the class distribution and the correlation between `Age` and `Experience` — this will inform your feature engineering decisions in Task 2.
# MAGIC
# MAGIC ### What to accomplish:
# MAGIC - Load `bank_loan` + `bank_loan_features` → join on `ID` → `loan_df` / `loan_pdf`
# MAGIC - Check class distribution of `Personal_Loan`
# MAGIC - Identify missing values
# MAGIC - Plot a correlation heatmap — specifically note the Age/Experience correlation
# MAGIC
# MAGIC > 💡 **Notice in the heatmap:** `Age` and `Experience` will show near-perfect correlation. This is a common real-world data quality issue — two features measuring the same underlying signal. You'll make a deliberate decision about this in Task 2.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t1-load" onclick="copyT1Load()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t1-load" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: I am working in a Databricks notebook on a bank personal loan prediction task.
# MAGIC Two tables have been set up and are accessible via DA.catalog_name and DA.schema_name.
# MAGIC
# MAGIC Task: Load and join the two tables into a single modeling DataFrame.
# MAGIC
# MAGIC Tables:
# MAGIC - bank_loan: main table with customer demographics, financial features, and the Personal_Loan target
# MAGIC - bank_loan_features: Feature Store table with Loan_to_Income_Ratio and Monthly_Income
# MAGIC - Join key: ID (left join from bank_loan to bank_loan_features)
# MAGIC
# MAGIC Requirements:
# MAGIC - Use spark.table() with f"{DA.catalog_name}.{DA.schema_name}.bank_loan" format
# MAGIC - Store the joined Spark DataFrame as loan_df
# MAGIC - Convert to pandas → store as loan_pdf
# MAGIC - Print: row count, column count, schema
# MAGIC - Display the first 10 rows
# MAGIC
# MAGIC Expected output: loan_pdf with all columns including Personal_Loan target.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT1Load() {
# MAGIC   const text = document.getElementById("prompt-t1-load").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t1-load");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 1: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# Confirm loan_pdf has Personal_Loan, bank_loan_features columns (Loan_to_Income_Ratio, Monthly_Income).

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 1. Load Data
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC
# MAGIC # Initialize Feature Engineering client
# MAGIC fe = FeatureEngineeringClient()
# MAGIC
# MAGIC # Load the main loan table
# MAGIC loan_base_df = spark.table(f"{DA.catalog_name}.{DA.schema_name}.bank_loan")
# MAGIC
# MAGIC # Load the feature store table
# MAGIC loan_features_df = spark.table(f"{DA.catalog_name}.{DA.schema_name}.bank_loan_features")
# MAGIC
# MAGIC # Join on ID
# MAGIC loan_df = loan_base_df.join(loan_features_df, on="ID", how="left")
# MAGIC
# MAGIC # Convert to pandas
# MAGIC loan_pdf = loan_df.toPandas()
# MAGIC
# MAGIC print(f"Dataset: {loan_df.count()} rows, {len(loan_df.columns)} columns")
# MAGIC print("\nSchema:")
# MAGIC loan_df.printSchema()
# MAGIC display(loan_df.limit(10))
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Task 1 — EDA: Explore the Dataset

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t1-eda" onclick="copyT1EDA()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t1-eda" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: I have a pandas DataFrame called loan_pdf loaded in this notebook.
# MAGIC Task: Binary classification — predict Personal_Loan (1 = accepted, 0 = declined).
# MAGIC
# MAGIC Task: Perform a comprehensive exploratory data analysis in a single cell.
# MAGIC
# MAGIC Output 1 — Class distribution of Personal_Loan:
# MAGIC - Count and percentage for each class
# MAGIC - Bar chart: green for class 0 (Declined), blue for class 1 (Accepted)
# MAGIC - Print: is the dataset balanced or imbalanced?
# MAGIC
# MAGIC Output 2 — Missing values:
# MAGIC - Null count and percentage per column (sorted descending)
# MAGIC - Only show columns with at least one null; otherwise print: "✅ No missing values."
# MAGIC
# MAGIC Output 3 — Correlation heatmap:
# MAGIC - All numeric columns including Personal_Loan as target
# MAGIC - seaborn heatmap, annotated values (fmt='.2f'), coolwarm colormap
# MAGIC - Title: "Feature Correlation — Bank Personal Loan Dataset"
# MAGIC - After the heatmap, print the top 5 features by absolute correlation with Personal_Loan
# MAGIC - IMPORTANT: Specifically call out the correlation between 'Age' and 'Experience'
# MAGIC
# MAGIC Expected: Three outputs that reveal class imbalance level, data quality, and which features matter most.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT1EDA() {
# MAGIC   const text = document.getElementById("prompt-t1-eda").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t1-eda");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 1: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# Note the Age-Experience correlation from the heatmap — you will address it in Task 2.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 1. EDA — Class Distribution
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC class_counts = loan_pdf['Personal_Loan'].value_counts()
# MAGIC class_pct = loan_pdf['Personal_Loan'].value_counts(normalize=True).mul(100).round(1)
# MAGIC
# MAGIC dist_summary = pd.DataFrame({'Count': class_counts, 'Percentage (%)': class_pct})
# MAGIC print("Personal_Loan Class Distribution:")
# MAGIC print(dist_summary.to_string())
# MAGIC
# MAGIC fig, ax = plt.subplots(figsize=(5, 4))
# MAGIC bars = ax.bar(
# MAGIC     ['Declined (0)', 'Accepted (1)'],
# MAGIC     class_counts.sort_index().values,
# MAGIC     color=['#42A5F5', '#EF5350'], edgecolor='white', width=0.5
# MAGIC )
# MAGIC ax.bar_label(bars, labels=[f"{v:,}\n({p}%)" for v, p in zip(class_counts.sort_index().values, class_pct.sort_index().values)], padding=4, fontsize=11)
# MAGIC ax.set_title("Personal Loan Class Distribution", fontsize=13, pad=10)
# MAGIC ax.set_ylabel("Customer Count", fontsize=11)
# MAGIC ax.set_ylim(0, class_counts.max() * 1.2)
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 1. EDA — Missing Values
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC null_counts = loan_pdf.isnull().sum()
# MAGIC null_pct = (null_counts / len(loan_pdf) * 100).round(2)
# MAGIC
# MAGIC null_summary = pd.DataFrame({
# MAGIC     'Column': null_counts.index,
# MAGIC     'Null Count': null_counts.values,
# MAGIC     'Null Percentage (%)': null_pct.values
# MAGIC }).sort_values('Null Count', ascending=False)
# MAGIC
# MAGIC null_filtered = null_summary[null_summary['Null Count'] > 0].reset_index(drop=True)
# MAGIC if null_filtered.empty:
# MAGIC     print("✅ No missing values found in the dataset.")
# MAGIC else:
# MAGIC     print(f"Columns with missing values ({len(null_filtered)} found):")
# MAGIC     print(null_filtered.to_string(index=False))
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 1. EDA — Correlation Heatmap
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import seaborn as sns
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC loan_corr_pdf = loan_pdf.copy()
# MAGIC # Personal_Loan is already numeric (0/1), just ensure int
# MAGIC loan_corr_pdf['Personal_Loan'] = loan_corr_pdf['Personal_Loan'].astype(int)
# MAGIC
# MAGIC numeric_cols = loan_corr_pdf.select_dtypes(include=['number']).columns.tolist()
# MAGIC # Exclude ID from correlation (not a feature)
# MAGIC numeric_cols = [c for c in numeric_cols if c != 'ID']
# MAGIC
# MAGIC corr_matrix = loan_corr_pdf[numeric_cols].corr()
# MAGIC
# MAGIC fig, ax = plt.subplots(figsize=(10, 8))
# MAGIC sns.heatmap(
# MAGIC     corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
# MAGIC     center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8}
# MAGIC )
# MAGIC ax.set_title('Feature Correlation Heatmap\n(Personal_Loan = target)', fontsize=13, pad=12)
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC
# MAGIC loan_corrs = corr_matrix['Personal_Loan'].drop('Personal_Loan').abs().sort_values(ascending=False)
# MAGIC print("\nTop features by correlation with Personal_Loan:")
# MAGIC print(loan_corrs.to_string())
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC > ✅ **Validate before moving on:**
# MAGIC - `loan_pdf` is loaded with all expected columns
# MAGIC - Class imbalance confirmed: ~91% Declined, ~9% Accepted
# MAGIC - Age and Experience show very high correlation (r > 0.95) — noted for Task 2
# MAGIC - Top predictive features identified from heatmap (Income, CCAvg, CD_Account likely top 3)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Task 2: Business-Driven Feature Engineering
# MAGIC
# MAGIC ### Objective
# MAGIC Apply deliberate, business-informed feature engineering — not just mechanical preprocessing. This task introduces two decisions that the demo's standard ColumnTransformer approach does not address:
# MAGIC
# MAGIC 1. **Feature redundancy:** `Age` and `Experience` are nearly perfectly correlated. Keeping both adds noise without information — you will drop `Experience`.
# MAGIC 2. **Derived feature:** `Income` alone understates affordability for larger families. You will create `Income_per_FamilyMember = Income / Family` to capture per-capita financial capacity — a stronger predictor of loan acceptance.
# MAGIC
# MAGIC ### What to accomplish:
# MAGIC - Drop `Experience` (redundant with `Age`)
# MAGIC - Create a new derived feature: `Income_per_FamilyMember`
# MAGIC - Build a `ColumnTransformer` preprocessing pipeline
# MAGIC - Stratified 80/20 train/test split
# MAGIC
# MAGIC > 💡 **This is real ML thinking.** Adding domain knowledge to feature engineering consistently outperforms applying generic transformations to all raw features.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t2" onclick="copyT2()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t2" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: I have loan_pdf in memory. The task is binary classification to predict Personal_Loan.
# MAGIC From EDA, I found that Age and Experience are nearly perfectly correlated (~0.99) — Experience is redundant.
# MAGIC I also want to create a derived feature that captures per-capita financial capacity.
# MAGIC
# MAGIC Task: Perform business-driven feature engineering and prepare splits for model training.
# MAGIC
# MAGIC Step 1 — Feature reduction and derivation:
# MAGIC - Define y = loan_pdf['Personal_Loan'] (already 0/1)
# MAGIC - Define X = loan_pdf.drop(columns=['ID', 'Personal_Loan'])
# MAGIC - Drop 'Experience' from X (redundant with Age — confirmed by EDA)
# MAGIC - Add derived feature: X['Income_per_FamilyMember'] = X['Income'] / X['Family']
# MAGIC - Print the final feature list and the first 5 values of Income_per_FamilyMember
# MAGIC
# MAGIC Step 2 — Preprocessing pipeline:
# MAGIC - Identify categorical_cols (object/category dtype) and numerical_cols (numeric dtype)
# MAGIC - Build ColumnTransformer 'preprocessor':
# MAGIC     * StandardScaler() for numerical_cols
# MAGIC     * OneHotEncoder(handle_unknown='ignore', sparse_output=False) for categorical_cols
# MAGIC     * remainder='drop'
# MAGIC - Print the feature counts for each type
# MAGIC
# MAGIC Step 3 — Stratified split:
# MAGIC - 80/20 split, random_state=42, stratified on y
# MAGIC - X_train, X_test, y_train, y_test
# MAGIC - Print shapes and class balance in both splits
# MAGIC
# MAGIC Expected: Feature matrix X with 'Experience' dropped and 'Income_per_FamilyMember' added, plus four clean train/test splits.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT2() {
# MAGIC   const text = document.getElementById("prompt-t2").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t2");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 2: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# Confirm: Experience is not in X.columns. Income_per_FamilyMember is in X.columns.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 2. Business-Driven Feature Engineering
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC from sklearn.compose import ColumnTransformer
# MAGIC from sklearn.preprocessing import StandardScaler, OneHotEncoder
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC
# MAGIC # Step 1 — Feature reduction + derived feature
# MAGIC y = loan_pdf['Personal_Loan']
# MAGIC X = loan_pdf.drop(columns=['ID', 'Personal_Loan'])
# MAGIC X = X.drop(columns=['Experience'])                             # drop redundant feature
# MAGIC X = X.copy()
# MAGIC X['Income_per_FamilyMember'] = X['Income'] / X['Family']      # derived feature
# MAGIC
# MAGIC print(f"Features ({len(X.columns)}): {X.columns.tolist()}")
# MAGIC print(f"\nIncome_per_FamilyMember sample (first 5):")
# MAGIC print(X['Income_per_FamilyMember'].head().to_string())
# MAGIC
# MAGIC # Step 2 — Preprocessing pipeline
# MAGIC categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
# MAGIC numerical_cols   = X.select_dtypes(include=['number']).columns.tolist()
# MAGIC
# MAGIC print(f"\nCategorical ({len(categorical_cols)}): {categorical_cols}")
# MAGIC print(f"Numerical   ({len(numerical_cols)}): {numerical_cols}")
# MAGIC
# MAGIC preprocessor = ColumnTransformer(transformers=[
# MAGIC     ('num', StandardScaler(), numerical_cols),
# MAGIC     ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
# MAGIC ], remainder='drop')
# MAGIC
# MAGIC # Step 3 — Stratified split
# MAGIC X_train, X_test, y_train, y_test = train_test_split(
# MAGIC     X, y, test_size=0.2, random_state=42, stratify=y
# MAGIC )
# MAGIC print(f"\nTrain: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
# MAGIC print(f"Train class balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")
# MAGIC print(f"Test  class balance: {y_test.value_counts(normalize=True).round(3).to_dict()}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC > ✅ **Validate before moving on:**
# MAGIC - `Experience` is NOT in `X.columns`
# MAGIC - `Income_per_FamilyMember` IS in `X.columns` with non-null values
# MAGIC - `preprocessor` is a `ColumnTransformer` (not yet fitted)
# MAGIC - Class balance is ~91%/9% in both splits — consistent with original dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Task 3: Multi-Model Comparison
# MAGIC
# MAGIC ### Objective
# MAGIC Train two classification models and compare them using MLflow. In the demo, a single Random Forest was trained. Here you will train both a **Logistic Regression baseline** and a **Random Forest**, log them to the same MLflow experiment, and use `mlflow.search_runs()` to build a data-driven comparison.
# MAGIC
# MAGIC This mirrors real ML workflows: you always establish a baseline before claiming a more complex model is "better."
# MAGIC
# MAGIC ### What to accomplish:
# MAGIC - Train Logistic Regression (simple, interpretable, interpretable baseline)
# MAGIC - Train Random Forest (non-linear, typically stronger on tabular data)
# MAGIC - Log both to the same MLflow experiment with identical metrics
# MAGIC - Use MLflow API to compare results and identify the better model
# MAGIC
# MAGIC > 💡 **MLflow as a comparison tool.** The `mlflow.search_runs()` function returns all runs in an experiment as a pandas DataFrame — enabling programmatic comparison. This is a Databricks-native pattern for multi-model experiments.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Task 3a — Train Logistic Regression (Baseline)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t3-lr" onclick="copyT3LR()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t3-lr" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: X_train, X_test, y_train, y_test, and preprocessor are defined.
# MAGIC Binary classification, class-imbalanced (~91% negative, ~9% positive).
# MAGIC
# MAGIC Task: Train a Logistic Regression classifier as a baseline model, logged to MLflow.
# MAGIC
# MAGIC Requirements:
# MAGIC - Build a Pipeline: ('preprocessor', preprocessor) + ('classifier', LogisticRegression)
# MAGIC - Use class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42
# MAGIC - MLflow:
# MAGIC     * Set experiment: mlflow.set_experiment(f"/Users/{DA.username}/loan_model_comparison")
# MAGIC     * Run name: 'lr_baseline'
# MAGIC     * Enable autologging: mlflow.sklearn.autolog(silent=True)
# MAGIC     * Explicitly log: C=1.0, max_iter=1000, solver='lbfgs', class_weight='balanced' as parameters
# MAGIC     * Compute and log: train_accuracy, test_accuracy, and test_roc_auc as metrics
# MAGIC     * Log pipeline artifact as 'loan_lr_model'
# MAGIC     * Store run.info.run_id as lr_run_id
# MAGIC - After training, print: lr_run_id, test_accuracy, test_roc_auc
# MAGIC
# MAGIC Expected: lr_pipeline trained, lr_run_id stored, first MLflow run in 'loan_model_comparison' experiment.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT3LR() {
# MAGIC   const text = document.getElementById("prompt-t3-lr").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t3-lr");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 3: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# After running: navigate to Experiments → loan_model_comparison. Confirm the lr_baseline run appears.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 3a. Train Logistic Regression (Baseline)
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC import mlflow.sklearn
# MAGIC from sklearn.linear_model import LogisticRegression
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.metrics import roc_auc_score
# MAGIC
# MAGIC mlflow.set_experiment(f"/Users/{DA.username}/loan_model_comparison")
# MAGIC mlflow.sklearn.autolog(log_input_examples=False, log_model_signatures=True, silent=True)
# MAGIC
# MAGIC with mlflow.start_run(run_name="lr_baseline") as run:
# MAGIC     lr_pipeline = Pipeline(steps=[
# MAGIC         ('preprocessor', preprocessor),
# MAGIC         ('classifier',   LogisticRegression(
# MAGIC             C=1.0, max_iter=1000, solver='lbfgs',
# MAGIC             class_weight='balanced', random_state=42
# MAGIC         ))
# MAGIC     ])
# MAGIC     lr_pipeline.fit(X_train, y_train)
# MAGIC
# MAGIC     train_acc = lr_pipeline.score(X_train, y_train)
# MAGIC     test_acc  = lr_pipeline.score(X_test,  y_test)
# MAGIC     test_auc  = roc_auc_score(y_test, lr_pipeline.predict_proba(X_test)[:, 1])
# MAGIC
# MAGIC     mlflow.log_params({"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "class_weight": "balanced"})
# MAGIC     mlflow.log_metrics({"train_accuracy": train_acc, "test_accuracy": test_acc, "test_roc_auc": test_auc})
# MAGIC     mlflow.sklearn.log_model(lr_pipeline, artifact_path="loan_lr_model")
# MAGIC     lr_run_id = run.info.run_id
# MAGIC
# MAGIC print(f"LR Run ID:        {lr_run_id}")
# MAGIC print(f"LR Test Accuracy: {test_acc:.4f}")
# MAGIC print(f"LR ROC-AUC:       {test_auc:.4f}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Task 3b — Train Random Forest

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t3-rf" onclick="copyT3RF()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t3-rf" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: X_train, X_test, y_train, y_test, and preprocessor are defined.
# MAGIC lr_run_id is stored. The MLflow experiment 'loan_model_comparison' already has the LR baseline run.
# MAGIC
# MAGIC Task: Train a Random Forest classifier in the same experiment for comparison.
# MAGIC
# MAGIC Requirements:
# MAGIC - Build a Pipeline: ('preprocessor', preprocessor) + ('classifier', RandomForestClassifier)
# MAGIC - Use n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
# MAGIC - MLflow:
# MAGIC     * Same experiment: /Users/{DA.username}/loan_model_comparison
# MAGIC     * Run name: 'rf_model'
# MAGIC     * Enable autologging: mlflow.sklearn.autolog(silent=True)
# MAGIC     * Explicitly log: n_estimators=100, max_depth=10, class_weight='balanced' as parameters
# MAGIC     * Compute and log: train_accuracy, test_accuracy, and test_roc_auc as metrics
# MAGIC     * Log pipeline artifact as 'loan_rf_model'
# MAGIC     * Store run.info.run_id as rf_run_id
# MAGIC - After training, print: rf_run_id, test_accuracy, test_roc_auc
# MAGIC
# MAGIC Expected: rf_pipeline trained, rf_run_id stored, second run added to 'loan_model_comparison' experiment.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT3RF() {
# MAGIC   const text = document.getElementById("prompt-t3-rf").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t3-rf");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 3: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# After running: confirm both lr_baseline and rf_model runs appear in Experiments → loan_model_comparison.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 3b. Train Random Forest
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.ensemble import RandomForestClassifier
# MAGIC
# MAGIC with mlflow.start_run(run_name="rf_model") as run:
# MAGIC     rf_pipeline = Pipeline(steps=[
# MAGIC         ('preprocessor', preprocessor),
# MAGIC         ('classifier',   RandomForestClassifier(
# MAGIC             n_estimators=100, max_depth=10,
# MAGIC             class_weight='balanced', random_state=42
# MAGIC         ))
# MAGIC     ])
# MAGIC     rf_pipeline.fit(X_train, y_train)
# MAGIC
# MAGIC     train_acc = rf_pipeline.score(X_train, y_train)
# MAGIC     test_acc  = rf_pipeline.score(X_test,  y_test)
# MAGIC     test_auc  = roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1])
# MAGIC
# MAGIC     mlflow.log_params({"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"})
# MAGIC     mlflow.log_metrics({"train_accuracy": train_acc, "test_accuracy": test_acc, "test_roc_auc": test_auc})
# MAGIC     mlflow.sklearn.log_model(rf_pipeline, artifact_path="loan_rf_model")
# MAGIC     rf_run_id = run.info.run_id
# MAGIC
# MAGIC print(f"RF Run ID:        {rf_run_id}")
# MAGIC print(f"RF Test Accuracy: {test_acc:.4f}")
# MAGIC print(f"RF ROC-AUC:       {test_auc:.4f}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Task 3c — Compare Models via MLflow API

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t3-compare" onclick="copyT3Compare()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t3-compare" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: Both lr_run_id and rf_run_id are stored. Both used experiment '/Users/{DA.username}/loan_model_comparison'.
# MAGIC
# MAGIC Task: Compare the two trained models using the MLflow Runs API.
# MAGIC
# MAGIC Requirements:
# MAGIC - Retrieve runs: mlflow.search_runs(experiment_names=[f"/Users/{DA.username}/loan_model_comparison"])
# MAGIC - Filter to only our two runs (run names: 'lr_baseline', 'rf_model')
# MAGIC - Build a clean comparison table with columns: Model, Test Accuracy, ROC-AUC, Run ID
# MAGIC - Create a side-by-side bar chart comparing test_roc_auc for both models (use matplotlib)
# MAGIC - Print: which model has higher ROC-AUC?
# MAGIC - Store the name of the better model as best_model_name (either 'lr_baseline' or 'rf_model')
# MAGIC - Store the corresponding pipeline as best_pipeline (either lr_pipeline or rf_pipeline)
# MAGIC
# MAGIC Expected: A comparison table, a bar chart, and best_pipeline set to the winning model.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT3Compare() {
# MAGIC   const text = document.getElementById("prompt-t3-compare").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t3-compare");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 3: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# Confirm best_pipeline is set. Note which model won on ROC-AUC and why you think that is.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 3c. Compare Models via MLflow API
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC # Retrieve both runs from the experiment
# MAGIC runs_df = mlflow.search_runs(
# MAGIC     experiment_names=[f"/Users/{DA.username}/loan_model_comparison"]
# MAGIC )
# MAGIC
# MAGIC # Filter to our two runs
# MAGIC our_runs = runs_df[runs_df['tags.mlflow.runName'].isin(['lr_baseline', 'rf_model'])].copy()
# MAGIC our_runs = our_runs[['tags.mlflow.runName', 'metrics.test_accuracy', 'metrics.test_roc_auc', 'run_id']]
# MAGIC our_runs.columns = ['Model', 'Test Accuracy', 'ROC-AUC', 'Run ID']
# MAGIC our_runs = our_runs.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
# MAGIC
# MAGIC print("Model Comparison:")
# MAGIC print(our_runs[['Model', 'Test Accuracy', 'ROC-AUC']].to_string(index=False))
# MAGIC
# MAGIC # Bar chart
# MAGIC fig, ax = plt.subplots(figsize=(7, 4))
# MAGIC colors = ['#2574B5', '#1B5162']
# MAGIC bars = ax.bar(our_runs['Model'], our_runs['ROC-AUC'], color=colors, width=0.4)
# MAGIC ax.bar_label(bars, labels=[f"{v:.4f}" for v in our_runs['ROC-AUC']], padding=3, fontsize=11)
# MAGIC ax.set_ylim(0, 1.1)
# MAGIC ax.set_title("Model Comparison — ROC-AUC on Test Set", fontsize=13, pad=10)
# MAGIC ax.set_ylabel("ROC-AUC")
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC
# MAGIC # Select best model
# MAGIC best_model_name = our_runs.iloc[0]['Model']
# MAGIC best_pipeline = rf_pipeline if best_model_name == 'rf_model' else lr_pipeline
# MAGIC best_run_id   = rf_run_id   if best_model_name == 'rf_model' else lr_run_id
# MAGIC
# MAGIC print(f"\n✅ Best model: {best_model_name} (ROC-AUC: {our_runs.iloc[0]['ROC-AUC']:.4f})")
# MAGIC print(f"   best_pipeline and best_run_id are set for Tasks 4 and 5.")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC > ✅ **Validate before moving on:**
# MAGIC - Both `lr_run_id` and `rf_run_id` are defined
# MAGIC - Comparison table shows both models' Test Accuracy and ROC-AUC
# MAGIC - `best_pipeline` is set to the winning model
# MAGIC - In **MLflow Experiments UI**: both runs appear in `loan_model_comparison`

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Task 4: Model Selection and Decision Threshold Optimization
# MAGIC
# MAGIC ### Objective
# MAGIC Evaluate the best model in depth — then go beyond the demo's default-threshold evaluation. In production loan models, **the decision threshold (default: 0.5) is a business parameter, not a fixed constant**. Lowering it increases recall (you catch more loan opportunities) but reduces precision (more false approvals). The right threshold depends on business cost assumptions.
# MAGIC
# MAGIC You will:
# MAGIC 1. Generate standard evaluation metrics for `best_pipeline`
# MAGIC 2. Plot a precision-recall curve across all thresholds
# MAGIC 3. Identify the threshold that maximizes F1-score for class 1 (loan acceptors)
# MAGIC 4. Compare default threshold vs. optimized threshold results
# MAGIC
# MAGIC > 💡 **Why this matters:** The demo evaluated at threshold 0.5 and reported recall. Here, you will show that with a better threshold choice, you can materially improve recall — potentially capturing significantly more loan candidates — without unacceptable precision loss.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Task 4a — Standard Evaluation at Default Threshold

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t4-eval" onclick="copyT4Eval()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t4-eval" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: best_pipeline has been selected and trained. X_test and y_test are available.
# MAGIC Task: Bank personal loan prediction (1 = accepted, 0 = declined).
# MAGIC
# MAGIC Task: Generate standard evaluation metrics for best_pipeline at the default threshold (0.5).
# MAGIC
# MAGIC Requirements:
# MAGIC 1. Predictions: y_pred = best_pipeline.predict(X_test)
# MAGIC    Probabilities: y_prob = best_pipeline.predict_proba(X_test)[:, 1]
# MAGIC
# MAGIC 2. Plot confusion matrix (ConfusionMatrixDisplay):
# MAGIC    - Labels: ['Declined (0)', 'Accepted (1)']
# MAGIC    - Blues colormap
# MAGIC    - Title: "Confusion Matrix — Bank Loan (Default Threshold = 0.5)"
# MAGIC
# MAGIC 3. Print classification report with target_names=['Declined (0)', 'Accepted (1)']
# MAGIC
# MAGIC 4. Print ROC-AUC score
# MAGIC
# MAGIC After the outputs, print a one-line summary:
# MAGIC "At threshold 0.5: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} for class 1 (Accepted)"
# MAGIC
# MAGIC Expected: Standard evaluation outputs that establish the baseline for threshold comparison.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT4Eval() {
# MAGIC   const text = document.getElementById("prompt-t4-eval").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t4-eval");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 4: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# Note the recall for class 1 (Accepted) at threshold 0.5 — you will compare this after threshold optimization.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 4a. Standard Evaluation at Default Threshold
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.metrics import (
# MAGIC     confusion_matrix, classification_report, roc_auc_score,
# MAGIC     ConfusionMatrixDisplay, precision_score, recall_score, f1_score
# MAGIC )
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC y_pred = best_pipeline.predict(X_test)
# MAGIC y_prob = best_pipeline.predict_proba(X_test)[:, 1]
# MAGIC
# MAGIC # Confusion matrix
# MAGIC fig, ax = plt.subplots(figsize=(6, 5))
# MAGIC ConfusionMatrixDisplay(
# MAGIC     confusion_matrix=confusion_matrix(y_test, y_pred),
# MAGIC     display_labels=['Declined (0)', 'Accepted (1)']
# MAGIC ).plot(ax=ax, colorbar=False, cmap='Blues')
# MAGIC ax.set_title("Confusion Matrix — Bank Loan (Default Threshold = 0.5)", fontsize=11, pad=10)
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC
# MAGIC # Classification report
# MAGIC print("Classification Report (threshold = 0.5):")
# MAGIC print(classification_report(y_test, y_pred, target_names=['Declined (0)', 'Accepted (1)']))
# MAGIC
# MAGIC # ROC-AUC
# MAGIC auc = roc_auc_score(y_test, y_prob)
# MAGIC print(f"ROC-AUC: {auc:.4f}")
# MAGIC
# MAGIC # One-line summary for class 1
# MAGIC p05 = precision_score(y_test, y_pred)
# MAGIC r05 = recall_score(y_test, y_pred)
# MAGIC f05 = f1_score(y_test, y_pred)
# MAGIC print(f"\nAt threshold 0.5: Precision={p05:.3f}, Recall={r05:.3f}, F1={f05:.3f} for class 1 (Accepted)")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Task 4b — Precision-Recall Curve and Threshold Optimization
# MAGIC
# MAGIC Now find the threshold that maximizes F1-score for class 1 — and compare it against the default.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t4-thresh" onclick="copyT4Thresh()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t4-thresh" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: y_prob (predicted probabilities from best_pipeline.predict_proba(X_test)[:, 1]) is already computed.
# MAGIC y_test is the true labels. Business context: for a bank loan model, we want to maximize recall for
# MAGIC class 1 (Accepted) to capture more loan opportunities, while keeping precision reasonable.
# MAGIC
# MAGIC Task: Perform decision threshold optimization for the loan prediction model.
# MAGIC
# MAGIC Step 1 — Precision-Recall curve:
# MAGIC - Use sklearn.metrics.precision_recall_curve(y_test, y_prob) to get precision, recall, thresholds
# MAGIC - Plot: x=recall, y=precision
# MAGIC - Annotate the point at threshold=0.5 with a marker
# MAGIC - Annotate the point at the optimal F1 threshold with a different marker
# MAGIC - Add a legend and title: "Precision-Recall Curve — Bank Loan Prediction"
# MAGIC
# MAGIC Step 2 — Find optimal threshold (maximizes F1 for class 1):
# MAGIC - For each threshold in numpy.linspace(0.1, 0.9, 81):
# MAGIC     * Apply threshold: y_pred_t = (y_prob >= threshold).astype(int)
# MAGIC     * Compute F1 for class 1: f1_score(y_test, y_pred_t, pos_label=1)
# MAGIC - Find the threshold with the highest F1
# MAGIC - Store as optimal_threshold
# MAGIC
# MAGIC Step 3 — Apply optimal threshold and compare:
# MAGIC - Generate y_pred_opt = (y_prob >= optimal_threshold).astype(int)
# MAGIC - Print classification report at optimal threshold
# MAGIC - Print a comparison table:
# MAGIC     | Metric      | Default (0.5) | Optimal ({optimal_threshold:.2f}) |
# MAGIC     |-------------|---------------|-----------------------------------|
# MAGIC     | Precision   | ...           | ...                               |
# MAGIC     | Recall      | ...           | ...                               |
# MAGIC     | F1-Score    | ...           | ...                               |
# MAGIC
# MAGIC Expected: A precision-recall curve, the optimal threshold value, and a clear before/after comparison.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT4Thresh() {
# MAGIC   const text = document.getElementById("prompt-t4-thresh").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t4-thresh");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 4: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# Compare recall at default vs optimal threshold. How many more loan opportunities does the optimized threshold capture?

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 4b. Precision-Recall Curve and Threshold Optimization
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import numpy as np
# MAGIC from sklearn.metrics import (
# MAGIC     precision_recall_curve, precision_score, recall_score, f1_score, classification_report
# MAGIC )
# MAGIC import matplotlib.pyplot as plt
# MAGIC
# MAGIC # Step 1 — Precision-Recall curve
# MAGIC precision_pts, recall_pts, thresh_pts = precision_recall_curve(y_test, y_prob)
# MAGIC
# MAGIC # Find point at default threshold 0.5
# MAGIC idx_05 = np.argmin(np.abs(thresh_pts - 0.5))
# MAGIC
# MAGIC # Step 2 — Find optimal threshold
# MAGIC thresholds = np.linspace(0.1, 0.9, 81)
# MAGIC f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int), pos_label=1) for t in thresholds]
# MAGIC optimal_threshold = thresholds[np.argmax(f1_scores)]
# MAGIC
# MAGIC # Find the matching point on PR curve
# MAGIC idx_opt = np.argmin(np.abs(thresh_pts - optimal_threshold))
# MAGIC
# MAGIC # Plot
# MAGIC fig, ax = plt.subplots(figsize=(8, 5))
# MAGIC ax.plot(recall_pts, precision_pts, lw=2, color='#2574B5', label='PR Curve')
# MAGIC ax.scatter(recall_pts[idx_05],  precision_pts[idx_05],
# MAGIC            s=120, zorder=5, color='orange', marker='o',
# MAGIC            label=f'Threshold = 0.50 (F1={f1_score(y_test,(y_prob>=0.5).astype(int)):.3f})')
# MAGIC ax.scatter(recall_pts[idx_opt], precision_pts[idx_opt],
# MAGIC            s=120, zorder=5, color='green', marker='*',
# MAGIC            label=f'Optimal threshold = {optimal_threshold:.2f} (F1={max(f1_scores):.3f})')
# MAGIC ax.set_xlabel('Recall', fontsize=11)
# MAGIC ax.set_ylabel('Precision', fontsize=11)
# MAGIC ax.set_title('Precision-Recall Curve — Bank Loan Prediction', fontsize=12, pad=12)
# MAGIC ax.legend(fontsize=10)
# MAGIC ax.grid(alpha=0.3)
# MAGIC plt.tight_layout()
# MAGIC plt.show()
# MAGIC
# MAGIC # Step 3 — Compare thresholds
# MAGIC y_pred_opt = (y_prob >= optimal_threshold).astype(int)
# MAGIC print(f"\nClassification Report at Optimal Threshold ({optimal_threshold:.2f}):")
# MAGIC print(classification_report(y_test, y_pred_opt, target_names=['Declined (0)', 'Accepted (1)']))
# MAGIC
# MAGIC # Comparison table
# MAGIC p_def, r_def, f_def = precision_score(y_test,y_pred), recall_score(y_test,y_pred), f1_score(y_test,y_pred)
# MAGIC p_opt, r_opt, f_opt = precision_score(y_test,y_pred_opt), recall_score(y_test,y_pred_opt), f1_score(y_test,y_pred_opt)
# MAGIC
# MAGIC print(f"{'Metric':<15} {'Default (0.50)':>15} {'Optimal ({:.2f})'.format(optimal_threshold):>18}")
# MAGIC print("-"*50)
# MAGIC for name, d, o in [("Precision", p_def, p_opt), ("Recall", r_def, r_opt), ("F1-Score", f_def, f_opt)]:
# MAGIC     print(f"{name:<15} {d:>15.3f} {o:>18.3f}")
# MAGIC print(f"\n✅ Optimal threshold: {optimal_threshold:.2f}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC > ✅ **Validate before moving on:**
# MAGIC - Precision-recall curve is plotted with both threshold points annotated
# MAGIC - `optimal_threshold` is defined and differs from 0.5
# MAGIC - Comparison table shows the recall improvement at the optimized threshold
# MAGIC - You can explain the business trade-off: higher recall means more loan opportunities captured

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Task 5: Register the Selected Model to Unity Catalog
# MAGIC
# MAGIC ### Objective
# MAGIC Register `best_pipeline` (the model selected in Task 3) to Unity Catalog. This is the model that will be served in Task 6.
# MAGIC
# MAGIC ### What to accomplish:
# MAGIC - Register from `best_run_id` to Unity Catalog
# MAGIC - Set alias: `champion`
# MAGIC - Include a description that references the model selection process
# MAGIC - Confirm registration in Unity Catalog → Models

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t5" onclick="copyT5()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t5" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: best_run_id is defined. The model artifact path is runs:/{best_run_id}/loan_rf_model
# MAGIC (or loan_lr_model if Logistic Regression was selected — use the correct artifact name).
# MAGIC DA.catalog_name and DA.schema_name are defined.
# MAGIC
# MAGIC Task: Register the selected model to Unity Catalog with a production alias.
# MAGIC
# MAGIC Requirements:
# MAGIC - mlflow.set_registry_uri("databricks-uc")
# MAGIC - Register:
# MAGIC     * model_uri = f"runs:/{best_run_id}/loan_rf_model"
# MAGIC     * name = f"{DA.catalog_name}.{DA.schema_name}.loan_rf_classifier"
# MAGIC - Use MlflowClient to:
# MAGIC     * Set alias "champion" on the registered version
# MAGIC     * Add description:
# MAGIC       "Best-performing model for bank personal loan prediction.
# MAGIC        Selected via multi-model comparison (Random Forest vs Logistic Regression baseline).
# MAGIC        Feature engineering: dropped redundant Experience, added Income_per_FamilyMember.
# MAGIC        Evaluated with precision-recall threshold optimization."
# MAGIC - Print: model name, version, alias
# MAGIC
# MAGIC Expected: Model appears in Unity Catalog with version 1 and 'champion' alias.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT5() {
# MAGIC   const text = document.getElementById("prompt-t5").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t5");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 5: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# Navigate to Unity Catalog → Models → confirm name, version 1, and 'champion' alias.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 5. Register the Model
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.tracking import MlflowClient
# MAGIC
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC client = MlflowClient()
# MAGIC
# MAGIC model_name = f"{DA.catalog_name}.{DA.schema_name}.loan_rf_classifier"
# MAGIC model_uri  = f"runs:/{run_id}/loan_rf_model"
# MAGIC
# MAGIC print(f"Registering model: {model_name}")
# MAGIC
# MAGIC # Register
# MAGIC registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
# MAGIC model_version = registered_model.version
# MAGIC print(f"✅ Registered version: {model_version}")
# MAGIC
# MAGIC # Set champion alias
# MAGIC client.set_registered_model_alias(name=model_name, alias="champion", version=model_version)
# MAGIC print(f"✅ Alias set: champion → version {model_version}")
# MAGIC
# MAGIC # Add description
# MAGIC client.update_model_version(
# MAGIC     name=model_name,
# MAGIC     version=model_version,
# MAGIC     description=(
# MAGIC         "Random Forest classifier for bank personal loan prediction. "
# MAGIC         "Trained via Genie Code (Agent Mode) in the ML Model Development course lab. "
# MAGIC         "Features: customer demographics, financial profile (Income, CCAvg, Mortgage), service usage."
# MAGIC     )
# MAGIC )
# MAGIC print(f"✅ Description added")
# MAGIC print(f"\nNavigate to: Unity Catalog → Models → {model_name}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC > ✅ **Validate before moving on:**
# MAGIC - Model appears in **Unity Catalog → Models** under your catalog and schema
# MAGIC - Version 1 has the `champion` alias
# MAGIC - Description references the model selection and feature engineering steps

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Task 6: Deploy and Test the Serving Endpoint
# MAGIC
# MAGIC ### Objective
# MAGIC Deploy the registered model to a real-time Databricks serving endpoint, verify it reaches `READY` state, and **send a sample inference request** to confirm the endpoint returns predictions. This closes the full ML lifecycle loop.
# MAGIC
# MAGIC ### What to accomplish:
# MAGIC - Create a real-time serving endpoint using `WorkspaceClient`
# MAGIC - Wait for `READY` state
# MAGIC - Send a sample inference request using `X_test` records
# MAGIC - Confirm the endpoint returns loan predictions (0 or 1)
# MAGIC
# MAGIC > ⚠️ **Endpoint creation takes 3–8 minutes.** The deployment cell blocks until `READY`. After the endpoint is live, the inference test cell confirms it responds correctly.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Task 6a — Create the Serving Endpoint

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t6-deploy" onclick="copyT6Deploy()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t6-deploy" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: A model is registered in Unity Catalog as:
# MAGIC {DA.catalog_name}.{DA.schema_name}.loan_rf_classifier  (alias: champion, version 1)
# MAGIC DA.catalog_name, DA.schema_name are defined. X_test is available.
# MAGIC
# MAGIC Task: Deploy the registered model to a Databricks real-time serving endpoint.
# MAGIC
# MAGIC Requirements:
# MAGIC 1. from databricks.sdk import WorkspaceClient
# MAGIC    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
# MAGIC 2. w = WorkspaceClient()
# MAGIC 3. endpoint_name = f"loan-rf-{DA.schema_name}".replace("_","-").replace(".","-")
# MAGIC    (lowercase with hyphens — no underscores or dots)
# MAGIC 4. Call w.serving_endpoints.create_and_wait():
# MAGIC    - name: endpoint_name
# MAGIC    - config: EndpointCoreConfigInput with ServedEntityInput:
# MAGIC        * entity_name: full UC model name
# MAGIC        * entity_version: "1"
# MAGIC        * scale_to_zero_enabled: True
# MAGIC        * workload_size: "Small"
# MAGIC 5. Print endpoint name and state after READY
# MAGIC
# MAGIC Expected: Endpoint in Serving UI with READY status.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT6Deploy() {
# MAGIC   const text = document.getElementById("prompt-t6-deploy").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t6-deploy");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 6: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# After the cell completes: navigate to Serving → Endpoints → confirm READY status.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 6a. Create the Serving Endpoint
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.serving import (
# MAGIC     EndpointCoreConfigInput,
# MAGIC     ServedEntityInput,
# MAGIC )
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC model_name    = f"{DA.catalog_name}.{DA.schema_name}.loan_rf_classifier"
# MAGIC endpoint_name = f"loan-rf-{DA.schema_name}".replace("_", "-").replace(".", "-")
# MAGIC
# MAGIC print(f"Creating endpoint: {endpoint_name}")
# MAGIC print(f"Model            : {model_name}@champion")
# MAGIC
# MAGIC endpoint = w.serving_endpoints.create_and_wait(
# MAGIC     name=endpoint_name,
# MAGIC     config=EndpointCoreConfigInput(
# MAGIC         served_entities=[
# MAGIC             ServedEntityInput(
# MAGIC                 entity_name=model_name,
# MAGIC                 entity_version="1",
# MAGIC                 scale_to_zero_enabled=True,
# MAGIC                 workload_size="Small",
# MAGIC             )
# MAGIC         ]
# MAGIC     ),
# MAGIC )
# MAGIC
# MAGIC print(f"\n✅ Endpoint '{endpoint.name}' is {endpoint.state.ready.value}")
# MAGIC print(f"   Navigate to: Serving → Endpoints → {endpoint_name}")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Task 6b — Test the Endpoint with a Sample Inference Request
# MAGIC
# MAGIC Now that the endpoint is live, verify it works by sending real records from your test set.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open <strong>Genie Code</strong> panel → switch mode to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press Enter — Genie Code generates and inserts a cell</li>
# MAGIC <li>Review the code, then run it</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-t6-infer" onclick="copyT6Infer()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="prompt-t6-infer" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: A Databricks serving endpoint named endpoint_name is READY.
# MAGIC X_test is available as a pandas DataFrame with the same features the model was trained on.
# MAGIC w = WorkspaceClient() is already initialized.
# MAGIC
# MAGIC Task: Send a sample inference request to the deployed endpoint and display the predictions.
# MAGIC
# MAGIC Requirements:
# MAGIC 1. Take the first 5 rows of X_test
# MAGIC 2. Convert to a list of dicts: sample_records = X_test.head(5).to_dict(orient='records')
# MAGIC 3. Call: response = w.serving_endpoints.query(name=endpoint_name, dataframe_records=sample_records)
# MAGIC 4. Extract predictions from response.predictions
# MAGIC 5. Print a table showing:
# MAGIC    - Row index
# MAGIC    - Key features (Income, Income_per_FamilyMember, CCAvg)
# MAGIC    - Predicted label (0 = Declined, 1 = Accepted)
# MAGIC    - True label from y_test
# MAGIC
# MAGIC Expected: A table showing 5 sample records, their predicted loan decisions, and the true labels.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyT6Infer() {
# MAGIC   const text = document.getElementById("prompt-t6-infer").innerText;
# MAGIC   const btn  = document.getElementById("copy-btn-t6-infer");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '\u2705 Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# TODO — Task 6: Use Genie Code (Agent Mode) to generate and run the cell above.
# After running, validate the output against the criteria below.
# Confirm the endpoint returns 0/1 predictions. Check if predictions match y_test for the sample rows.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC <details style="margin: 8px 0;">
# MAGIC <summary style="background: linear-gradient(135deg, #1B5162, #2574B5); color: white; padding: 14px 20px; cursor: pointer; font-weight: 700; font-size: 13pt; border-radius: 8px; user-select: none; display: flex; align-items: center; gap: 10px;">
# MAGIC <span style="background: rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 8px; font-size: 11pt;">SOLUTION</span> Task 6b. Test the Endpoint with Sample Inference Request
# MAGIC </summary>
# MAGIC <div style="border: 2px solid #1B5162; border-top: none; border-radius: 0 0 8px 8px; padding: 18px 20px; background: #F8F9FC; position: relative;"><button onclick="var c=this.parentElement.querySelector('pre code');var t=document.createElement('textarea');t.value=c?c.textContent:'';t.style.position='fixed';t.style.opacity='0';document.body.appendChild(t);t.select();document.execCommand('copy');document.body.removeChild(t);this.textContent='Copied!';var b=this;setTimeout(function(){b.textContent='Copy'},1500)" style="position:absolute;top:10px;right:12px;background:linear-gradient(135deg,#1B5162,#2574B5);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11pt;font-weight:600;cursor:pointer;z-index:2;">Copy</button>
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC
# MAGIC # Take 5 test samples for inference
# MAGIC sample_x  = X_test.head(5).reset_index(drop=True)
# MAGIC sample_y  = y_test.head(5).reset_index(drop=True)
# MAGIC
# MAGIC # Convert to list of records for the serving endpoint
# MAGIC sample_records = sample_x.to_dict(orient='records')
# MAGIC
# MAGIC # Query the endpoint
# MAGIC response = w.serving_endpoints.query(
# MAGIC     name=endpoint_name,
# MAGIC     dataframe_records=sample_records
# MAGIC )
# MAGIC
# MAGIC # Extract predictions
# MAGIC predictions = response.predictions
# MAGIC
# MAGIC # Display comparison table
# MAGIC result_df = pd.DataFrame({
# MAGIC     'Income': sample_x['Income'].values,
# MAGIC     'Income_per_FM': sample_x['Income_per_FamilyMember'].round(2).values,
# MAGIC     'CCAvg': sample_x['CCAvg'].values,
# MAGIC     'Predicted': predictions,
# MAGIC     'Actual': sample_y.values,
# MAGIC     'Correct': ['✅' if p == a else '❌' for p, a in zip(predictions, sample_y.values)]
# MAGIC })
# MAGIC
# MAGIC print(f"Endpoint: {endpoint_name}")
# MAGIC print(f"\nSample Inference Results (5 rows):")
# MAGIC print(result_df.to_string(index=False))
# MAGIC print(f"\n✅ Endpoint is live and returning predictions.")
# MAGIC ```
# MAGIC
# MAGIC </div>
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC > ✅ **Validate before moving on:**
# MAGIC - Endpoint returns a response without error
# MAGIC - Predictions are 0 or 1 (loan declined or accepted)
# MAGIC - The Correct column shows at least some ✅ matches — confirming the model is working
# MAGIC - Navigate to **Serving → Endpoints → {endpoint_name}** to view the request log

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Conclusion
# MAGIC
# MAGIC You have completed an end-to-end agentic ML workflow — and gone **further than the demo** with three new skills.
# MAGIC
# MAGIC | Task | What You Did | New skill vs. Demo |
# MAGIC |------|-------------|-------------------|
# MAGIC | Task 1 — Load & EDA | Loaded bank loan data, explored class balance, nulls, correlations | EDA on new dataset |
# MAGIC | Task 2 — Feature Engineering | Dropped redundant `Experience`, created `Income_per_FamilyMember` derived feature | **Business-driven feature decisions** |
# MAGIC | Task 3 — Multi-Model Comparison | Trained Logistic Regression + Random Forest, compared via `mlflow.search_runs()` | **Multi-model comparison + MLflow API** |
# MAGIC | Task 4 — Threshold Optimization | Generated precision-recall curve, found optimal decision threshold | **Threshold optimization for business context** |
# MAGIC | Task 5 — Registration | Registered the selected model to Unity Catalog with `champion` alias | UC model lifecycle |
# MAGIC | Task 6 — Deployment + Inference | Deployed to serving endpoint, tested with live inference request | **End-to-end serving verification** |
# MAGIC
# MAGIC **Key takeaways:**
# MAGIC - Feature engineering with business logic (dropping correlated features, creating derived features) is more powerful than generic preprocessing
# MAGIC - Comparing models via `mlflow.search_runs()` is a Databricks-native pattern for multi-model experiments
# MAGIC - The decision threshold is a **business parameter** — not a fixed constant. Optimizing it can materially improve business outcomes
# MAGIC - Closing the full loop (deploy + test inference) confirms the entire ML system works end-to-end
# MAGIC
# MAGIC > 💡 **Reflect:** Where in this lab did Genie Code produce the most useful output? Where did your prompt specificity make the biggest difference?

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>