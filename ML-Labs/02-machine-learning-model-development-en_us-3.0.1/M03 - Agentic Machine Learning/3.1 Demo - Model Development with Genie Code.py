# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - Model Development with Genie Code
# MAGIC
# MAGIC In this demo, we use **Genie Code** operating in **Agent Mode** to drive a complete, end-to-end machine learning workflow on Databricks — from loading a Feature Store dataset through registering a production-ready classifier in Unity Catalog. Rather than writing code directly, we express intent through structured natural language prompts and observe how Genie Code generates, executes, and tracks each step in context.
# MAGIC
# MAGIC The workflow follows four explicit phases: **Explore** (data loading and EDA), **Iterate** (synthesizing findings into a modeling plan), **Build** (feature engineering pipeline and MLflow-tracked training), and **Validate** (model evaluation, interpretation, and Unity Catalog registration). Reference implementations are provided below each prompt so you can compare Genie Code's output against a known-good solution.
# MAGIC
# MAGIC
# MAGIC
# MAGIC The workflow follows four explicit phases:
# MAGIC
# MAGIC | Phase | What Happens |
# MAGIC |-------|-------------|
# MAGIC | **1 — Explore** | Load data, run EDA, understand the dataset |
# MAGIC | **2 — Iterate** | Ask Genie Code to synthesize findings and recommend next steps |
# MAGIC | **3 — Build** | Generate the feature engineering and training pipeline |
# MAGIC | **4 — Validate** | Evaluate results, interpret model behavior, register to Unity Catalog |
# MAGIC
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC - Use structured natural language prompts in Genie Code Agent Mode to drive each phase of an ML workflow.
# MAGIC - Observe how Genie Code carries **contextual state** across prompts - from data loading through model evaluation - without re-specifying the dataset each time.
# MAGIC - Apply the **Explore → Iterate → Build → Validate** framework to any ML classification task on Databricks.
# MAGIC - Track and evaluate a Genie Code–generated classifier using **MLflow**, and register it to **Unity Catalog** with a production alias.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## REQUIRED - SELECT A COMPUTE ENVIRONMENT
# MAGIC
# MAGIC <div style="border-left: 4px solid #F44336; background: #FFEBEE; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC   <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC     <div>
# MAGIC       <strong style="color: #C62828; font-size: 1.1em;">Select Compute</strong>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;">Before starting this notebook, select the required compute environment listed below.</p>
# MAGIC       <ul style="margin: 12px 0 0 16px; color: #333;">
# MAGIC         <li><strong>Serverless Compute, Version 5</strong> — <a href="https://docs.databricks.com/aws/en/compute/serverless/dependencies#-select-an-environment-version" style="color: #1976D2; text-decoration: underline;">How to select an environment version</a></li>
# MAGIC       </ul>
# MAGIC       <p style="margin: 8px 0 0 0; color: #333;"><strong>NOTE:</strong> This notebook was <strong>developed and tested using Serverless V5</strong>. Other compute options may work but are not guaranteed to support all features demonstrated, particularly the Genie Code Agent Mode integration.</p>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="width: 100%; font-family: sans-serif;"><div style="background: #F9F7F4; border-radius: 10px; padding: 24px 28px; box-shadow: 0 2px 8px rgba(27,49,57,0.06); border-top: 6px solid #FF5F46;">  <img src="../Includes/images/genie-code.png" style="height: 64px; margin-bottom: 10px;">  <div style="font-size: 15pt; color: #0B2026; line-height: 1.7; margin-bottom: 16px;">    Want to know more about Genie Code Agent Mode and end-to-end ML workflows on Databricks? Ask Genie Code. Click on the genie icon <img src="../Includes/images/genie-icon.png" style="height: 32px; vertical-align: middle;"> and begin querying. For example, click the <strong>Copy</strong> button below and paste into <strong>Genie Code</strong>.  </div>  <div style="display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 10px 14px; font-size: 14pt; font-family: monospace; color: #0B2026;">    <span id="genie-query-demo" style="flex: 1;">What is Genie Code Agent Mode in Databricks? How can I use it to build an end-to-end machine learning workflow - from loading Feature Store data to logging a model in MLflow and registering it to Unity Catalog?</span>    <button onclick="      var text = document.getElementById('genie-query-demo').innerText;      var ta = document.createElement('textarea');      ta.value = text;      ta.style.position = 'fixed';      ta.style.opacity = '0';      document.body.appendChild(ta);      ta.select();      document.execCommand('copy');      document.body.removeChild(ta);      this.innerText = 'Copied!';      var btn = this;      setTimeout(function(){ btn.innerText = 'Copy'; }, 2000);    " style="background: #FF5F46; color: white; border: none; border-radius: 4px; padding: 4px 12px; font-size: 13pt; cursor: pointer; white-space: nowrap;">Copy</button>  </div></div></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## From Writing Code to Expressing Intent
# MAGIC
# MAGIC In Modules 1 and 2, every step was deliberate and manual:
# MAGIC - Loaded feature tables directly from the Databricks Feature Store
# MAGIC - Engineered and transformed features using scikit-learn Pipelines
# MAGIC - Trained classification models with full MLflow tracking
# MAGIC - Tuned hyperparameters with Optuna across grid, random, and Bayesian strategies
# MAGIC
# MAGIC We built that foundation on purpose. **You need to understand what good ML looks like before you can ask for it.**
# MAGIC
# MAGIC In this module, we shift from *writing code* to *expressing intent*. Genie Code interprets your natural language prompts, generates and executes ML code, and integrates natively with Unity Catalog, Feature Store, and Model Serving.
# MAGIC
# MAGIC > 💡 **Core principle:** The quality of what Genie Code produces is directly proportional to the quality of your prompts. Vague intent → generic code. Precise, domain-grounded intent → production-ready code.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## A Note on Prompt Complexity
# MAGIC
# MAGIC The prompts in this demo are intentionally detailed — they are **teaching examples** that make every design decision visible and explicit. But Genie Code does not require this level of specificity to be useful.
# MAGIC
# MAGIC **Two ways to use the prompts in this demo:**
# MAGIC
# MAGIC 1. **Run them as-is** *(recommended for first-time learners)* — The detailed prompts expose every requirement and constraint so you understand exactly what Genie Code is doing and why. Use these to build intuition.
# MAGIC 2. **Start simpler, then refine** — Begin with a concise high-level request and iterate from Genie Code's response, or ask Genie Code to help you write a more precise prompt (see meta-prompting below).
# MAGIC
# MAGIC ### The One-Shot Approach
# MAGIC
# MAGIC You do not need a multi-step, multi-prompt workflow to get a working result. A single, clearly stated prompt can kick off the entire ML pipeline:
# MAGIC
# MAGIC > *"I have customer churn data loaded as `churn_pdf`. Build me a complete ML pipeline: prep the features, split the data, train a classifier to predict who will churn, log everything to MLflow, and report how well it works. Use scikit-learn best practices."*
# MAGIC
# MAGIC This produces a functional end-to-end result in one shot. The phased approach in this demo is more detailed because it is **designed to teach** each decision — not because Genie Code requires that level of instruction.
# MAGIC
# MAGIC ### Meta-Prompting: Ask Genie Code to Write the Prompt for You
# MAGIC
# MAGIC If you are unsure how to phrase a request, describe your goal in plain language and ask Genie Code to help you craft a better prompt before running it:
# MAGIC
# MAGIC > *"I have my churn data loaded as `churn_pdf`. Help me write a prompt I can use to ask Genie Code to prepare the features and split the data so I can train a model to predict who will churn using best practices."*
# MAGIC
# MAGIC Genie Code will return a refined, specific prompt you can then paste back and run. This **meta-prompting** pattern is especially useful when starting a new task where you know the goal but are not yet sure which technical requirements to include.
# MAGIC
# MAGIC > 💡 **Keep in mind:** The detailed prompts in this demo are intentionally explicit — every requirement and constraint is spelled out so you can see exactly what Genie Code is responding to and why. This is a learning scaffold, not a template for how you should write prompts going forward. Once you've built intuition by running these, start simpler on your own tasks and iterate from Genie Code's responses. You can also use meta-prompting — ask Genie Code to help you write the prompt — as shown above.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classroom Setup
# MAGIC
# MAGIC Run the following cell to configure your working environment for this course.
# MAGIC
# MAGIC This setup will:
# MAGIC - Initialize the `DA` object (Databricks Academy helper)
# MAGIC - Configure your **default catalog** and **schema**
# MAGIC - Create the `customer_churn` table and `customer_churn_features` Feature Store table used throughout this demo

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.1

# COMMAND ----------

# MAGIC %md
# MAGIC **Classroom variables available throughout this demo:**

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Datasets Location: {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Section 1 — Setting Up Genie Code Agent Mode
# MAGIC
# MAGIC Before we start, let's configure Genie Code so it produces grounded, project-aware code — not generic output.
# MAGIC
# MAGIC Two things make Genie Code context-aware in an ML project:
# MAGIC
# MAGIC | Component | What it does |
# MAGIC |-----------|-------------|
# MAGIC | **Agent Mode** | Enables Genie Code to read notebook state, execute tool calls, and reason across multiple steps |
# MAGIC | **Custom Instructions** | Provides project-level context: dataset name, task type, target column, experiment name, and coding standards |
# MAGIC
# MAGIC > ⚠️ **Without Agent Mode**, Genie Code behaves like a basic code assistant — it cannot read your DataFrame, access notebook variables, or execute multi-step reasoning. Always switch to **Agent** before running the prompts in this demo.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### How to Use Genie Code Agent Mode
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 10px 0;">
# MAGIC   <strong style="color: #0d47a1;">Step-by-step — activate Agent Mode:</strong>
# MAGIC   <ol style="margin: 8px 0 0 18px; color: #333; line-height: 1.8;">
# MAGIC     <li>Click the <strong>Genie</strong> icon on the right-hand panel of your notebook (AI wand or chat bubble).</li>
# MAGIC     <li>In the mode drop-down at the top of the panel, select <strong>Agent</strong> (not the default mode).</li>
# MAGIC     <li>You are now in Agent Mode. Genie Code can read notebook state, insert and execute cells, and reason across tool calls.</li>
# MAGIC     <li>For each prompt in this demo: click <strong>Copy Genie Prompt</strong>, paste into the Agent chat, press Enter.</li>
# MAGIC     <li>After Genie Code generates a cell, <strong>review the code</strong>, then run it.</li>
# MAGIC   </ol>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #f57c00; background: #fff3e0; padding: 14px 18px; border-radius: 4px; margin: 10px 0;">
# MAGIC   <strong style="color: #e65100;">Always Review Generated Code</strong>
# MAGIC   <p style="margin: 6px 0 0 0; color: #333;">Genie Code accelerates development, but <em>you remain responsible</em> for reviewing and understanding what it generates before running it. If something looks wrong or off-target, refine the prompt and regenerate.</p>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Custom Instructions for This Session
# MAGIC
# MAGIC The classroom setup cell above automatically wrote a `.assistant_instructions.md` file to this notebook's workspace folder. Genie Code reads this file when the Genie panel opens, so it is already grounded in the project context before you send the first prompt — no manual copy-paste required.
# MAGIC
# MAGIC <div style="border-left: 4px solid #2e7d32; background: #e8f5e9; padding: 14px 18px; border-radius: 4px; margin: 10px 0;">
# MAGIC   <strong style="color: #1b5e20;">✅ Auto-configured — no action needed</strong>
# MAGIC   <p style="margin: 6px 0 0 0; color: #333;">The file was written by the setup script and will be loaded automatically each time you open Genie Code in this notebook. Re-running classroom setup rewrites it fresh (create or replace).</p>
# MAGIC </div>
# MAGIC
# MAGIC <p style="margin: 14px 0 6px 0; font-weight: 600; color: #263238;">Contents of <code>.assistant_instructions.md</code> for this session:</p>
# MAGIC
# MAGIC <div style="background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px 20px; font-family: ui-monospace, monospace; font-size: 0.88rem; line-height: 1.6; margin: 0;">
# MAGIC You are assisting with a binary classification task to predict customer churn.<br><br>
# MAGIC Dataset: customer_churn (joined with customer_churn_features via CustomerID)<br>
# MAGIC Join key: CustomerID (left join from customer_churn to customer_churn_features)<br>
# MAGIC Target column: Churn (Yes = churned, No = retained) — encode as 1/0<br>
# MAGIC Catalog: {DA.catalog_name}<br>
# MAGIC Schema: {DA.schema_name}<br>
# MAGIC MLflow experiment: /Users/{DA.username}/churn_prediction_genie<br>
# MAGIC Modeling library: scikit-learn (Pipeline-based preprocessing preferred)<br>
# MAGIC All experiments must be logged to MLflow with explicit parameter and metric logging.<br>
# MAGIC Register final models to Unity Catalog using mlflow.set_registry_uri("databricks-uc").
# MAGIC </div>
# MAGIC
# MAGIC > 💡 **What this means for you:** The custom instructions file was written to your workspace folder by the setup script — your catalog name, schema name, and MLflow experiment path are already filled in with your actual values. When you open the Genie panel, Genie Code automatically loads this file as pre-loaded context. That is what makes every prompt in this demo produce project-specific output rather than generic boilerplate — Genie Code already knows your environment before you type a single word.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open the <strong>Genie</strong> panel (right sidebar → AI icon)</li>
# MAGIC <li>Switch the mode drop-down to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press <strong>Enter</strong> — Genie Code will generate and insert a new cell</li>
# MAGIC <li>Review the generated code, then run the cell</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC <button id="copy-btn-orient" onclick="copyOrientPrompt()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-btn-or" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC What tools, data sources, and connections are available for this session?
# MAGIC
# MAGIC Please list:
# MAGIC 1. Unity Catalog connections and the catalog/schema in scope
# MAGIC 2. Any Feature Store or Delta tables you can access
# MAGIC 3. The MLflow experiment path configured in the custom instructions
# MAGIC 4. Confirm you are in Agent Mode and can execute cells in this notebook
# MAGIC
# MAGIC This is a readiness check before we start the ML workflow.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyOrientPrompt() {
# MAGIC   const text = document.getElementById("copy-btn-or").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-orient");
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

# MAGIC %md
# MAGIC > 💡 **What to notice:** When Genie Code responds to this readiness check, it should list your Unity Catalog, the Feature Store tables, and the MLflow experiment path — pulled directly from the custom instructions loaded at startup. Notice that Genie Code read your catalog name from the workspace; you didn't type it. This is the difference between a generic code assistant and a grounded agent: it already knows the shape of your project before the first ML prompt.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### ℹ️ Before You Begin — Genie Code Chat Limits
# MAGIC
# MAGIC <div style="border-left: 4px solid #546e7a; background: #eceff1; padding: 16px 20px; border-radius: 4px; margin: 12px 0;">
# MAGIC
# MAGIC <p style="margin: 0 0 10px 0; color: #263238;">This notebook is structured to be worked through in <strong>a single Genie Code chat thread</strong>. A few things to keep in mind as the thread grows:</p>
# MAGIC
# MAGIC <ul style="margin: 0 0 14px 18px; color: #333; line-height: 1.9;">
# MAGIC   <li><strong>Chat history is finite.</strong> Genie Code carries prior turns as context, but the conversation has an upper token budget. Once that budget is reached, the <strong>oldest turns drop out</strong> of context first — newer turns are not affected, but Genie can lose awareness of decisions made earlier in the notebook.</li>
# MAGIC   <li><strong>Side panel vs. inline behave differently.</strong>
# MAGIC     <ul style="margin: 4px 0 0 18px; line-height: 1.8;">
# MAGIC       <li><em>Side panel</em> chats persist across page refreshes and navigation, so the thread keeps accumulating context across cells.</li>
# MAGIC       <li><em>Inline</em> (per-cell) Genie Code is scoped to that cell and the current session only — use the up/down arrows to recall prior prompts, but past responses are not replayed as context.</li>
# MAGIC     </ul>
# MAGIC   </li>
# MAGIC   <li><strong>There is no user-visible "compact" button.</strong> Trimming is automatic. If Genie starts referencing the wrong table, forgetting a transformation applied earlier, or producing code that contradicts an earlier cell, that is a signal the early context has been evicted.</li>
# MAGIC </ul>
# MAGIC
# MAGIC <p style="margin: 0 0 8px 0; color: #263238;"><strong>Recommended practice while working through this notebook:</strong></p>
# MAGIC <ol style="margin: 0 0 14px 18px; color: #333; line-height: 1.9;">
# MAGIC   <li><strong>Start a new chat thread at natural breakpoints</strong> — for example, between data prep → model training → evaluation. A fresh thread is the most reliable form of compaction.</li>
# MAGIC   <li><strong>Re-ground Genie when switching topics.</strong> Paste the relevant DataFrame schema, table name, or prior result back into the prompt instead of relying on it to remember.</li>
# MAGIC   <li><strong>Prefer inline Genie Code for small, self-contained edits</strong> (one cell, one transformation). Use the side panel when you want continuity across cells.</li>
# MAGIC   <li><strong>Watch for drift.</strong> If answers start to feel "off," re-state your goal explicitly or restart the thread — it is faster than debugging Genie Code's stale assumptions.</li>
# MAGIC </ol>
# MAGIC
# MAGIC <p style="margin: 0; color: #546e7a; font-size: 0.9em;">Rate limits exist (per-user and per-workspace) but are set for abuse prevention — you are unlikely to hit them in a class session. For current behavior and limits, see the <a href="https://docs.databricks.com/aws/en/genie-code" style="color: #1565c0;">Genie Code documentation</a>.</p>
# MAGIC
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Phase 1 — Explore: Data Loading & EDA
# MAGIC
# MAGIC We begin by understanding the data before building anything. Three EDA prompts in sequence — run each generated cell before moving to the next prompt.
# MAGIC
# MAGIC **Dataset:** Telco customer churn — binary classification task.
# MAGIC **Target:** `Churn` (Yes = churned, No = retained)
# MAGIC **Tables:**
# MAGIC - `customer_churn`: CustomerID, Gender, SeniorCitizen, Partner, InternetService, Contract, PaperlessBilling, PaymentMethod, Churn
# MAGIC - `customer_churn_features`: CustomerID, AverageMonthlyCharges (Feature Store–registered)
# MAGIC
# MAGIC > 💡 **Why use the Feature Store API instead of `spark.read.table()`?** Loading via the Feature Engineering client preserves feature lineage in Unity Catalog, enabling full traceability from training data to deployed model. Genie Code knows this from the custom instructions.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open the <strong>Genie</strong> panel (right sidebar → AI icon)</li>
# MAGIC <li>Switch the mode drop-down to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press <strong>Enter</strong> — Genie Code will generate and insert a new cell</li>
# MAGIC <li>Review the generated code, then run the cell</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC <button id="copy-btn-load" onclick="copyLoadPrompt()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-btn-" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: I am working on a customer churn prediction task in a Databricks notebook.
# MAGIC Two tables have been set up in this session and are accessible via DA.catalog_name and DA.schema_name.
# MAGIC
# MAGIC Task: Load and join these two tables into a single modeling DataFrame:
# MAGIC - Main table: customer_churn — contains customer profile features and the 'Churn' target label
# MAGIC - Feature table: customer_churn_features — contains AverageMonthlyCharges (registered in Feature Store)
# MAGIC - Join key: CustomerID (left join from main to features)
# MAGIC
# MAGIC Requirements:
# MAGIC - Use spark.table() with f"{DA.catalog_name}.{DA.schema_name}.customer_churn" format
# MAGIC - Store the joined Spark DataFrame as customer_churn_df
# MAGIC - Convert to pandas and store as churn_pdf (for visualization in subsequent steps)
# MAGIC - Print the total row count, column count, and schema
# MAGIC - Display the first 10 rows
# MAGIC
# MAGIC Expected output: A joined pandas DataFrame with both profile and feature columns, ready for EDA.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyLoadPrompt() {
# MAGIC   const text = document.getElementById("copy-btn-").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-load");
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

# MAGIC %md
# MAGIC > 🔍 **What to observe:**
# MAGIC - Genie Code referenced `DA.catalog_name` and `DA.schema_name` — it read notebook variables, not hardcoded strings.
# MAGIC - The join preserved all columns from both tables. If AverageMonthlyCharges appears, the Feature Store join succeeded.
# MAGIC - Note the `Churn` column contains string values ('Yes'/'No') — we'll encode these in Phase 3.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### EDA Prompt 1 — Class Distribution
# MAGIC
# MAGIC > Run the prompt, wait for Genie Code to generate the cell, then execute it before moving on.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open the <strong>Genie</strong> panel (right sidebar → AI icon)</li>
# MAGIC <li>Switch the mode drop-down to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press <strong>Enter</strong> — Genie Code will generate and insert a new cell</li>
# MAGIC <li>Review the generated code, then run the cell</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC <button id="copy-btn-eda1" onclick="copyEDA1()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-btn-" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: I have a pandas DataFrame called churn_pdf with a 'Churn' column containing 'Yes' (churned) and 'No' (retained) string values.
# MAGIC
# MAGIC Task: Analyze the class distribution of the Churn target variable.
# MAGIC
# MAGIC Requirements:
# MAGIC - Count and percentage for each class (Yes/No)
# MAGIC - A bar chart with count labels on each bar and clear axis labels
# MAGIC - Colors: green for 'No' (retained), red for 'Yes' (churned)
# MAGIC
# MAGIC Expected output: A labeled bar chart and a printed summary table showing whether the dataset is balanced or imbalanced. This will inform modeling strategy.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyEDA1() {
# MAGIC   const text = document.getElementById("copy-btn-").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-eda1");
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

# MAGIC %md
# MAGIC > 🔍 **What to observe:**
# MAGIC - If 'No' (Retained) significantly outnumbers 'Yes' (Churned), the dataset is **class-imbalanced** — this affects which metrics matter (recall on Churned > overall accuracy).
# MAGIC - **What to notice:** You should see approximately 73% retained customers vs. 27% churned — a roughly 2.7:1 class imbalance. This matters because a model trained on imbalanced data tends to over-predict the majority class (retained), inflating accuracy while missing the churners you actually care about. We address this directly in Phase 3 by setting `class_weight="balanced"` on the classifier, which adjusts the model to give more weight to the minority class during training.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### EDA Prompt 2 — Missing Values
# MAGIC
# MAGIC > Run the prompt and execute the generated cell before continuing.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open the <strong>Genie</strong> panel (right sidebar → AI icon)</li>
# MAGIC <li>Switch the mode drop-down to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press <strong>Enter</strong> — Genie Code will generate and insert a new cell</li>
# MAGIC <li>Review the generated code, then run the cell</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC <button id="copy-btn-eda2" onclick="copyEDA2()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-btn-" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: I have a pandas DataFrame called churn_pdf already loaded in this notebook.
# MAGIC
# MAGIC Task: Check for missing or null values across all columns.
# MAGIC
# MAGIC Requirements:
# MAGIC - Show a summary table: column name | null count | null percentage (%)
# MAGIC - Sort by null count descending
# MAGIC - Only show columns that have at least one missing value
# MAGIC - If no missing values exist, print a clear confirmation message
# MAGIC
# MAGIC Expected output: A data quality report. Any column with >5% nulls may need imputation strategy before modeling.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyEDA2() {
# MAGIC   const text = document.getElementById("copy-btn-").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-eda2");
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

# MAGIC %md
# MAGIC ---
# MAGIC ### EDA Prompt 3 — Feature Correlations
# MAGIC
# MAGIC > Run the prompt and execute the generated cell before continuing.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open the <strong>Genie</strong> panel (right sidebar → AI icon)</li>
# MAGIC <li>Switch the mode drop-down to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press <strong>Enter</strong> — Genie Code will generate and insert a new cell</li>
# MAGIC <li>Review the generated code, then run the cell</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC <button id="copy-btn-eda3" onclick="copyEDA3()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-btn-" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: I have churn_pdf in memory. The target column is 'Churn' (string: 'Yes'/'No').
# MAGIC
# MAGIC Task: Create a correlation heatmap of all numerical features, including the encoded target.
# MAGIC
# MAGIC Requirements:
# MAGIC - Encode 'Churn' as a binary column (Yes=1, No=0) and name it 'Churn_binary'
# MAGIC - Encode 'SeniorCitizen' as integer if it is stored as a string
# MAGIC - Select only numeric columns for the correlation matrix
# MAGIC - Use seaborn heatmap with annotated values (fmt='.2f'), coolwarm colormap, centered at 0
# MAGIC - Add a title: "Feature Correlation Heatmap (Churn_binary = target)"
# MAGIC - After the heatmap, print the top 5 features most correlated with Churn_binary (absolute correlation, sorted descending)
# MAGIC
# MAGIC Expected output: A heatmap and a ranked list of the strongest churn predictors.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyEDA3() {
# MAGIC   const text = document.getElementById("copy-btn-").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-eda3");
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

# MAGIC %md
# MAGIC ---
# MAGIC ## Phase 2 — Iterate: Synthesize EDA Findings
# MAGIC
# MAGIC Before building the pipeline, we stop and ask Genie Code to *reason* about what the EDA revealed. This is the **iterate** phase — we are not asking for code. We are asking for analytical judgment.
# MAGIC
# MAGIC This prompt demonstrates the difference between a **code assistant** (generates code on demand) and an **AI agent** (reasons over prior outputs and recommends a strategy).
# MAGIC
# MAGIC > ⚠️ **This prompt does not generate a code cell.** Genie Code responds in the chat panel with a structured analysis.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <button id="copy-btn-iterate" onclick="copyIterate()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-btn-ite" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Based on the three EDA results we just reviewed in this notebook, provide a structured analysis:
# MAGIC
# MAGIC 1. Class imbalance: What ratio did you observe between retained and churned customers?
# MAGIC    What modeling strategy should we use to prevent the model from ignoring the minority class?
# MAGIC
# MAGIC 2. Feature priorities: Name the top 3 features most correlated with churn based on the heatmap.
# MAGIC    For each, briefly explain why it might be a churn predictor (business reasoning).
# MAGIC
# MAGIC 3. Feature engineering plan: Are there any categorical features that need encoding?
# MAGIC    Which numerical features may need scaling?
# MAGIC    Suggest the appropriate scikit-learn transformers for each type.
# MAGIC
# MAGIC 4. Evaluation strategy: Given the class imbalance, which metric should be our primary focus —
# MAGIC    accuracy, precision, recall, or F1? Explain your choice in the context of churn prediction.
# MAGIC
# MAGIC Do NOT generate code. Provide a structured, actionable plan that we will execute in the next phase.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyIterate() {
# MAGIC   const text = document.getElementById("copy-btn-ite").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-iterate");
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

# MAGIC %md
# MAGIC > 💡 **What just happened:** Genie Code reasoned over the EDA output — it named specific features from your heatmap, applied business context to explain *why* those features predict churn, and recommended a concrete modeling strategy. This is not autocomplete. It's an agent synthesizing observations from prior steps and turning them into an actionable plan. Keep this response visible in the chat panel as you move into Phase 3 — the pipeline you're about to build should directly reflect what Genie Code recommended here.
# MAGIC
# MAGIC > 🚀 **Try this:** If Genie Code's recommendation differs from what you expected, ask it directly: *"Why did you recommend Random Forest over Logistic Regression for this task?"* — it will explain its reasoning based on the class imbalance and feature types it observed. This kind of follow-up question is one of the most powerful ways to deepen your understanding of both ML concepts and agentic reasoning.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Phase 3 — Build: Feature Engineering Pipeline
# MAGIC
# MAGIC We now execute the plan Genie Code recommended in Phase 2: encode categoricals, scale numericals, and split the data — all in a single prompt that produces a complete scikit-learn preprocessing pipeline.
# MAGIC
# MAGIC > 💡 **Context continuity:** Notice this prompt references `churn_pdf` — a variable Genie Code loaded two sections ago. We do not need to re-describe the dataset. Genie Code maintains a working understanding of notebook state across all prompts in the session.
# MAGIC
# MAGIC > 🪄 **Prompt simplicity tip:** Not sure how to phrase this yourself? Try asking Genie Code to write the prompt for you first:
# MAGIC >
# MAGIC > *"I have my churn data loaded as `churn_pdf`. Help me write a prompt I can use to ask Genie Code to prep the features and split the data so I can train a model to predict who will churn using best practices."*
# MAGIC >
# MAGIC > Genie Code will return a structured, requirements-aware prompt you can refine and run. The detailed prompt below is provided as a reference implementation — your own version may be shorter and still produce excellent results.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open the <strong>Genie</strong> panel (right sidebar → AI icon)</li>
# MAGIC <li>Switch the mode drop-down to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press <strong>Enter</strong> — Genie Code will generate and insert a new cell</li>
# MAGIC <li>Review the generated code, then run the cell</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC <button id="copy-btn-fe" onclick="copyFE()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-bt" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: I have churn_pdf in memory. The task is binary classification to predict customer churn.
# MAGIC Target column: Churn (Yes = churned, No = retained) — encode as 1/0.
# MAGIC
# MAGIC Task: Prepare the features for model training using a scikit-learn preprocessing pipeline.
# MAGIC
# MAGIC Requirements:
# MAGIC - Drop CustomerID (not a predictive feature)
# MAGIC - Encode the target: Churn Yes=1, No=0 → store as pandas Series called y
# MAGIC - Store the feature DataFrame as X (all columns except CustomerID, Churn, Churn_binary if it exists)
# MAGIC - Automatically identify categorical columns (object/category dtype) and numerical columns
# MAGIC - Build a scikit-learn ColumnTransformer named 'preprocessor':
# MAGIC     - OneHotEncoder(handle_unknown='ignore', sparse_output=False) for all categorical columns
# MAGIC     - StandardScaler() for all numerical columns
# MAGIC - Perform a stratified 80/20 train/test split (random_state=42) → X_train, X_test, y_train, y_test
# MAGIC - Print the shapes of each split and confirm class balance is preserved in both sets
# MAGIC
# MAGIC Expected output: A preprocessor object and four train/test splits, ready for model training.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyFE() {
# MAGIC   const text = document.getElementById("copy-bt").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-fe");
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

# MAGIC %md
# MAGIC > 🔍 **What to observe:**
# MAGIC - The `ColumnTransformer` applies different transformations per column type — matching the recommendation from Phase 2.
# MAGIC - Stratified split ensures class proportions are preserved in both train and test sets — crucial for imbalanced data.
# MAGIC - The `preprocessor` object is not yet fitted — it will be fitted inside the model pipeline during training.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Model Training with MLflow
# MAGIC
# MAGIC This is the core of Phase 3: a complete, MLflow-tracked training run from a single prompt.
# MAGIC
# MAGIC > After running the cell, navigate to **Experiments** in the left nav to confirm the run appeared. Open the run and explore the logged parameters, metrics, and model artifact — then return to continue.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open the <strong>Genie</strong> panel (right sidebar → AI icon)</li>
# MAGIC <li>Switch the mode drop-down to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press <strong>Enter</strong> — Genie Code will generate and insert a new cell</li>
# MAGIC <li>Review the generated code, then run the cell</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC <button id="copy-btn-train" onclick="copyTrain()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-btn-t" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: X_train, X_test, y_train, y_test, and preprocessor are all defined in this notebook.
# MAGIC The task is binary classification (churn prediction) on a class-imbalanced dataset.
# MAGIC
# MAGIC Task: Train a Random Forest classifier wrapped in a full scikit-learn Pipeline, fully logged to MLflow.
# MAGIC
# MAGIC Requirements:
# MAGIC - Use class_weight='balanced' to address the class imbalance identified in EDA
# MAGIC - Build a Pipeline with two steps: ('preprocessor', preprocessor) and ('classifier', RandomForestClassifier)
# MAGIC - MLflow configuration:
# MAGIC     * Set experiment to: /Users/{DA.username}/churn_prediction_genie
# MAGIC     * Enable Databricks Autologging: mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
# MAGIC     * Explicitly log n_estimators, max_depth, random_state, and class_weight as parameters
# MAGIC     * Explicitly log train_accuracy and test_accuracy as metrics
# MAGIC     * Log the fitted pipeline as a model artifact named 'churn_rf_model'
# MAGIC     * Store run.info.run_id in a variable called run_id
# MAGIC - Use: n_estimators=100, max_depth=10, random_state=42
# MAGIC
# MAGIC After training, print the run_id, train accuracy, and test accuracy with clear labels.
# MAGIC
# MAGIC Expected output: A trained rf_pipeline object and a confirmed MLflow run with all parameters, metrics, and the model artifact logged.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyTrain() {
# MAGIC   const text = document.getElementById("copy-btn-t").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-train");
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

# MAGIC %md
# MAGIC > 💡 **Take a moment to explore the MLflow UI:** Navigate to **Experiments → churn_prediction_genie** and open the run that just appeared. You'll see every parameter, every metric, and the model artifact — the exact same MLflow structure you built by hand in Module 1. The rigor hasn't changed; what changed is that Genie Code eliminated the boilerplate. This is an important thing to internalize: agentic ML doesn't remove your responsibility for tracking and reproducibility — it removes the repetitive scaffolding so you can focus on the decisions that matter.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Phase 4 — Validate: Model Evaluation
# MAGIC
# MAGIC We evaluate the classifier with **two sequential prompts**. The first generates metrics and visualizations. The second asks Genie Code to *reason* about the results and recommend a concrete improvement.
# MAGIC
# MAGIC Run both prompts in order. The second prompt requires no code cell — Genie Code responds in the chat.
# MAGIC
# MAGIC > 💡 **Key insight for churn models:** Overall accuracy is misleading on imbalanced data. **Recall on class 1** (churned customers) is the priority metric. A model that misses 40% of churners has real business cost — even if overall accuracy is 85%.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open the <strong>Genie</strong> panel (right sidebar → AI icon)</li>
# MAGIC <li>Switch the mode drop-down to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press <strong>Enter</strong> — Genie Code will generate and insert a new cell</li>
# MAGIC <li>Review the generated code, then run the cell</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC <button id="copy-btn-eval1" onclick="copyEval1()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-btn-e" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: rf_pipeline has been trained and X_test, y_test are available in this notebook.
# MAGIC
# MAGIC Task: Generate a comprehensive evaluation of the Random Forest classifier.
# MAGIC
# MAGIC Requirements:
# MAGIC 1. Generate predictions: y_pred = rf_pipeline.predict(X_test)
# MAGIC    Generate probabilities: y_prob = rf_pipeline.predict_proba(X_test)[:, 1]
# MAGIC
# MAGIC 2. Plot a confusion matrix using ConfusionMatrixDisplay
# MAGIC    - Display labels: ['Retained (0)', 'Churned (1)']
# MAGIC    - Use Blues colormap
# MAGIC    - Add a descriptive title
# MAGIC
# MAGIC 3. Print a full classification report with precision, recall, F1-score per class and macro/weighted averages
# MAGIC
# MAGIC 4. Calculate and print the ROC-AUC score
# MAGIC
# MAGIC All metrics must be clearly labeled. Use matplotlib for the confusion matrix plot.
# MAGIC
# MAGIC Expected output: A confusion matrix plot, a classification report, and the ROC-AUC score printed below it.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyEval1() {
# MAGIC   const text = document.getElementById("copy-btn-e").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-eval1");
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

# MAGIC %md-sandbox
# MAGIC ### Genie Code Prompt 2 — Interpret Results & Recommend
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 14px 18px; border-radius: 4px; margin: 10px 0 16px 0;">
# MAGIC   <strong style="color: #0d47a1;">Why this prompt is different:</strong>
# MAGIC   <p style="margin: 6px 0 0 0; color: #333;">This prompt asks for <em>reasoning and a concrete recommendation</em> — not code generation. Genie Code will look at the metrics it just produced, draw conclusions, and suggest an improvement. This is the defining behavior of an AI agent vs. a code assistant.</p>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Make sure Genie Code is still in <strong>Agent Mode</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press <strong>Enter</strong> — Genie Code responds in the chat panel (no new code cell)</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC
# MAGIC <button id="copy-btn-eval2" onclick="copyEval2()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-btn-eval" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: I have a trained Random Forest churn classifier evaluated on a class-imbalanced test set.
# MAGIC
# MAGIC Based on the confusion matrix and classification report you just generated, answer these three questions:
# MAGIC
# MAGIC 1. Churn detection: What is the recall for class 1 (Churned customers)?
# MAGIC    In business terms, what fraction of actual churners is the model catching?
# MAGIC    Is this acceptable for a churn prevention use case?
# MAGIC
# MAGIC 2. Class imbalance impact: Looking at the precision/recall difference between class 0 and class 1,
# MAGIC    is the class_weight='balanced' parameter working effectively? How can you tell?
# MAGIC
# MAGIC 3. Concrete next step: Given these specific metric values, what is ONE actionable improvement
# MAGIC    you would recommend — be specific about the technique (e.g., threshold tuning, SMOTE,
# MAGIC    a different algorithm) and explain why it fits this dataset.
# MAGIC
# MAGIC Reference the actual numbers from the classification report in your answer.
# MAGIC Do NOT generate code unless asked.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyEval2() {
# MAGIC   const text = document.getElementById("copy-btn-eval").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-eval2");
# MAGIC   navigator.clipboard.writeText(text).then(() => {
# MAGIC     btn.innerText = '✅ Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   }).catch(() => {
# MAGIC     const ta = document.createElement('textarea'); ta.value = text;
# MAGIC     ta.style.position = 'fixed'; ta.style.left = '-9999px';
# MAGIC     document.body.appendChild(ta); ta.focus(); ta.select();
# MAGIC     document.execCommand('copy'); document.body.removeChild(ta);
# MAGIC     btn.innerText = '✅ Copied!'; setTimeout(() => { btn.innerText = 'Copy Genie Prompt'; }, 2000);
# MAGIC   });
# MAGIC }
# MAGIC </script>

# COMMAND ----------

# MAGIC %md
# MAGIC > 💡 **What just happened:** Genie Code reasoned over its own output — it cited specific recall values from the classification report, interpreted their business meaning (how many actual churners the model is catching), and recommended a concrete improvement technique. This is an agent operating across the full loop: generate code → observe results → reason about outcomes → prescribe a next step. This self-correcting, iterative pattern is the core of agentic ML, and it mirrors how experienced practitioners actually work.
# MAGIC
# MAGIC > 🚀 **Try this:** Ask Genie Code: *"Implement your suggestion."* — It will immediately scaffold the improvement based on the recommendation it just made. There's no need to write a new prompt from scratch. This illustrates how the agent carries its own reasoning forward — each turn builds on the last.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### Deployment Preparation — Model Registration
# MAGIC
# MAGIC The model is trained, evaluated, and tracked in MLflow. The final step: register it to **Unity Catalog** with a production alias — making it governed, versioned, and addressable by a serving endpoint.
# MAGIC
# MAGIC > ⚠️ **Endpoint deployment** is covered in the **Lab (Lesson 16)**. In the lab, you will use Genie Code to configure and launch a real-time serving endpoint from this registered model.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="border-left:3px solid #42a5f5; background:#e3f2fd; padding:10px 14px; border-radius:4px; margin:0 0 12px 0; font-size:0.88rem;">
# MAGIC <strong style="color:#0d47a1;">How to run this prompt:</strong>
# MAGIC <ol style="margin:4px 0 0 16px; color:#333; line-height:1.6;">
# MAGIC <li>Open the <strong>Genie</strong> panel (right sidebar → AI icon)</li>
# MAGIC <li>Switch the mode drop-down to <strong>Agent</strong></li>
# MAGIC <li>Click <strong>Copy Genie Prompt</strong> below and paste into the chat</li>
# MAGIC <li>Press <strong>Enter</strong> — Genie Code will generate and insert a new cell</li>
# MAGIC <li>Review the generated code, then run the cell</li>
# MAGIC </ol>
# MAGIC </div>
# MAGIC <button id="copy-btn-reg" onclick="copyReg()" style="background:#ff6b3d; color:white; border:none; padding:8px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-bottom:10px;">
# MAGIC Copy Genie Prompt
# MAGIC </button>
# MAGIC
# MAGIC <pre id="copy-btn" style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace; border:1px solid #e5e7eb; border-radius:10px; background:#f8fafc; padding:14px 16px; font-size:0.85rem; line-height:1.5; white-space:pre-wrap;">
# MAGIC Context: I have a completed MLflow training run. The run_id variable is defined in this notebook.
# MAGIC The model artifact is stored at path: runs:/{run_id}/churn_rf_model
# MAGIC
# MAGIC Task: Register this model to Unity Catalog with a production alias and provenance description.
# MAGIC
# MAGIC Requirements:
# MAGIC - Use mlflow.set_registry_uri("databricks-uc") to point to the Unity Catalog registry
# MAGIC - Register the model using mlflow.register_model() with:
# MAGIC     * model_uri = f"runs:/{run_id}/churn_rf_model"
# MAGIC     * name = f"{DA.catalog_name}.{DA.schema_name}.churn_rf_classifier"
# MAGIC - Use MlflowClient to:
# MAGIC     * Set alias "champion" on the registered version
# MAGIC     * Add this description: "Random Forest classifier for customer churn prediction.
# MAGIC       Trained via Genie Code (Agent Mode) as part of the ML Model Development course.
# MAGIC       Features: customer demographics, service plan, payment method, average monthly charges."
# MAGIC - Print confirmation of: registered model name, version number, and alias
# MAGIC
# MAGIC Expected output: Model appears in Unity Catalog → Models with version 1 and alias 'champion'.
# MAGIC </pre>
# MAGIC
# MAGIC <script>
# MAGIC function copyReg() {
# MAGIC   const text = document.getElementById("copy-btn").innerText;
# MAGIC   const btn = document.getElementById("copy-btn-reg");
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

# MAGIC %md
# MAGIC > 💡 **Why the `champion` alias matters:** Aliases are a Unity Catalog governance pattern for production model management. A Model Serving endpoint is configured to reference `@champion` by name — not by version number. This means when you train a better model and promote it to `champion`, the endpoint automatically serves the new version with no configuration changes required. It decouples deployment from versioning, which is essential in production ML systems where models are retrained regularly. In the lab, you'll deploy a real-time serving endpoint from this exact registered model using one more Genie Code prompt.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Demo Summary
# MAGIC
# MAGIC We completed an end-to-end ML workflow — driven entirely by structured natural language prompts in Genie Code Agent Mode.
# MAGIC
# MAGIC | Phase | What We Did | Prompts |
# MAGIC |-------|-------------|---------|
# MAGIC | **1 — Explore** | Data loading (Feature Store join), class distribution, null check, correlation heatmap | 4 |
# MAGIC | **2 — Iterate** | Synthesized EDA findings, got feature priorities and modeling strategy from Genie Code | 1 |
# MAGIC | **3 — Build** | Feature engineering pipeline (OneHotEncoder + StandardScaler), MLflow-tracked Random Forest | 2 |
# MAGIC | **4 — Validate** | Confusion matrix, classification report, ROC-AUC, interpretive analysis, UC registration | 3 |
# MAGIC
# MAGIC **What did not change:**
# MAGIC - MLflow experiment structure and full reproducibility
# MAGIC - Unity Catalog governance and feature lineage
# MAGIC - Model evaluation rigor and metric interpretation
# MAGIC - Your accountability for every decision made
# MAGIC
# MAGIC Genie Code removed the boilerplate. You remained in control of the intent.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2026 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>