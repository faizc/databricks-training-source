# Databricks notebook source
import os

def check_serverless_runtime(required_version: str) -> bool:
    """
    Validates that the notebook is running on Serverless Compute 
    with the recommended runtime version.

    Args:
        required_version (str): The expected Serverless runtime version (e.g., "5").

    Returns:
        bool: True if the environment meets the requirement, False otherwise.
    """
    
    is_serverless = os.environ.get("IS_SERVERLESS", "")
    runtime_version = os.environ.get("DATABRICKS_RUNTIME_VERSION", "")

    is_valid = (
        is_serverless == "TRUE" and 
        runtime_version.startswith(f"client.{required_version}.")
    )

    if is_valid:
        print(
            f"✅ Correct environment detected\n"
            f"Running on Serverless Compute (runtime version {required_version})."
        )
    else:
        print(
            f"⚠️ Recommended Environment Not Detected\n"
            f"This notebook is designed to run on Serverless Compute "
            f"(runtime version {required_version}).\n"
            f"Please switch to the appropriate Serverless environment to ensure full compatibility."
        )

    return is_valid