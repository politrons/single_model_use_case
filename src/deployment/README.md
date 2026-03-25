# Deployment Module

## Overview

This module contains notebooks and utilities to deploy trained models either for **batch inference** or as **managed endpoints**.

## Notebooks

* **ModelDeployment.py** – Deploys a registered model to the target environment.
* **BatchInference.py** – Runs batch inference over a Delta table and writes predictions.

## Expected Input Parameters

The notebooks rely on the following *Databricks notebook widgets* (or equivalent job parameters).

| Parameter           | Description                                |
| ------------------- | ------------------------------------------ |
| `env`               | Target environment (dev / staging / prod). |
| `input_table_name`  | Unity Catalog table name.                  |
| `model_name`        | Name of the registered MLflow model.       |
| `output_table_name` | Unity Catalog table name.                  |

---

