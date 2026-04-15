# Databricks notebook source
import logging
import re
import shutil
from functools import reduce
from pathlib import Path
from pyspark.sql.types import ArrayType, BinaryType, MapType, StructType

# COMMAND ----------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("payload_flat_export")

# Source catalog/schema
catalog = "your_catalog"
schema = "your_schema"

# Output target mode:
# - "workspace_files": save CSV in /Workspace/Users/...
# - "dbfs": save CSV in dbfs:/...
output_mode = "workspace_files"

# Your Workspace user folder (provided)
workspace_user_dir = "/Workspace/Users/pablo.garcia.external@axa-uk.co.uk"
workspace_output_dir = f"{workspace_user_dir}/exports"
workspace_tmp_dir = f"{workspace_user_dir}/.tmp_payload_flat_exports"

# DBFS fallback output directory
dbfs_output_dir = "dbfs:/tmp/payload_flat_exports"

# COMMAND ----------

def _workspace_download_url(workspace_file_path: str) -> str:
    """
    Convert '/Workspace/...' path to a browser URL.

    Databricks serves Workspace files under '/workspace-files/...'.
    """
    if not workspace_file_path.startswith("/Workspace/"):
        return workspace_file_path
    return "/workspace-files/" + workspace_file_path[len("/Workspace/"):]


def _write_single_csv_part(df_union, prefix: str) -> Path:
    """
    Write a single CSV part to a Workspace-local temp folder and return part file path.

    UC Shared access mode blocks writes to generic local paths like '/tmp'.
    Therefore all local staging must stay under '/Workspace/...'.
    """
    tmp_dir = Path(workspace_tmp_dir) / f"{prefix}_payload_flat_csv_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing temporary CSV part for '%s' to %s", prefix, str(tmp_dir))
    (
        df_union.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(f"file:{tmp_dir}")
    )

    part_files = sorted(p for p in tmp_dir.iterdir() if p.suffix == ".csv")
    if not part_files:
        raise RuntimeError(f"No CSV part file generated for prefix '{prefix}'")
    return part_files[0]


def _export_to_workspace_files(df_union, prefix: str) -> None:
    """
    Export CSV to Workspace Files under the configured user directory.
    """
    part_file = _write_single_csv_part(df_union, prefix)
    destination_dir = Path(workspace_output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_file = destination_dir / f"{prefix}_payload_flat.csv"

    shutil.copyfile(part_file, destination_file)
    logger.info("CSV saved to Workspace Files: %s", str(destination_file))

    download_url = _workspace_download_url(str(destination_file))
    logger.info("Download URL for '%s': %s", prefix, download_url)
    displayHTML(
        f"""
    <div style="padding:8px;border:1px solid #ddd;border-radius:8px;">
      <b>{prefix.upper()} CSV ready</b><br/>
      <a href="{download_url}" target="_blank" download>Download {prefix}_payload_flat.csv</a><br/>
      <small>Saved at: {destination_file}</small>
    </div>
    """
    )


def _export_to_dbfs(df_union, prefix: str) -> None:
    """
    Export CSV to DBFS path.
    """
    output_dir = f"{dbfs_output_dir}/{prefix}_payload_flat_csv"
    logger.info("Writing CSV for '%s' to DBFS path: %s", prefix, output_dir)
    (
        df_union.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(output_dir)
    )
    logger.info("DBFS export completed for '%s'", prefix)
    displayHTML(
        f"""
    <div style="padding:8px;border:1px solid #ddd;border-radius:8px;">
      <b>{prefix.upper()} CSV written to DBFS</b><br/>
      <small>Path: {output_dir}</small>
    </div>
    """
    )


def _drop_unsupported_csv_columns(df_union, prefix: str):
    """
    Drop columns that CSV datasource cannot serialize (nested/complex types).
    """
    unsupported_types = (MapType, StructType, ArrayType, BinaryType)
    unsupported = [
        (field.name, field.dataType.simpleString())
        for field in df_union.schema.fields
        if isinstance(field.dataType, unsupported_types)
    ]

    if not unsupported:
        return df_union

    unsupported_names = {name for name, _ in unsupported}
    logger.warning(
        "Dropping %d unsupported CSV columns for '%s': %s",
        len(unsupported),
        prefix,
        ", ".join([f"{name} ({dtype})" for name, dtype in unsupported]),
    )

    supported_columns = [field.name for field in df_union.schema.fields if field.name not in unsupported_names]
    if not supported_columns:
        raise ValueError(f"After dropping unsupported CSV columns, no columns remain for '{prefix}'.")

    return df_union.select(*supported_columns)


def export_payload_csv(prefix: str) -> None:
    """
    Discover tables by prefix, union all rows, preview in notebook, and export CSV.
    """
    pattern = re.compile(rf"^{prefix}_.*_payload_flat$")

    logger.info("Discovering tables for prefix '%s' in %s.%s", prefix, catalog, schema)
    tables = [
        row.tableName
        for row in spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
        if pattern.match(row.tableName)
    ]

    if not tables:
        logger.warning("No tables found for prefix '%s'", prefix)
        return

    logger.info("Found %d tables for '%s'", len(tables), prefix)
    dfs = [spark.table(f"{catalog}.{schema}.{table_name}") for table_name in tables]
    df_union = reduce(lambda left, right: left.unionByName(right, allowMissingColumns=True), dfs)
    df_csv = _drop_unsupported_csv_columns(df_union, prefix)

    logger.info("Showing preview for '%s'", prefix)
    display(df_csv.limit(200))

    if output_mode == "workspace_files":
        _export_to_workspace_files(df_csv, prefix)
    elif output_mode == "dbfs":
        _export_to_dbfs(df_csv, prefix)
    else:
        raise ValueError(f"Unsupported output_mode '{output_mode}'. Use 'workspace_files' or 'dbfs'.")


# COMMAND ----------

for prefix_name in ["ibnr", "inflation"]:
    export_payload_csv(prefix_name)
