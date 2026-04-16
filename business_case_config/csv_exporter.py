# Databricks notebook source
import csv
import logging
import re
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
workspace_max_rows_per_file = 1_000_000

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


def _normalize_csv_value(value):
    """
    Convert values to a CSV-safe scalar representation.
    """
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _write_workspace_csv_streaming(
    df_union,
    destination_dir: Path,
    prefix: str,
    max_rows_per_file: int = 1_000_000,
) -> list[Path]:
    """
    Write CSV files to Workspace Files using Python streaming.

    This avoids Spark file commit protocol issues on Workspace local FS.
    It also splits output into multiple files to avoid Workspace file size limits.
    """
    if max_rows_per_file <= 0:
        raise ValueError("max_rows_per_file must be > 0")

    destination_dir.mkdir(parents=True, exist_ok=True)
    columns = list(df_union.columns)
    logger.info("Streaming CSV rows for '%s' to %s", prefix, str(destination_dir))

    part_index = 1
    rows_in_part = 0
    total_rows = 0
    output_files: list[Path] = []
    current_file_path: Path | None = None
    current_handle = None
    current_writer = None

    def _open_new_part(file_index: int):
        file_path = destination_dir / f"{prefix}_payload_flat_part_{file_index:05d}.csv"
        fh = file_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(columns)
        return file_path, fh, writer

    try:
        current_file_path, current_handle, current_writer = _open_new_part(part_index)
        output_files.append(current_file_path)

        try:
            iterator = df_union.toLocalIterator()
        except Exception:
            logger.warning("toLocalIterator() failed for '%s', falling back to collect()", prefix)
            iterator = df_union.collect()

        for row in iterator:
            if rows_in_part >= max_rows_per_file:
                current_handle.close()
                part_index += 1
                rows_in_part = 0
                current_file_path, current_handle, current_writer = _open_new_part(part_index)
                output_files.append(current_file_path)

            values = [_normalize_csv_value(row[col]) for col in columns]
            try:
                current_writer.writerow(values)
            except OSError as error:
                if getattr(error, "errno", None) == 27:
                    logger.warning(
                        "File too large for '%s' in %s. Rolling over to next part.",
                        prefix,
                        str(current_file_path),
                    )
                    current_handle.close()
                    part_index += 1
                    rows_in_part = 0
                    current_file_path, current_handle, current_writer = _open_new_part(part_index)
                    output_files.append(current_file_path)
                    current_writer.writerow(values)
                else:
                    raise

            rows_in_part += 1
            total_rows += 1
    finally:
        if current_handle is not None and not current_handle.closed:
            current_handle.close()

    logger.info(
        "CSV export completed for '%s': %d rows, %d file(s)",
        prefix,
        total_rows,
        len(output_files),
    )
    return output_files


def _export_to_workspace_files(df_union, prefix: str) -> None:
    """
    Export CSV to Workspace Files under the configured user directory.
    """
    destination_dir = Path(workspace_output_dir) / prefix
    files = _write_workspace_csv_streaming(
        df_union=df_union,
        destination_dir=destination_dir,
        prefix=prefix,
        max_rows_per_file=workspace_max_rows_per_file,
    )

    logger.info("CSV file(s) saved to Workspace Files folder: %s", str(destination_dir))

    links_html = []
    for file_path in files:
        download_url = _workspace_download_url(str(file_path))
        links_html.append(
            f'<a href="{download_url}" target="_blank" download>{file_path.name}</a>'
        )

    links_block = "<br/>".join(links_html)
    displayHTML(
        f"""
    <div style="padding:8px;border:1px solid #ddd;border-radius:8px;">
      <b>{prefix.upper()} CSV file(s) ready</b><br/>
      {links_block}<br/>
      <small>Folder: {destination_dir}</small>
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
