# Databricks notebook source
import logging
import re
from functools import reduce

# COMMAND ----------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("payload_flat_export")

catalog = "your_catalog"
schema = "your_schema"

# COMMAND ----------

def export_payload_csv(prefix: str) -> None:
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

    dfs = [spark.table(f"{catalog}.{schema}.{table_name}") for table_name in tables]
    df_union = reduce(lambda left, right: left.unionByName(right, allowMissingColumns=True), dfs)

    logger.info("Showing preview for '%s'", prefix)
    display(df_union.limit(200))

    tmp_dir = f"dbfs:/tmp/{prefix}_payload_flat_csv_tmp"
    logger.info("Writing CSV to temp path: %s", tmp_dir)
    (
        df_union.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(tmp_dir)
    )

    part_file = next(file_info.path for file_info in dbutils.fs.ls(tmp_dir) if file_info.name.endswith(".csv"))
    final_dbfs_path = f"dbfs:/FileStore/exports/{prefix}_payload_flat.csv"
    dbutils.fs.cp(part_file, final_dbfs_path, True)

    download_url = f"/files/exports/{prefix}_payload_flat.csv"
    logger.info("Download URL for '%s': %s", prefix, download_url)
    displayHTML(f"""
    <div style="padding:8px;border:1px solid #ddd;border-radius:8px;">
      <b>{prefix.upper()} CSV ready</b><br/>
      <a href="{download_url}" target="_blank" download>Download {prefix}_payload_flat.csv</a>
    </div>
    """)

# COMMAND ----------

for prefix_name in ["ibnr", "inflation"]:
    export_payload_csv(prefix_name)
